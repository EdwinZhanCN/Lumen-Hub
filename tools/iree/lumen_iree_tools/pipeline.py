"""`convert`: fp32 ONNX components -> IREE artifact set for one model.

Output layout (mirrors the Hugging Face model repository):

  <out>/<model>/model_info.json            (only when --model-info is given)
  <out>/<model>/iree/<component>.<precision>.irpa
  <out>/<model>/iree/<component>.<precision>.<target>.vmfb
  <out>/<model>/iree/BUILD.<precision>.json (provenance + verification; not downloaded by the hub)
  <out>/.work/<model>/...                   (intermediates; safe to delete)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import onnx_prep
from .constants import (
    CPU_DATA_TILING_FLAG,
    IREE_VERSION,
    ONNX_OPSET,
    TARGETS,
    TARGETS_BY_PRECISION,
    irpa_name,
    vmfb_name,
)
from .importer import import_entries
from .quantize import quantize_model
from .recipes import Recipe, check_model_info, resolve_output_shape
from .verify import host_cpu_target, verify_component


class ConvertError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_toolchain() -> None:
    import iree.compiler
    import iree.runtime  # noqa: F401

    version = getattr(iree.compiler, "__version__", None) or _compiler_version()
    if not version.startswith(IREE_VERSION):
        raise ConvertError(f"iree-base-compiler {version} installed, {IREE_VERSION} required")


def _compiler_version() -> str:
    from importlib.metadata import version

    return version("iree-base-compiler")


def iree_compile_bin() -> str:
    from iree.compiler.tools.binaries import find_tool

    return find_tool("iree-compile")


def compile_module(mlir: Path, out: Path, target: str, cpu_data_tiling: bool) -> None:
    spec = TARGETS[target]
    cmd = [iree_compile_bin(), str(mlir), *spec.flags]
    if spec.is_cpu and cpu_data_tiling:
        cmd.append(CPU_DATA_TILING_FLAG)
    cmd += ["-o", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConvertError(f"iree-compile failed for {out.name}:\n{result.stderr[-4000:]}")


def prepare_component(
    recipe: Recipe, component, source: Path, workdir: Path
) -> tuple[list[tuple[str, Path]], dict[str, Path]]:
    """Returns ([(entry_name, final_onnx_path)], {entry_name: prepared_fp32_path})."""
    entries = []
    fp32 = {}
    for entry in component.entries:
        tag = f"{component.name}.{entry.name}"
        prepared = onnx_prep.prepare_entry(
            source,
            workdir,
            tag,
            component.input_dtype,
            entry.shape,
            component.keep_outputs,
            tuple(resolve_output_shape(s, entry.shape) for s in component.output_shapes),
        )
        fp32[entry.name] = prepared
        final = prepared
        if component.quantize:
            model = onnx_prep.load_full(prepared)
            quantized = quantize_model(model)
            if not quantized:
                raise ConvertError(f"{component.name}: W8A32 rule matched no weights")
            final = workdir / f"{tag}.{recipe.precision}.onnx"
            onnx_prep.save_external(model, final)
            (workdir / f"{tag}.quantized.json").write_text(json.dumps(quantized, indent=1))
        entries.append((entry.name, final))
    return entries, fp32


def convert(
    recipe: Recipe,
    sources: dict[str, Path],
    out_root: Path,
    targets: list[str],
    model_info_path: Path | None,
    keep_work: bool,
    real_inputs: dict[str, Path] | None = None,
) -> Path:
    real_inputs = real_inputs or {}
    check_toolchain()
    missing = [c.name for c in recipe.components if c.name not in sources]
    extra = [name for name in sources if name not in {c.name for c in recipe.components}]
    if missing or extra:
        raise ConvertError(f"--onnx must name exactly the recipe components; missing={missing} extra={extra}")
    allowed = TARGETS_BY_PRECISION[recipe.precision]
    for target in targets:
        if target not in TARGETS:
            raise ConvertError(f"unknown target {target}; known: {sorted(TARGETS)}")
        if target not in allowed:
            raise ConvertError(f"target {target} is not published for precision {recipe.precision}; allowed: {allowed}")

    model_info = None
    if model_info_path is not None:
        model_info = json.loads(model_info_path.read_text())
        if model_info.get("name") != recipe.model:
            raise ConvertError(f"model_info.json name {model_info.get('name')!r} != recipe model {recipe.model!r}")
        check_model_info(recipe, model_info)

    model_dir = out_root / recipe.model
    iree_dir = model_dir / "iree"
    workdir = out_root / ".work" / recipe.model
    if workdir.exists():
        shutil.rmtree(workdir)
    iree_dir.mkdir(parents=True, exist_ok=True)

    build = {
        "iree_version": IREE_VERSION,
        "onnx_opset": ONNX_OPSET,
        "recipe": recipe.path.name,
        "recipe_sha256": sha256_file(recipe.path),
        "precision": recipe.precision,
        "targets": targets,
        "components": {},
    }
    for component in recipe.components:
        source = sources[component.name]
        cwork = workdir / component.name
        print(f"[{recipe.model}] {component.name}: preparing {len(component.entries)} entr{'y' if len(component.entries) == 1 else 'ies'}", file=sys.stderr)
        entries, fp32_entries = prepare_component(recipe, component, source, cwork)
        mlir = cwork / f"{component.name}.{recipe.precision}.mlir"
        irpa = iree_dir / irpa_name(component.name, recipe.precision)
        param_count = import_entries(entries, mlir, irpa)
        files = {irpa.name: sha256_file(irpa)}
        for target in targets:
            vmfb = iree_dir / vmfb_name(component.name, recipe.precision, target)
            print(f"[{recipe.model}] {component.name}: compiling {target}", file=sys.stderr)
            compile_module(mlir, vmfb, target, recipe.cpu_data_tiling)
            files[vmfb.name] = sha256_file(vmfb)
        host = host_cpu_target()
        host_vmfb = iree_dir / vmfb_name(component.name, recipe.precision, host)
        if host not in targets:
            host_vmfb = cwork / vmfb_name(component.name, recipe.precision, host)
            compile_module(mlir, host_vmfb, host, recipe.cpu_data_tiling)
        print(f"[{recipe.model}] {component.name}: verifying on {host}", file=sys.stderr)
        verification = verify_component(
            recipe.model, recipe.precision, component, entries, fp32_entries,
            host_vmfb, irpa, real_inputs.get(component.name),
        )
        build["components"][component.name] = {
            "source_onnx_sha256": sha256_file(source),
            "entries": {entry.name: list(entry.shape) for entry in component.entries},
            "quantized": component.quantize,
            "parameters": param_count,
            "files": files,
            "verification": verification,
        }

    (iree_dir / f"BUILD.{recipe.precision}.json").write_text(json.dumps(build, indent=2) + "\n")
    if model_info is not None:
        runtimes = model_info.setdefault("runtimes", {})
        if sorted(targets) != sorted(allowed):
            raise ConvertError("model_info.json is only written for full artifact sets (--targets all)")
        # Exactly the existing RuntimeSpec fields: released hubs parse this file
        # with deny_unknown_fields, so no new keys may appear here.
        runtimes["iree"] = {
            "available": True,
            "components": [c.name for c in recipe.components],
            "precisions": [recipe.precision],
        }
        (model_dir / "model_info.json").write_text(json.dumps(model_info, indent=2, ensure_ascii=False) + "\n")
    if not keep_work:
        shutil.rmtree(workdir)
    return model_dir
