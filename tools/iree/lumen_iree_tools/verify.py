"""Verification of a freshly converted artifact set on the host CPU.

Gates (all must pass; `convert` aborts otherwise):

  parity   For every component and every entry: the IREE host-CPU module and
           onnxruntime, both running the exact graph that was imported
           (quantized graph for quantized components), agree with cosine
           similarity >= PARITY_MIN_COSINE on every output, for
           SYNTHETIC_SAMPLES deterministic synthetic inputs.

  quality  For quantized components only: the IREE output vs onnxruntime on
           the prepared fp32 graph has cosine >= the recipe's
           min_quantized_cosine for every sample of the REAL inputs supplied
           with --inputs <component>=<file.npz>. Real inputs are mandatory for
           quantized components.

GPU targets are not executed here; they are covered by the lumen-hub L1 suites.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np

from .constants import PARAM_SCOPE

PARITY_MIN_COSINE = 0.9999
SYNTHETIC_SAMPLES = 2
TOKEN_ID_RANGE = (1, 1000)


class VerifyError(RuntimeError):
    pass


def host_cpu_target() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "cpu-x86_64"
    if machine in ("arm64", "aarch64"):
        return "cpu-aarch64"
    raise VerifyError(f"unsupported host machine {machine}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    return float(a @ b / (na * nb + 1e-300))


def synthetic_inputs(seed_text: str, dtype: str, shape: tuple[int, ...], count: int) -> list[np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if dtype == "float32":
        return [rng.uniform(-1.0, 1.0, size=shape).astype(np.float32) for _ in range(count)]
    return [rng.integers(TOKEN_ID_RANGE[0], TOKEN_ID_RANGE[1], size=shape, dtype=np.int64) for _ in range(count)]


class IreeModule:
    def __init__(self, vmfb: Path, irpa: Path):
        import iree.runtime as rt

        self.config = rt.Config("local-task")
        index = rt.ParameterIndex()
        index.load(str(irpa))
        modules = [
            rt.create_io_parameters_module(self.config.vm_instance, index.create_provider(scope=PARAM_SCOPE)),
            rt.create_hal_module(self.config.vm_instance, self.config.device),
            rt.VmModule.mmap(self.config.vm_instance, str(vmfb)),
        ]
        self.context = rt.SystemContext(vm_modules=modules, config=self.config)
        self.functions = {n for n in modules[-1].function_names if not n.endswith("$async") and not n.startswith("__")}

    def run(self, function: str, x: np.ndarray) -> list[np.ndarray]:
        if function not in self.functions:
            raise VerifyError(f"module has no function {function}")
        result = self.context.modules.module[function](x)
        results = result if isinstance(result, (list, tuple)) else [result]
        return [np.asarray(r.to_host()) for r in results]


def _ort(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def verify_component(
    model: str,
    precision: str,
    component,
    entries: list[tuple[str, Path]],
    fp32_entries: dict[str, Path],
    vmfb: Path,
    irpa: Path,
    real_inputs: Path | None,
) -> dict:
    target = host_cpu_target()
    module = IreeModule(vmfb, irpa)
    report = {"target": target, "entries": {}}
    if component.quantize and real_inputs is None:
        raise VerifyError(f"{component.name}: quantized components need --inputs {component.name}=<npz>")
    for entry_name, final_path in entries:
        shape = tuple(next(e.shape for e in component.entries if e.name == entry_name))
        imported = _ort(final_path)
        input_name = imported.get_inputs()[0].name
        worst_parity = 1.0
        for x in synthetic_inputs(f"{model}/{component.name}/{entry_name}", component.input_dtype, shape, SYNTHETIC_SAMPLES):
            want = imported.run(None, {input_name: x})
            got = module.run(entry_name, x)
            if len(got) != len(want):
                raise VerifyError(f"{component.name}.{entry_name}: {len(got)} outputs, onnxruntime has {len(want)}")
            for w, g in zip(want, got):
                if w.shape != g.shape:
                    raise VerifyError(f"{component.name}.{entry_name}: output shape {g.shape} != {w.shape}")
                worst_parity = min(worst_parity, cosine(w, g))
        if worst_parity < PARITY_MIN_COSINE:
            raise VerifyError(f"{component.name}.{entry_name}: parity cosine {worst_parity:.6f} < {PARITY_MIN_COSINE}")
        entry_report = {"parity_min_cosine": worst_parity}
        if component.quantize:
            data = np.load(real_inputs)
            if len(data.files) != 1:
                raise VerifyError(f"{real_inputs}: expected exactly one array")
            samples = data[data.files[0]]
            if tuple(samples.shape[1:]) != shape[1:] or samples.shape[0] < 1:
                raise VerifyError(f"{real_inputs}: array shape {samples.shape} does not match entry {shape}")
            reference = _ort(fp32_entries[entry_name])
            worst_quality = 1.0
            for i in range(samples.shape[0]):
                dtype = np.float32 if component.input_dtype == "float32" else np.int64
                x = np.ascontiguousarray(samples[i : i + 1]).astype(dtype)
                want = reference.run(None, {reference.get_inputs()[0].name: x})
                got = module.run(entry_name, x)
                for w, g in zip(want, got):
                    worst_quality = min(worst_quality, cosine(w, g))
            if worst_quality < component.min_quantized_cosine:
                raise VerifyError(
                    f"{component.name}.{entry_name}: quantized cosine {worst_quality:.5f} < {component.min_quantized_cosine}"
                )
            entry_report["quality_min_cosine"] = worst_quality
            entry_report["quality_samples"] = int(samples.shape[0])
        report["entries"][entry_name] = entry_report
    return report
