"""Command line entry point: python -m lumen_iree_tools <command> ...

Commands
  convert       fp32 ONNX components -> verified IREE artifact set for one model
  pack-inputs   stack preprocessed .npy tensors dumped by lumen-hub into one .npz
  reference     fp32 onnxruntime reference outputs (JSON) for golden/L1 tests
  qa-fixture    regenerate the committed qa-tiny fixture for the L0 suites
  check-recipes validate every recipe under recipes/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .constants import ALL_TARGETS, TARGETS_BY_PRECISION

TOOLS_DIR = Path(__file__).resolve().parent.parent
RECIPES_DIR = TOOLS_DIR / "recipes"


def _pairs(values: list[str] | None, flag: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"{flag} expects <component>=<path>, got {value!r}")
        key, path = value.split("=", 1)
        if key in out:
            raise SystemExit(f"{flag} names component {key!r} twice")
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"{flag}: {p} does not exist")
        out[key] = p
    return out


def cmd_convert(args: argparse.Namespace) -> None:
    from .pipeline import convert
    from .recipes import load_recipe

    recipe = load_recipe(args.recipe)
    if args.targets == "all":
        targets = list(TARGETS_BY_PRECISION[recipe.precision])
    else:
        targets = [t for t in args.targets.split(",") if t]
    model_dir = convert(
        recipe,
        _pairs(args.onnx, "--onnx"),
        Path(args.out).expanduser().resolve(),
        targets,
        Path(args.model_info).expanduser().resolve() if args.model_info else None,
        args.keep_work,
        _pairs(args.inputs, "--inputs"),
    )
    print(model_dir)


def cmd_pack_inputs(args: argparse.Namespace) -> None:
    arrays = [np.load(path) for path in args.npy]
    if not arrays:
        raise SystemExit("no .npy files given")
    shape, dtype = arrays[0].shape, arrays[0].dtype
    for path, array in zip(args.npy, arrays):
        if array.shape != shape or array.dtype != dtype:
            raise SystemExit(f"{path}: {array.shape}/{array.dtype} differs from {shape}/{dtype}")
    stacked = np.concatenate(arrays, axis=0) if shape and shape[0] == 1 else np.stack(arrays)
    np.savez(args.out, inputs=stacked)
    print(f"{args.out}: {stacked.shape} {stacked.dtype}")


def cmd_reference(args: argparse.Namespace) -> None:
    import json
    import tempfile

    import onnxruntime as ort

    from .onnx_prep import prepare_entry
    from .recipes import load_recipe, resolve_output_shape

    recipe = load_recipe(args.recipe)
    component = recipe.component(args.component)
    entry = next((e for e in component.entries if e.name == args.entry), None)
    if entry is None:
        raise SystemExit(f"{component.name} has no entry {args.entry}")
    source = _pairs([f"{component.name}={args.onnx}"], "--onnx")[component.name]
    data = np.load(args.inputs)
    if len(data.files) != 1:
        raise SystemExit(f"{args.inputs}: expected exactly one array")
    samples = data[data.files[0]]
    dtype = np.float32 if component.input_dtype == "float32" else np.int64
    with tempfile.TemporaryDirectory() as tmp:
        prepared = prepare_entry(
            source,
            Path(tmp),
            f"{component.name}.{entry.name}",
            component.input_dtype,
            entry.shape,
            component.keep_outputs,
            tuple(resolve_output_shape(s, entry.shape) for s in component.output_shapes),
        )
        session = ort.InferenceSession(str(prepared), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        results = []
        for i in range(samples.shape[0]):
            x = np.ascontiguousarray(samples[i : i + 1]).astype(dtype)
            if tuple(x.shape) != entry.shape:
                raise SystemExit(f"sample {i} has shape {x.shape}, entry needs {entry.shape}")
            outputs = session.run(None, {name: x})
            results.append([{"shape": list(o.shape), "values": [float(v) for v in o.ravel()]} for o in outputs])
    document = {
        "model": recipe.model,
        "component": component.name,
        "entry": entry.name,
        "reference": "fp32 onnxruntime (CPUExecutionProvider) on the prepared graph",
        "samples": results,
    }
    Path(args.out).write_text(json.dumps(document) + "\n")
    print(f"{args.out}: {len(results)} samples")


def cmd_qa_fixture(args: argparse.Namespace) -> None:
    from .qa_fixture import generate

    generate(Path(args.out).expanduser().resolve(), RECIPES_DIR / "qa-tiny.json")


def cmd_check_recipes(_: argparse.Namespace) -> None:
    from .recipes import load_recipe

    for path in sorted(RECIPES_DIR.glob("*.json")):
        recipe = load_recipe(path)
        entries = sum(len(c.entries) for c in recipe.components)
        print(f"{path.name}: {recipe.model} {recipe.precision} components={len(recipe.components)} entries={entries}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lumen_iree_tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert one model")
    p.add_argument("--recipe", required=True)
    p.add_argument("--onnx", action="append", required=True, metavar="COMPONENT=PATH")
    p.add_argument("--out", required=True)
    p.add_argument("--targets", default="all", help=f"'all' (every target published for the recipe precision) or a comma list of {', '.join(ALL_TARGETS)}")
    p.add_argument("--model-info", help="existing model_info.json; an updated copy is written to the output")
    p.add_argument("--inputs", action="append", metavar="COMPONENT=NPZ", help="real preprocessed inputs (required for quantized components)")
    p.add_argument("--keep-work", action="store_true", help="keep intermediate graphs under <out>/.work")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("pack-inputs", help="stack .npy dumps into one .npz")
    p.add_argument("--out", required=True)
    p.add_argument("npy", nargs="+")
    p.set_defaults(func=cmd_pack_inputs)

    p = sub.add_parser("reference", help="fp32 onnxruntime reference outputs for tests")
    p.add_argument("--recipe", required=True)
    p.add_argument("--component", required=True)
    p.add_argument("--entry", default="main")
    p.add_argument("--onnx", required=True, help="fp32 ONNX of the component")
    p.add_argument("--inputs", required=True, help=".npz with one [N, ...] array (pack-inputs output)")
    p.add_argument("--out", required=True, help="output .json")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("qa-fixture", help="regenerate the qa-tiny fixture")
    p.add_argument("--out", required=True, help="fixture root: <repo>/fixtures/iree")
    p.set_defaults(func=cmd_qa_fixture)

    p = sub.add_parser("check-recipes", help="validate all recipes")
    p.set_defaults(func=cmd_check_recipes)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
