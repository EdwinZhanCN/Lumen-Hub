# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx>=1.17.0", "onnxsim>=0.4.36", "onnxruntime>=1.18"]
# ///
"""Normalize an fp32 ONNX model for burn-onnx import.

Pipeline: load -> upgrade opset to a target (default 21) -> shape inference ->
save (external data when the model exceeds the 2GB protobuf limit). This is the
"controllable source" gate for the Lumen int8 toolchain: burn-onnx's ModelGen
consumes the *prepared* ONNX produced here.

Usage (single file):
  uv run tools/onnx_prep.py in.onnx out.onnx
  uv run tools/onnx_prep.py in.onnx out.onnx --opset 21

Large models (>2GB) keep external data automatically; pass the .onnx whose
sibling data file is referenced by the proto's `location` fields.

Batch a whole `onnx/<model-id>/<component>.onnx` tree in place:
  uv run tools/onnx_prep.py --tree onnx/ --suffix .prepared
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import shape_inference, version_converter

# protobuf hard limit; above this onnx must use external data + path-based APIs.
TWO_GB = 2 * 1024 * 1024 * 1024


def _opset(model: onnx.ModelProto) -> int:
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    return 0


def resolve_auto_pad(model: onnx.ModelProto) -> int:
    """Rewrite `auto_pad=SAME_UPPER/SAME_LOWER` to explicit `pads`.

    burn-onnx cannot codegen SAME padding with dynamic input shapes (common in
    PaddleOCR exports). For stride-1 convolutions the SAME pad is independent of
    input size, so converting to explicit symmetric pads is exact and preserves
    dynamic input. Stride>1 SAME with dynamic shapes can't be resolved statically
    and is left in place (and warned about).
    """
    inits = {i.name: i for i in model.graph.initializer}
    conv = ("Conv", "ConvTranspose")
    pool = ("MaxPool", "AveragePool")
    changed = 0
    for node in model.graph.node:
        if node.op_type not in conv + pool:
            continue
        attrs = {a.name: a for a in node.attribute}
        ap = attrs.get("auto_pad")
        if ap is None or ap.s.decode() in ("NOTSET", "VALID"):
            continue
        mode = ap.s.decode()
        if "kernel_shape" in attrs:
            kshape = list(attrs["kernel_shape"].ints)
        elif node.op_type in conv:
            weight = inits.get(node.input[1])
            if weight is None:
                raise SystemExit(f"cannot resolve kernel shape for {node.name}")
            kshape = list(weight.dims[2:])
        else:
            raise SystemExit(f"{node.op_type} {node.name} missing kernel_shape")
        ndim = len(kshape)
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1] * ndim
        dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1] * ndim
        if any(s != 1 for s in strides):
            print(
                f"  WARN: {node.name or node.op_type} {mode} stride={strides} with dynamic "
                f"input cannot be made static; left as-is",
                file=sys.stderr,
            )
            continue
        begins, ends = [], []
        for k, d in zip(kshape, dilations):
            total = d * (k - 1)
            lo = total // 2
            hi = total - lo
            if mode == "SAME_UPPER":
                begins.append(lo)
                ends.append(hi)
            else:  # SAME_LOWER puts the extra pad at the start
                begins.append(hi)
                ends.append(lo)
        node.attribute.remove(ap)
        for a in list(node.attribute):
            if a.name == "pads":
                node.attribute.remove(a)
        node.attribute.append(onnx.helper.make_attribute("pads", begins + ends))
        changed += 1
    if changed:
        print(f"  auto_pad SAME -> explicit pads on {changed} conv(s)")
    return changed


def set_static_input(model: onnx.ModelProto, shape: list[int]) -> None:
    """Pin the first graph input to a fixed shape.

    burn-onnx can't codegen models whose shape-computation subgraphs depend on a
    dynamic input (it emits invalid Rust that indexes an int tensor). Fully
    convolutional classifiers run at a fixed crop size anyway, so pinning the input
    lets the shape subgraph constant-fold away.
    """
    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    del dims[:]
    for s in shape:
        dims.add().dim_value = s
    del model.graph.value_info[:]  # force re-inference of intermediate shapes
    print(f"  static  input {inp.name} = {shape}")


def prepare(src: Path, dst: Path, target_opset: int, input_shape: list[int] | None = None, simplify: bool = False) -> None:
    print(f"  load    {src}")
    # load_external_data=True (default) resolves sibling data files next to `src`.
    model = onnx.load(str(src))
    current = _opset(model)
    print(f"  opset   {current} -> {target_opset}")

    if current != target_opset:
        try:
            model = version_converter.convert_version(model, target_opset)
        except Exception as exc:  # noqa: BLE001 - report and stop, don't silently ship wrong opset
            print(
                f"  ERROR: opset {current}->{target_opset} conversion failed: {exc}\n"
                f"         re-export from the source framework at opset {target_opset}.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    resolve_auto_pad(model)

    if input_shape is not None:
        set_static_input(model, input_shape)

    if simplify:
        # Constant-folds dynamic Shape/Slice/Concat reshape subgraphs that burn-onnx
        # can't codegen. Requires a static input shape to fold fully.
        from onnxsim import simplify as onnxsim_simplify

        print("  simplify (onnxsim)")
        model, ok = onnxsim_simplify(model)
        if not ok:
            raise SystemExit("onnxsim simplification failed validation")

    print("  infer   shapes")
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    onnx.checker.check_model(model)

    dst.parent.mkdir(parents=True, exist_ok=True)
    size = model.ByteSize()
    if size >= TWO_GB:
        data_file = dst.name + "_data"
        print(f"  save    {dst} (+external data {data_file}, model is {size/1e9:.1f} GB)")
        onnx.save_model(
            model,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            convert_attribute=False,
        )
    else:
        print(f"  save    {dst}")
        onnx.save_model(model, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", type=Path, help="input .onnx")
    ap.add_argument("output", nargs="?", type=Path, help="output .onnx")
    ap.add_argument("--opset", type=int, default=21, help="target opset (default 21)")
    ap.add_argument("--tree", type=Path, help="batch every .onnx under this dir")
    ap.add_argument("--suffix", default=".prepared", help="suffix for --tree outputs")
    ap.add_argument(
        "--input-shape",
        help="pin the first input to a static shape, e.g. 1,3,80,160 "
        "(needed for fully-conv classifiers that burn-onnx can't codegen dynamically)",
    )
    ap.add_argument(
        "--simplify",
        action="store_true",
        help="run onnxsim to constant-fold dynamic shape subgraphs (pair with --input-shape)",
    )
    args = ap.parse_args()
    input_shape = [int(x) for x in args.input_shape.split(",")] if args.input_shape else None

    if args.tree:
        for src in sorted(args.tree.rglob("*.onnx")):
            if src.stem.endswith(args.suffix):
                continue
            dst = src.with_name(f"{src.stem}{args.suffix}.onnx")
            print(f"== {src.relative_to(args.tree)} ==")
            prepare(src, dst, args.opset)
        return

    if not args.input or not args.output:
        ap.error("provide input and output, or use --tree")
    prepare(args.input, args.output, args.opset, input_shape, args.simplify)
    print("done.")


if __name__ == "__main__":
    main()
