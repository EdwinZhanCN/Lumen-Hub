"""Normalizes a user-supplied fp32 ONNX graph into one static-shape graph per entry.

For every entry point the source graph is:
  1. checked to have exactly one runtime input of the recipe dtype,
  2. given the entry's static input shape,
  3. pruned to the recipe's `keep_outputs` (in that order),
  4. converted to opset ONNX_OPSET and IR version ONNX_IR_VERSION,
  5. given non-negative `Concat` axes (onnxruntime's symbolic shape inference
     only accepts axis 0 when it concatenates shape vectors, but exporters
     write the equivalent axis -1),
  6. run through onnxruntime's quantization pre-processing (symbolic shape
     inference + basic, standard-op-only graph optimizations),
  7. checked to contain only default-domain (ai.onnx) operators.

All intermediate files are written with external data so graphs above the
2 GiB protobuf limit (e.g. the SigLIP so400m text tower) work unchanged.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, version_converter
from onnx.external_data_helper import load_external_data_for_model

from .constants import ONNX_IR_VERSION, ONNX_OPSET, PARAM_MIN_ELEMENTS

# Tensors smaller than this many bytes stay inline in the .onnx protobuf; larger
# ones go to the sidecar file. Every tensor with fewer than PARAM_MIN_ELEMENTS
# elements (at most 8 bytes each) is therefore inline, which is what the IREE
# importer requires for the constants it embeds into the module.
EXTERNAL_DATA_THRESHOLD_BYTES = 1024
assert PARAM_MIN_ELEMENTS * 8 <= EXTERNAL_DATA_THRESHOLD_BYTES

DTYPES = {"float32": TensorProto.FLOAT, "int64": TensorProto.INT64}


class PrepError(ValueError):
    pass


def save_external(model: onnx.ModelProto, path: Path) -> None:
    """Saves `model` with every initializer in one sidecar `<name>.data` file."""
    data_name = path.name + ".data"
    if (path.parent / data_name).exists():
        (path.parent / data_name).unlink()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=EXTERNAL_DATA_THRESHOLD_BYTES,
    )


def load_full(path: Path) -> onnx.ModelProto:
    """Loads a model including its external data."""
    return onnx.load_model(str(path), load_external_data=True)


def runtime_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
    initializers = {init.name for init in model.graph.initializer}
    return [value for value in model.graph.input if value.name not in initializers]


def canonicalize_concat_axes(model: onnx.ModelProto) -> None:
    """Rewrites every negative `Concat` axis as `axis + rank` (same semantics)."""
    concats = [node for node in model.graph.node if node.op_type == "Concat"]
    if not any(attr.name == "axis" and attr.i < 0 for node in concats for attr in node.attribute):
        return
    inferred = onnx.shape_inference.infer_shapes(model)
    ranks: dict[str, int] = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("shape"):
            ranks[value.name] = len(tensor_type.shape.dim)
    for init in model.graph.initializer:
        ranks[init.name] = len(init.dims)
    for node in concats:
        for attr in node.attribute:
            if attr.name != "axis" or attr.i >= 0:
                continue
            rank = ranks.get(node.output[0])
            if rank is None:
                rank = next((ranks[name] for name in node.input if name in ranks), None)
            if rank is None:
                raise PrepError(f"Concat {node.name!r}: rank unknown, cannot normalize axis {attr.i}")
            attr.i += rank


def prepare_entry(
    src: Path,
    workdir: Path,
    tag: str,
    input_dtype: str,
    shape: tuple[int, ...],
    keep_outputs: tuple[int, ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> Path:
    """Returns the path of the prepared static-shape fp32 graph for one entry."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    workdir.mkdir(parents=True, exist_ok=True)
    model = onnx.load_model(str(src), load_external_data=False)

    # Old IR versions list initializers as graph inputs; drop those.
    initializers = {init.name for init in model.graph.initializer}
    kept_inputs = [value for value in model.graph.input if value.name not in initializers]
    del model.graph.input[:]
    model.graph.input.extend(kept_inputs)

    inputs = runtime_inputs(model)
    if len(inputs) != 1:
        raise PrepError(f"{src}: expected exactly one runtime input, found {[i.name for i in inputs]}")
    graph_input = inputs[0]
    elem_type = graph_input.type.tensor_type.elem_type
    if elem_type != DTYPES[input_dtype]:
        raise PrepError(
            f"{src}: input {graph_input.name} has ONNX elem_type {elem_type}, recipe expects {input_dtype}"
        )
    dims = graph_input.type.tensor_type.shape.dim
    if len(dims) != len(shape):
        raise PrepError(f"{src}: input rank {len(dims)} != recipe rank {len(shape)}")
    for dim, value in zip(dims, shape):
        if dim.HasField("dim_value") and dim.dim_value not in (0, value) and dim.dim_value > 0:
            raise PrepError(f"{src}: input dim fixed to {dim.dim_value}, recipe wants {value}")
        dim.Clear()
        dim.dim_value = value

    outputs = list(model.graph.output)
    if max(keep_outputs) >= len(outputs):
        raise PrepError(f"{src}: keep_outputs {keep_outputs} but graph has {len(outputs)} outputs")
    kept = [outputs[i] for i in keep_outputs]
    del model.graph.output[:]
    for value in kept:
        value.type.tensor_type.ClearField("shape")  # recomputed by shape inference
        model.graph.output.append(value)
    del model.graph.value_info[:]

    default_opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    if default_opset is None:
        raise PrepError(f"{src}: graph has no default-domain opset import")
    if default_opset > ONNX_OPSET:
        raise PrepError(f"{src}: opset {default_opset} is newer than the pinned opset {ONNX_OPSET}")
    if default_opset < ONNX_OPSET:
        model = version_converter.convert_version(model, ONNX_OPSET)
    model.ir_version = ONNX_IR_VERSION
    canonicalize_concat_axes(model)

    # The graph edits above never touch tensor payloads; pull them in from the
    # source directory now and write one self-contained file set.
    load_external_data_for_model(model, str(src.parent))
    stage0 = workdir / f"{tag}.stage0.onnx"
    save_external(model, stage0)
    del model

    prepared = workdir / f"{tag}.fp32.onnx"
    quant_pre_process(
        str(stage0),
        str(prepared),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        external_data_location=prepared.name + ".data",
        external_data_size_threshold=EXTERNAL_DATA_THRESHOLD_BYTES,
    )
    check_prepared(prepared, input_dtype, shape, output_shapes)
    return prepared


def check_prepared(
    path: Path, input_dtype: str, shape: tuple[int, ...], output_shapes: tuple[tuple[int, ...], ...]
) -> None:
    model = onnx.load_model(str(path), load_external_data=False)
    foreign = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    if foreign:
        raise PrepError(f"{path}: non-standard operator domains after preprocessing: {foreign}")
    inputs = runtime_inputs(model)
    if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != DTYPES[input_dtype]:
        raise PrepError(f"{path}: unexpected inputs after preprocessing")
    got = tuple(d.dim_value for d in inputs[0].type.tensor_type.shape.dim)
    if got != shape:
        raise PrepError(f"{path}: input shape {got} != {shape}")
    if len(model.graph.output) != len(output_shapes):
        raise PrepError(f"{path}: {len(model.graph.output)} outputs, expected {len(output_shapes)}")
    for output, expected in zip(model.graph.output, output_shapes):
        dims = output.type.tensor_type.shape.dim
        if not dims or any(not d.HasField("dim_value") for d in dims):
            raise PrepError(f"{path}: output {output.name} does not have a fully static shape")
        got = tuple(d.dim_value for d in dims)
        if got != tuple(expected):
            raise PrepError(f"{path}: output {output.name} has shape {got}, recipe expects {tuple(expected)}")
