"""W8A32 weight quantization.

Rule (applied to one prepared fp32 graph):

* Eligible weights are initializers with rank >= 2 and at least
  MIN_QUANT_ELEMENTS elements that are consumed ONLY as
    - input 1 of Conv            -> one scale per output channel (axis 0)
    - input 1 of ConvTranspose   -> one scale per output channel (axis 1)
    - input 1 of MatMul (rank 2) -> one scale per output column  (axis 1)
    - input 1 of Gemm            -> axis 0 if transB == 1 else axis 1
    - input 0 of Gather (rank 2, Gather axis 0) -> one scale per row (axis 0)
  and every consumer agrees on the same axis. Anything else stays float32.
* Values: symmetric int8 in [-127, 127], q = round_half_even(w / s),
  s = max(|w| over the channel) / 127 (s = 1.0 for an all-zero channel),
  zero point 0.
* Each quantized initializer W is replaced by initializers W__w8q (int8),
  W__w8s (float32 scales) and W__w8z (int8 zeros) feeding one
  DequantizeLinear node (attribute axis) whose output W__w8dq replaces W.

IREE 3.12 keeps these weights int8 in memory and dequantizes inside the
consuming dispatch; block-wise DequantizeLinear (opset 21 block_size) is not
supported by its importer, which is why the scale granularity is per channel.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper

MIN_QUANT_ELEMENTS = 4096


def _attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _axis_for_use(node: onnx.NodeProto, input_index: int, rank: int) -> int | None:
    op = node.op_type
    if input_index == 1 and op == "Conv":
        return 0
    if input_index == 1 and op == "ConvTranspose":
        return 1
    if input_index == 1 and op == "MatMul" and rank == 2:
        return 1
    if input_index == 1 and op == "Gemm":
        return 0 if _attr(node, "transB", 0) == 1 else 1
    if input_index == 0 and op == "Gather" and rank == 2 and _attr(node, "axis", 0) == 0:
        return 0
    return None


def quantize_tensor(weight: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    weight = weight.astype(np.float32, copy=False)
    reduce_axes = tuple(i for i in range(weight.ndim) if i != axis)
    max_abs = np.abs(weight).max(axis=reduce_axes)
    scale = np.where(max_abs > 0, max_abs / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    shape = [1] * weight.ndim
    shape[axis] = -1
    q = np.clip(np.rint(weight / scale.reshape(shape)), -127, 127).astype(np.int8)
    return q, scale


def quantize_model(model: onnx.ModelProto) -> list[str]:
    """Quantizes eligible weights in place; returns the quantized initializer names
    (sorted). `model` must have its external data loaded."""
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    uses: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in initializers:
                uses[name].append((node, index))
    graph_outputs = {value.name for value in graph.output}

    plan: dict[str, int] = {}
    for name, init in initializers.items():
        rank = len(init.dims)
        numel = int(np.prod(init.dims)) if init.dims else 1
        if rank < 2 or numel < MIN_QUANT_ELEMENTS or name in graph_outputs:
            continue
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        axes = {_axis_for_use(node, index, rank) for node, index in uses.get(name, [])}
        if not uses.get(name) or None in axes or len(axes) != 1:
            continue
        plan[name] = axes.pop()

    dq_nodes = []
    for name in sorted(plan):
        axis = plan[name]
        init = initializers[name]
        q, scale = quantize_tensor(numpy_helper.to_array(init), axis)
        graph.initializer.extend(
            [
                numpy_helper.from_array(q, name + "__w8q"),
                numpy_helper.from_array(scale, name + "__w8s"),
                numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int8), name + "__w8z"),
            ]
        )
        dq_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [name + "__w8q", name + "__w8s", name + "__w8z"],
                [name + "__w8dq"],
                name=name + "__w8dqnode",
                axis=axis,
            )
        )
        for node, index in uses[name]:
            node.input[index] = name + "__w8dq"
        graph.initializer.remove(init)

    nodes = dq_nodes + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return sorted(plan)
