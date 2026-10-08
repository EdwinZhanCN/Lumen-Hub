"""Generates the committed qa-tiny IREE fixture used by the lumen-hub L0 suites.

qa-tiny: embedding = l2_normalize(pixels[1, 3072] @ W[3072, 16] + b[16]),
with the same integer-derived weights the Burn QA model used
(crates/lumen-hub/src/models/qa/model.rs, `deterministic_value`).

Output (committed under fixtures/iree/qa-tiny/ at the repository root):
  model_info.json
  iree/net.{fp32,w8a32}.irpa
  iree/net.fp32.{cpu-x86_64,cpu-aarch64,metal-macos}.vmfb
  iree/net.w8a32.{cpu-x86_64,cpu-aarch64}.vmfb
  iree/BUILD.fp32.json, iree/BUILD.w8a32.json
  expected.json   canonical input definition + onnxruntime reference outputs
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .constants import ONNX_OPSET, PRECISION_FP32, PRECISION_W8A32, TARGETS_BY_PRECISION
from .pipeline import convert
from .recipes import load_recipe

MODEL = "qa-tiny"
INPUT_NUMEL = 3 * 32 * 32
EMBED_DIM = 16
# The published target set of each precision minus cuda-sm_75 (CI has no NVIDIA GPU).
FIXTURE_TARGETS = {
    precision: [t for t in targets if t != "cuda-sm_75"] for precision, targets in TARGETS_BY_PRECISION.items()
}
U64 = (1 << 64) - 1


def deterministic_value(row: int, col: int) -> np.float32:
    """Bit-identical to the Rust `deterministic_value` (usize wrapping math, f32)."""
    h = (((row * 31) & U64) + ((col * 17) & U64)) & U64
    h %= 197
    return np.float32((np.float32(h) / np.float32(197.0) - np.float32(0.5)) * np.float32(0.1))


def build_onnx(path: Path) -> None:
    weight = np.empty((INPUT_NUMEL, EMBED_DIM), dtype=np.float32)
    for row in range(INPUT_NUMEL):
        for col in range(EMBED_DIM):
            weight[row, col] = deterministic_value(row, col)
    bias_row = U64 // 2  # Rust: usize::MAX / 2 on 64-bit targets
    bias = np.array([deterministic_value(bias_row, col) for col in range(EMBED_DIM)], dtype=np.float32)
    nodes = [
        helper.make_node("MatMul", ["pixels", "proj.weight"], ["mm"]),
        helper.make_node("Add", ["mm", "proj.bias"], ["y"]),
        helper.make_node("Mul", ["y", "y"], ["y2"]),
        helper.make_node("ReduceSum", ["y2", "axes"], ["ss"], keepdims=1),
        helper.make_node("Sqrt", ["ss"], ["norm"]),
        helper.make_node("Max", ["norm", "eps"], ["norm_c"]),
        helper.make_node("Div", ["y", "norm_c"], ["embedding"]),
    ]
    graph = helper.make_graph(
        nodes,
        "qa_tiny",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, INPUT_NUMEL])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, EMBED_DIM])],
        [
            numpy_helper.from_array(weight, "proj.weight"),
            numpy_helper.from_array(bias, "proj.bias"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.array(1e-12, dtype=np.float32), "eps"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))


def canonical_input() -> np.ndarray:
    """pixels[i] = (i % 251) / 251, the input used by the Rust L0 tests."""
    values = (np.arange(INPUT_NUMEL, dtype=np.int64) % 251).astype(np.float32) / np.float32(251.0)
    return values.reshape(1, INPUT_NUMEL)


def calibration_inputs() -> np.ndarray:
    base = canonical_input()[0]
    return np.stack([np.roll(base, shift) for shift in range(0, 8 * 97, 97)]).astype(np.float32)


def generate(out_dir: Path, recipe_path: Path) -> None:
    import onnxruntime as ort

    base_recipe = load_recipe(recipe_path)
    if base_recipe.model != MODEL or base_recipe.precision != PRECISION_FP32:
        raise ValueError(f"{recipe_path} must be the fp32 {MODEL} recipe")
    w8_recipe = dataclasses.replace(
        base_recipe,
        precision=PRECISION_W8A32,
        components=tuple(
            dataclasses.replace(c, quantize=True, min_quantized_cosine=0.999) for c in base_recipe.components
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "qa-tiny.fp32.onnx"
        build_onnx(source)
        inputs = tmp / "inputs.npz"
        np.savez(inputs, pixels=calibration_inputs())
        build_root = tmp / "out"
        for recipe in (base_recipe, w8_recipe):
            convert(
                recipe,
                {"net": source},
                build_root,
                FIXTURE_TARGETS[recipe.precision],
                model_info_path=None,
                keep_work=recipe is w8_recipe,
                real_inputs={"net": inputs},
            )
        # Reference outputs straight from the graphs that were imported.
        x = canonical_input()
        work = build_root / ".work" / MODEL / "net"
        fp32_ref = ort.InferenceSession(str(work / "net.main.fp32.onnx")).run(None, {"pixels": x})[0]
        w8_ref = ort.InferenceSession(str(work / "net.main.w8a32.onnx")).run(None, {"pixels": x})[0]

        target = out_dir / MODEL
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(build_root / MODEL, target)

    model_info = {
        "name": MODEL,
        "version": "1.0.0",
        "description": "Tiny deterministic QA model for the e2e harness.",
        "model_type": "qa",
        "source": {"format": "huggingface", "repo_id": f"Lumilio-Photos/{MODEL}"},
        "runtimes": {
            "iree": {
                "available": True,
                "components": ["net"],
                "precisions": [PRECISION_FP32, PRECISION_W8A32],
            }
        },
    }
    (target / "model_info.json").write_text(json.dumps(model_info, indent=2) + "\n")
    expected = {
        "input": "pixels[0][i] = (i % 251) / 251 for i in 0..3072",
        "input_shape": [1, INPUT_NUMEL],
        "function": "main",
        "outputs": {
            PRECISION_FP32: [float(v) for v in fp32_ref.ravel()],
            PRECISION_W8A32: [float(v) for v in w8_ref.ravel()],
        },
    }
    (target / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
