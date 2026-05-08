# /// script
# dependencies = ["onnx"]
# ///

import sys

import onnx

CANDLE_ONNX_OPS = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Exp",
    "Log",
    "Sqrt",
    "Equal",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "Not",
    "And",
    "Or",
    "Xor",
    "MatMul",
    "Gemm",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Concat",
    "Split",
    "Slice",
    "Expand",
    "Tile",
    "Gather",
    "GatherElements",
    "ScatterND",
    "Shape",
    "Size",
    "Range",
    "Where",
    "Clip",
    "Cast",
    "Constant",
    "ConstantOfShape",
    "Identity",
    "Conv",
    "MaxPool",
    "AveragePool",
    "BatchNormalization",
    "Softmax",
    "LogSoftmax",
    "Abs",
    "Cos",
    "Sin",
    "Neg",
    "Erf",
    "Tanh",
    "Sigmoid",
    "Gelu",
    "Relu",
    "PRelu",
    "LeakyRelu",
    "Selu",
    "HardSwish",
    "Ceil",
    "Floor",
    "Sign",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceSum",
    "ReduceL2",
    "ArgMin",
    "ArgMax",
    "Dropout",
    "CumSum",
    "If",
    "Pad",
    "RandomUniform",
    "RandomNormal",
    "LSTM",
    "RNN",
    "OneHot",
    "Resize",
    "Trilu",
}

if len(sys.argv) != 2:
    print(f"Usage: uv run --with onnx {sys.argv[0]} model.onnx")
    sys.exit(2)

model_path = sys.argv[1]
model = onnx.load(model_path)

model_ops = sorted({node.op_type for node in model.graph.node})

supported = [op for op in model_ops if op in CANDLE_ONNX_OPS]
missing = [op for op in model_ops if op not in CANDLE_ONNX_OPS]

print("Model ops:")
for op in model_ops:
    mark = "✅" if op in CANDLE_ONNX_OPS else "❌"
    print(f"{mark} {op}")

print()
print(f"Supported: {len(supported)} / {len(model_ops)}")

if missing:
    print()
    print("Missing / unsupported by candle-onnx:")
    for op in missing:
        print(f"- {op}")
    sys.exit(1)

print()
print("All model ops are listed in candle-onnx supported ops.")
sys.exit(0)
