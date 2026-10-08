# Lumen IREE toolchain

Turns the maintainer's fp32 ONNX graphs into verified IREE artifact sets for
lumen-hub, and generates the committed `fixtures/iree/qa-tiny` test fixture.
The normative description (precision policy, targets, entry points, gates) is
`docs/lumen-hub-iree-migration-plan.md`; this file is only the quick start.

## Setup

```bash
cd tools/iree
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt          # exact pins; IREE must stay 3.12.0
python -m unittest discover -s tests     # self-tests (~10 s)
python -m lumen_iree_tools check-recipes
```

## Convert one model

```bash
python -m lumen_iree_tools convert \
    --recipe recipes/siglip2-base-patch16-224.json \
    --onnx vision=/path/vision.fp32.onnx \
    --onnx text=/path/text.fp32.onnx \
    --onnx aesthetic=/path/aesthetic.fp32.onnx \
    --inputs vision=/path/vision.npz --inputs text=/path/text.npz \
    --model-info /path/model_info.json \
    --out out --targets all
```

- `--onnx`: exactly one fp32 graph per recipe component (files above 2 GiB must
  use ONNX external data next to the `.onnx`).
- `--inputs`: real preprocessed tensors for every quantized component, produced
  by `lumen-hub dump-inputs` and stacked with
  `python -m lumen_iree_tools pack-inputs --out vision.npz dump/vision/*.npy`.
- `--targets all` builds every target published for the recipe precision
  (`fp32`: cpu-x86_64, cpu-aarch64, cuda-sm_75, metal-macos; `w8a32`: the same
  without metal-macos).

The result `out/<model>/` mirrors the Hugging Face repository layout:
`model_info.json` and `iree/<component>.<precision>.{irpa,<target>.vmfb}` plus
`iree/BUILD.<precision>.json` with hashes and the verification report. The
command fails, and writes no `model_info.json`, if any shape check or
verification gate fails.

## Other commands

```bash
python -m lumen_iree_tools reference --recipe recipes/<model>.json --component vision \
    --onnx /path/vision.fp32.onnx --inputs bus.npz --out <model>.vision.bus.fp32-reference.json
python -m lumen_iree_tools qa-fixture --out ../../fixtures/iree
```

## Layout

| Path | Content |
|---|---|
| `lumen_iree_tools/constants.py` | IREE pin, precision tags, targets and their compiler flags |
| `lumen_iree_tools/recipes.py` | recipe schema and validation |
| `lumen_iree_tools/onnx_prep.py` | per-entry static-shape normalization |
| `lumen_iree_tools/quantize.py` | the W8A32 rule |
| `lumen_iree_tools/importer.py` | multi-entry import with shared parameters |
| `lumen_iree_tools/pipeline.py` | `convert` |
| `lumen_iree_tools/verify.py` | parity and quality gates |
| `lumen_iree_tools/qa_fixture.py` | qa-tiny fixture generator |
| `recipes/*.json` | one recipe per model |
| `tests/` | self-tests |
