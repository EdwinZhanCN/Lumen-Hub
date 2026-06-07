# lumen-convert

Offline toolchain that turns **fp32 ONNX** models into the artifacts lumen-hub
loads at runtime:

1. the generated Burn architecture (`Model<B>` Rust, synced into
   `lumen-hub/src/model_arch/`), and
2. a **weight-only int8** burnpack (`{component}.int8.bpk`) for low-memory
   inference.

ONNX is the single controllable source of truth; everything downstream is
generated. This crate is **not** part of the lumen-hub runtime build.

## Pipeline

```
raw fp32 .onnx
   │  tools/onnx_prep.py   (uv, PEP 723)   opset->21 + shape inference
   ▼
onnx/<model-id>/<component>.prepared.onnx
   │  build.rs  (burn_onnx::ModelGen)      ONNX -> Model<B> .rs + fp32 .bpk
   ▼
   │  src/bin/quantize.rs  (SelectiveQuantizer)
   ▼
out/<model-id>/burn/<component>.int8.bpk   (+ accuracy report vs fp32)
```

## 1. ONNX file contract (what you drop in)

Per model component, place the fp32 ONNX at:

```
crates/lumen-convert/onnx/<model-id>/<component>.onnx
```

| model-id | components |
|---|---|
| `siglip2-base-patch16-224`   | `text`, `vision` |
| `siglip2-so400m-patch14-384` | `text`, `vision` |
| `bioclip-2`                  | `vision` (`text` optional) |
| `pp-ocrv5`                   | `detection`, `recognition` |
| `antelopev2`                 | `detection`, `recognition` |

Rules:
- **fp32** graph, no pre-inserted QDQ nodes.
- Single `<component>.onnx` when weights < 2 GB; otherwise `<component>.onnx`
  **plus** its external-data file, whose name must match the `location` field
  embedded in the proto (e.g. `<component>.onnx_data`).
- Opset is handled for you — `onnx_prep.py` upgrades to 21 and runs shape
  inference. If your exporter already emits opset 21 it's idempotent.

## 2. Prepare the ONNX

```sh
# one file
uv run tools/onnx_prep.py onnx/siglip2-base-patch16-224/vision.onnx \
                          onnx/siglip2-base-patch16-224/vision.prepared.onnx

# or the whole tree
uv run tools/onnx_prep.py --tree onnx/
```

## 3. Generate + quantize (Rust)

```sh
cargo run -p lumen-convert --bin quantize          # all models
cargo run -p lumen-convert --bin quantize -- siglip2-base-patch16-224
```

This regenerates the Burn arch (synced into `lumen-hub/src/model_arch/`),
produces `out/<model-id>/burn/<component>.int8.bpk`, and prints the fp32-vs-int8
cosine for each component.

## 4. Graduate arch into the runtime

`build.rs` generates the Burn `Model<B>` into `OUT_DIR`. To run a model in
lumen-hub, sync that arch into the committed `model_arch` with the `conv_fwd`
pointwise-conv patch applied (the same workaround the existing mobile arch
carries — routes 1×1 convs through matmul to dodge a burn-wgpu/metal bug):

```sh
cargo build -p lumen-convert            # runs ModelGen into OUT_DIR
tools/sync_arch.sh pp_ocrv5_server detection recognition classification
```

This writes `lumen-hub/src/model_arch/<repo>/<component>.rs` + `mod.rs`. The only
edit to the generated code is `self.conv2dN.forward(x)` →
`crate::model_arch::conv_fwd(&self.conv2dN, x)` (mathematically equivalent,
idempotent). Then register `pub mod <repo>;` in `model_arch/mod.rs` and add a
dispatch arm in `models/<family>/model.rs`.

## Quantization scheme

Weight-only, symmetric **Q8** (int8), **per-block** along the last axis with an
adaptive block size (largest divisor of the last dim ≤ cap). Only weights inside
`Linear` / `Conv*` / `Embedding` modules (rank ≥ 2) are quantized; biases,
norm params, and bare graph constants stay fp32. Validated on SigLIP-base vision:

| block cap | cosine vs fp32 |
|---|---|
| per-tensor | 0.945 |
| ≤128 | 0.995 |
| ≤64  | 0.998 |
| ≤32  | 0.999 |

Default cap: **32** (cosine ~0.999, ~3.5× smaller than fp32). Each component is
validated against its own fp32 cosine at build time; the cap is a per-model knob.
