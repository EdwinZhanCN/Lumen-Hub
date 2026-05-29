# Lumen Hub

A self-hosted, multi-model ML inference server in Rust. Every model runs natively
on [Burn](https://burn.dev) — the compute backend (CPU, Metal, Vulkan/wgpu, CUDA,
ROCm) is chosen at build time, so there are no external runtime libraries to ship.

Models are exposed over gRPC behind a uniform task API, with dynamic request
batching for tensor inputs.

## Models & tasks
car
| Service | Task | Input → Output |
|---|---|---|
| `siglip` | `semantic_text_embed` / `semantic_image_embed` | text / image → L2-normalized `embedding_v1` |
| `ppocr` | `ocr` | image → `ocr_v1` (boxes + recognized text) |
| `insightface` | `face_recognition` | image → `face_v1` (boxes, landmarks, 512-d embeddings) |
| `bioclip` | `bioclip_classify` | image → `labels_v1` top-k taxonomy labels |

Bundled architectures: SigLIP 2 (`base-patch16-224`, `so400m-patch14-384`),
PP-OCRv5, antelopev2 (SCRFD + ArcFace), and BioCLIP-2. BioCLIP classification runs
the vision encoder, then does HNSW ANN search + exact rerank over a precomputed
TreeOfLife taxon catalog.

## Quick start

```bash
lumen-cli init      # detect hardware, pick a backend + preset, write config
lumen-cli start     # download the matching hub build + models, then run
```

Or run the hub directly against a config:

```bash
lumen-hub --config config.yaml
```

Models are fetched on first start into `metadata.cache_dir` from
`Lumilio-Photos/<model>` on Hugging Face (`region: cn` uses the hf-mirror).

### Config sketch

```yaml
metadata:   { version: "0.1.0", region: other, cache_dir: "~/.lumen/models" }
deployment: { mode: hub, services: [siglip, ocr, face] }
server:     { host: "0.0.0.0", port: 50051, batching: { enabled: true, max_batch_size: 8, queue_latency_ms: 2 } }
services:
  siglip:
    enabled: true
    package: siglip
    models:
      default: { model: siglip2-base-patch16-224, runtime: burn, precision: fp32 }
```

`runtime` is always `burn`; the compute backend is a build-time choice, not config.

## Build

```bash
cargo build --release                      # default: cpu backend + all models
cargo build --release --no-default-features --features metal,siglip,ppocr,insightface,clip
```

Backend features (pick one; priority cuda > rocm > vulkan > metal > wgpu > cpu):
`cpu` (ndarray), `metal`, `vulkan`, `wgpu`, `cuda`, `rocm`.
Model features: `siglip`, `ppocr`, `insightface`, `clip` (BioCLIP).

Requires Rust 1.94+ (pinned in `rust-toolchain.toml`).

## Develop & test

```bash
cargo test --workspace                     # unit/integration (E2E skip if models absent)

# End-to-end against real weights (set LUMEN_MODELS_DIR to the model repo root):
cargo test --release --test e2e_siglip --test e2e_ppocr --test e2e_insightface --test e2e_bioclip
cargo test --release --features metal --test e2e_siglip   # same, on Metal
```

E2E tests load FP32 weights from `LUMEN_MODELS_DIR` (default
`/Volumes/CodeBase/Projects/lumen-models`) and skip gracefully when absent.

## Release

```bash
cargo xtask dist --profile linux-x64-gpu   # build + package one profile to dist/
cargo xtask release-metadata               # write manifest.json + checksums.txt
```

Profiles: `{darwin-arm64,windows-x64,linux-x64,linux-arm64}` × backend, e.g.
`darwin-arm64-metal`, `linux-x64-{cpu,gpu,cuda,rocm}`, `linux-arm64-{cpu,gpu,jetson}`
(`*-gpu` = wgpu/Vulkan; `jetson` = arm64 CUDA on L4T). CI builds every hosted
profile and runs the test suite; `.github/workflows/release.yml` produces the
signed artifacts + installer on tag push.

## Workspace layout

```
crates/
  lumen-hub/      # the server: models/, model_arch/ (generated Burn graphs),
                  #   service/ + daemon/ (gRPC, batching), backend.rs
  lumen-schema/   # config + result schemas (embedding_v1, ocr_v1, face_v1, labels_v1)
  lumen-cli/      # installer/launcher (hardware detect, backend select, download)
  xtask/          # dist packaging + release metadata
```

Adding a model variant = drop a generated arch under `model_arch/<id>/` and add one
match arm in the matching `models::<family>::model` dispatcher.

## License

MIT © Edwin Zhan
