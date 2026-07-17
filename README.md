# Lumen Hub

A self-hosted, multi-model ML inference server in Rust. Every model runs natively
on [Burn](https://burn.dev) — the compute backend (CPU, Metal, Vulkan/wgpu, CUDA,
ROCm) is chosen at build time, so there are no external runtime libraries to ship.

Models are exposed over gRPC behind a uniform task API, with dynamic request
batching for tensor inputs.

## Models & tasks

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

Pick the path that matches the machine the hub will run on:

| Machine | Path |
|---|---|
| Same PC as Lumilio Photos | Enable AI from the Lumilio Photos desktop app (it downloads and supervises the hub for you) |
| Linux server / NAS | Docker — see below |
| A spare Mac/Windows/Linux box on your LAN | `lumen-cli` — see below |
| Anything else / scripting | Bare binary: `lumen-hub --config config.yaml` |

**Docker** (tags: `cpu` = any machine, `vulkan` = Intel iGPU/AMD, `cuda` = NVIDIA;
details and compose file in [`packaging/docker/`](packaging/docker/README.md)):

```bash
docker run -d -p 50051:50051 -v lumen-models:/models ghcr.io/edwinzhancn/lumen-hub:cpu
```

**CLI** (detects hardware, picks a backend + preset, writes config, downloads
the matching hub build, runs it):

```bash
lumen-cli init
lumen-cli start
```

Models are fetched on first start into `metadata.cache_dir` from
`Lumilio-Photos/<model>` on Hugging Face (`region: cn` uses the hf-mirror).

### Config sketch

```yaml
metadata:   { version: "0.1.0", region: other, cache_dir: "~/.lumen/models" }
deployment: { mode: hub, services: [siglip, face] }
server:     { host: "0.0.0.0", port: 50051, batching: { enabled: true, max_batch_size: 8, queue_latency_ms: 2 } }
services:
  siglip:
    enabled: true
    package: siglip
    models:
      default: { model: siglip2-base-patch16-224, runtime: burn, precision: fp16q8 }
```

`runtime` is always `burn`; the compute backend is a build-time choice, not config.

## Control plane & health

The gRPC port binds immediately on startup — before models are downloaded or
loaded — and serves three services side by side:

- `home_native.v1.Inference` — the data plane (frozen contract). Returns
  `UNAVAILABLE` until the hub is ready.
- `lumen.control.v1.Control` (`proto/control.proto`) — read-only observability:
  `GetStatus` / `WatchStatus` stream the lifecycle phase (starting →
  downloading → loading → warmup → ready | failed), per-file download progress,
  per-service state, and the last fatal error; `TailLogs` streams structured
  log entries from an in-memory ring. Supervisors (the Lumilio Photos desktop
  app) use this instead of tailing log files.
- `grpc.health.v1.Health` — the standard health protocol (`NOT_SERVING` until
  warmup completes), so `grpc_health_probe`, Docker `HEALTHCHECK`, and k8s
  probes work out of the box.

If startup fails, the process stays up in `PHASE_FAILED` with health
`NOT_SERVING` so the error is queryable; mDNS is only advertised once ready.

## Build

```bash
cargo build --release                      # default: cpu backend + all models
cargo build --release --no-default-features --features metal,siglip,ppocr,insightface,clip
```

Backend features (pick one; priority cuda > rocm > vulkan > metal > wgpu > cpu):
`cpu` (Burn Flex), `metal`, `vulkan`, `wgpu`, `cuda`, `rocm`.
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
