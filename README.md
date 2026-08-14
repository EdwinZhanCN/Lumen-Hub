# Lumen Hub

A self-hosted, multi-model ML inference server in Rust. Every model runs
natively on [Burn](https://burn.dev) — the compute backend (CPU, Metal,
Vulkan/wgpu, CUDA, ROCm) is chosen at build time, so there are no external
runtime libraries to ship. Models are exposed over gRPC behind a uniform task
API with dynamic request batching.

## Models & tasks

| Service | Task | Input → Output |
|---|---|---|
| `siglip` | `semantic_text_embed` / `semantic_image_embed` | text / image → `embedding_v1` |
| `ppocr` | `ocr` | image → `ocr_v1` (boxes + text) |
| `insightface` | `face_recognition` | image → `face_v1` (boxes, landmarks, embeddings) |
| `bioclip` | `bioclip_classify` | image → `labels_v1` top-k taxonomy labels |

Bundled: SigLIP 2, PP-OCRv5, antelopev2 (SCRFD + ArcFace), BioCLIP-2 (HNSW +
rerank over a TreeOfLife catalog).

## Quick start

| Machine | Path |
|---|---|
| Same PC as Lumilio Photos | Enable AI in the desktop app — it installs and supervises the hub |
| Linux server / NAS | Docker: `docker run -d --network host -v lumen-models:/models ghcr.io/edwinzhancn/lumen-hub:cpu` (host networking keeps LAN mDNS discovery working; tags `cpu`/`vulkan`/`cuda`; see [`packaging/docker/`](packaging/docker/README.md)) |
| Spare box on your LAN | `lumen-cli configure && lumen-cli start` (detects hardware, downloads the matching build) |
| Anything else | `lumen-hub --config config.yaml` |

Model weights download on first start into `metadata.cache_dir` from
`Lumilio-Photos/<model>` on Hugging Face (`region: cn` → hf-mirror).

```yaml
# config sketch — `runtime` is always burn; the compute backend is build-time
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

## Control plane & health

The gRPC port binds immediately — before models download — and serves:

- `home_native.v1.Inference` — data plane (frozen contract); `UNAVAILABLE` until ready.
- `lumen.control.v1.Control` ([`proto/control.proto`](crates/lumen-hub/proto/control.proto)) —
  read-only observability: `GetStatus`/`WatchStatus` (lifecycle phase, download
  progress, per-service state, last fatal error) and `TailLogs` (structured log
  ring). Supervisors use this instead of tailing files.
- `grpc.health.v1.Health` — `NOT_SERVING` until warmup completes; works with
  `grpc_health_probe`, Docker `HEALTHCHECK`, k8s probes.

Startup failure keeps the process serving (`PHASE_FAILED`, error queryable);
mDNS is advertised only once ready.

## Build & test

Requires Rust 1.94+ (pinned in `rust-toolchain.toml`) and
[just](https://github.com/casey/just) 1.58+ for the developer recipes
(`cargo binstall just@1.58.0` or `brew install just`).

```bash
just ci                     # fmt + workspace tests + config fixtures + L0 e2e
just check-backend metal    # compile lumen-hub for a backend + all models
just contract               # proto provenance + buf lint + daemon codegen drift

cargo build --release      # default: cpu backend + all models
cargo build --release --no-default-features --features metal,siglip,ppocr,insightface,clip

cargo test --workspace     # unit + integration

# L0 e2e (every PR in CI): real binary + mock model repo + a tiny deterministic
# QA model — lifecycle, control plane, batching, quantization. No downloads.
just l0

# L1 (nightly in CI): real weights — semantic checks + golden regression.
LUMEN_MODELS_DIR=/path/to/lumen-models just l1-backend cpu
just golden                # regenerate tests/golden/ after intentional changes
```

Backend features (pick one; priority cuda > rocm > vulkan > metal > wgpu > cpu):
`cpu`, `metal`, `vulkan`, `wgpu`, `cuda`, `rocm`. `metal`, `vulkan`, and `wgpu` are
the same `cubecl-wgpu` backend parameterized by which `GraphicsApi` it hands to
the `wgpu` crate (`Metal` / `Vulkan` / auto-detect) — not three separate
implementations. `wgpu` (auto) is what the dist profiles actually ship on
Linux/Windows; `vulkan` exists as an escape hatch to force that API instead of
relying on auto-detection. `metal` is required on macOS (no Vulkan there —
auto-detect resolves to Metal anyway, so the named feature just makes that
explicit). `cuda`/`rocm` are unrelated: they talk to the vendor compute API
directly (`cubecl-cuda`/`cubecl-hip`), bypassing the graphics-API layer entirely.
Model features: `siglip`, `ppocr`, `insightface`, `clip` (BioCLIP).

## Release

Tag `v<version>` (matching the `lumen-cli` crate version) →
`.github/workflows/release.yml` builds the continuously exercised release
profiles, the CLI installers, and the release catalog. The published hub
profiles are macOS arm64 Metal/CPU, Windows x64 wgpu/CPU, Linux x64
wgpu/CUDA/CPU, and Linux arm64 CPU. ROCm, generic Linux arm64 GPU, and Jetson
remain explicit source-build recipes and are not installer choices or release
artifacts.

- `manifest.json` — schema-versioned (`schemaVersion: 2`) catalog generated
  from `lumen-schema`: presets, model metadata, resource guidance, the profiles
  that actually have release artifacts, artifact URL/SHA-256, and protocol
  provenance (`dataPlaneMajor` + proto SHA-256s). Consumers should pin only the
  machine facts they execute; explanatory product copy remains local.
- `SHA256SUMS` — top-level digest file covering every asset including
  `manifest.json`.

Locally: `cargo xtask dist --profile <profile>`;
`cargo xtask release-metadata [--assets-dir <dir>]` regenerates the catalog.
Preset/custom config fixtures under `fixtures/config/` are the stable goldens
shared by CLI, launcher, and Docker; regenerate with
`just config-fixtures` (CI checks them via `just ci`).

Linux x64 profiles build on `ubuntu-22.04` (glibc 2.35), not 24.04
(glibc 2.39): glibc is forward-compatible only, so building on the oldest
maintained hosted runner used by this workflow maximizes which end-user
distros the shipped binary runs on. The arm64 CPU profile uses GitHub's native
arm64 runner and the same source-defined packaging contract.

## Workspace layout

```
crates/
  lumen-hub/         # the server: models/, model_arch/ (generated Burn graphs),
                     #   service/ + daemon/ (gRPC, batching), status.rs (control plane)
  lumen-schema/      # config + result schemas + release catalog + canonical render
                     #   (manifest.rs, config/render.rs, preset.rs)
  lumen-launcher/    # install/config/run library behind lumen-cli
  lumen-cli/         # end-user installer/launcher CLI
  lumen-quant-core/  # int8 weight-quantization primitives (runtime fp16q8 + offline)
  xtask/             # dist packaging + release metadata
```

Adding a model variant = drop a generated arch under `model_arch/<id>/` and add
one match arm in the matching `models::<family>::model` dispatcher.

## License

MIT © Edwin Zhan
