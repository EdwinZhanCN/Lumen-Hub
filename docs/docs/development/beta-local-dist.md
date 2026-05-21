---
sidebar_position: 2
---

# Beta Dist

Hub runtime profiles are built with `cargo xtask dist`. The public `lumen-cli` release archives and shell/PowerShell installers are built with `cargo-dist`.

## Profiles

| Dist profile | Backend route | Notes |
| --- | --- | --- |
| `universal-cpu` | ORT CPU | Portable CPU baseline. |
| `darwin-arm64` | ORT CPU/XNNPACK + MNN Metal | No CoreML, no Candle. |
| `linux-x64-cuda` | ORT CUDA | CUDA package is ORT-only for beta. MNN CUDA is not bundled. |
| `windows-x64-dml` | ORT DirectML + ORT CPU | No MNN. |
| `linux-x64-openvino` | ORT OpenVINO + dynamic ORT loading | Bundles OpenVINO-enabled ONNX Runtime libraries. |

All beta profiles include `clip`, `siglip`, `insightface`, and `ppocr`.

## Build

```bash
cargo xtask dist --profile universal-cpu
cargo xtask dist --profile darwin-arm64
cargo xtask dist --profile linux-x64-cuda
cargo xtask dist --profile windows-x64-dml
cargo xtask dist --profile linux-x64-openvino
```

CLI release assets are produced by the release workflow with:

```bash
dist build --artifacts=local --target <target> --tag <tag>
dist build --artifacts=global --tag <tag>
```

Each artifact is written under `dist/lumen-hub-<profile>/` with this layout:

```text
bin/
lib/
licenses/
README.md
checksums.txt
```

## Run

```bash
./bin/lumen-hub --config /path/to/lumen-config.json
```

On startup, `lumen-hub` downloads missing model files into `metadata.cache_dir`, validates `model_info.json`, then builds the service hub.
