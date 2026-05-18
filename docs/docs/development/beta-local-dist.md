---
sidebar_position: 2
---

# Beta Local Dist

The first Lumen Hub beta ships as local dist directories, not GitHub Release tags.

## Profiles

| Dist profile | Backend route | Notes |
| --- | --- | --- |
| `universal-cpu` | ORT CPU | Portable CPU baseline. |
| `darwin-arm64` | ORT CPU/XNNPACK + MNN Metal | No CoreML, no Candle. |
| `linux-x64-cuda` | ORT CUDA + MNN CUDA | Requires CUDA runtime libraries on the target machine. |
| `windows-x64-dml` | ORT DirectML + ORT CPU | No MNN. |
| `linux-x64-openvino` | ORT OpenVINO + dynamic ORT loading | Bundles OpenVINO-enabled ONNX Runtime libraries. |

All beta profiles include `clip`, `siglip`, `insightface`, `ppocr`, and `fastvlm`.

## Build

```bash
cargo xtask dist --profile universal-cpu
cargo xtask dist --profile darwin-arm64
cargo xtask dist --profile linux-x64-cuda
cargo xtask dist --profile windows-x64-dml
cargo xtask dist --profile linux-x64-openvino
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
