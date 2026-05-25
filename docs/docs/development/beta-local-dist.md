---
sidebar_position: 2
---

# Beta Dist

Hub runtime profiles are built with `cargo xtask dist`. The public `lumen-cli` release archives and shell/PowerShell installers are built with `cargo-dist`.

Rust 1.88+ is required by the current dependency set. The repository pins
`rust-toolchain.toml`, so `rustup` will install/use the right toolchain when it
is available in the build environment.

## Profiles

| Dist profile | Backend route | Notes |
| --- | --- | --- |
| `universal-cpu` | ORT CPU | Portable CPU baseline. |
| `darwin-arm64` | ORT CPU/XNNPACK + MNN Metal | No CoreML, no Candle. |
| `linux-x64-cuda` | ORT CUDA | CUDA package is ORT-only for beta. MNN CUDA is not bundled. |
| `linux-arm64` | ORT CPU + MNN Vulkan | Native Linux ARM64 / aarch64 package. Vulkan-capable MNN runtime is bundled. |
| `linux-arm64-jetson` | ORT CUDA via dynamic ONNX Runtime | JetPack 6+ / CUDA 12.6-oriented profile. Install a compatible Jetson ONNX Runtime GPU package or set `LUMNN_ORT_DYLIB_PATH`. |
| `windows-x64-dml` | ORT DirectML + ORT CPU | No MNN. |
| `linux-x64-openvino` | ORT OpenVINO + dynamic ORT loading | Bundles OpenVINO-enabled ONNX Runtime libraries. |

All beta profiles include `clip`, `siglip`, `insightface`, and `ppocr`.

## Build

```bash
cargo xtask dist --profile universal-cpu
cargo xtask dist --profile darwin-arm64
cargo xtask dist --profile linux-x64-cuda
cargo xtask dist --profile linux-arm64
cargo xtask dist --profile linux-arm64-jetson
cargo xtask dist --profile windows-x64-dml
cargo xtask dist --profile linux-x64-openvino
```

CLI release assets are produced by the release workflow with:

```bash
dist build --artifacts=local --target <target> --tag <tag>
dist build --artifacts=global --tag <tag>
```

## Jetson Runtime

`linux-arm64-jetson` is built for JetPack 6+ and dynamic ONNX Runtime CUDA.
`lumen-cli start --profile linux-arm64-jetson` checks for `onnxruntime-gpu`
through `python3`, detects the local CUDA 12 minor version, and installs
`onnxruntime-gpu` from the matching Jetson AI Lab index when missing: `cu126`,
`cu128`, or `cu129`. If CUDA cannot be detected, it defaults to `cu126`.

You can still point Lumen at a custom runtime:

```bash
export LUMNN_ORT_DYLIB_PATH=/path/to/onnxruntime/capi/libonnxruntime.so
./bin/lumen-hub --config ~/.lumen/config.yaml
```

The package launcher checks common wheel install locations before falling back
to `LUMNN_ORT_DYLIB_PATH`.

The Rust `ort` binding is built against ONNX Runtime API 23, so 1.23.x and
newer runtimes are accepted when the wheel provides the required CUDA execution
provider.

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
./bin/lumen-hub --config ~/.lumen/config.yaml
```

On startup, `lumen-hub` downloads missing model files into `metadata.cache_dir`, validates `model_info.json`, then builds the service hub.
