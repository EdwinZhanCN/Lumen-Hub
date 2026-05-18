---
sidebar_position: 4
---

# OpenVINO Package

`linux-x64-openvino` bundles ONNX Runtime OpenVINO libraries from:

```text
onnxruntime-openvino==1.24.1
```

The source wheel is the Linux x64 `manylinux_2_28` PyPI wheel. The package targets glibc 2.28+ environments.

## xtask Behavior

`cargo xtask dist --profile linux-x64-openvino` downloads the wheel, verifies SHA256, and extracts `.so*` files into:

```text
dist/lumen-hub-linux-x64-openvino/lib/
```

The artifact includes `lib/libonnxruntime.so` and OpenVINO provider libraries.

## Launcher

The package exposes `bin/lumen-hub` as a wrapper. It sets:

```bash
LUMNN_ORT_DYLIB_PATH=$APP_HOME/lib/libonnxruntime.so
LD_LIBRARY_PATH=$APP_HOME/lib:$LD_LIBRARY_PATH
```

Then it executes `bin/lumen-hub-bin`.

Use `LUMNN_ORT_OPENVINO_DEVICE=CPU` to force OpenVINO CPU execution on machines without a usable Intel GPU or NPU.
