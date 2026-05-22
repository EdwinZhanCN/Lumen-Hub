---
sidebar_position: 1
---

# Lumen Hub

**Lumen Hub** is a unified multimodal inference gateway. It receives gRPC inference requests, routes them to registered model services, and returns schema-validated responses.

The project is in **beta**: end users install [`lumen-cli`](https://github.com/EdwinZhanCN/Lumen-Hub/releases) (`0.1.0-beta.x`), run `init` to generate a YAML config, and `start` to download the matching hub runtime from the release manifest.

## Core capabilities

- **Multi-model serving**: CLIP, SigLIP, PP-OCR, InsightFace, and BioCLIP (dataset classification via CLIP) in one process
- **Unified protocol**: gRPC `Infer` streaming; the transport layer is model-agnostic
- **Dynamic batching**: Merges compatible preprocessed tensor requests to improve ONNX throughput
- **Service discovery**: Optional mDNS advertisement (`_lumen._tcp`)
- **YAML configuration**: Validated by `lumen-schema`; models download from Hugging Face (`Lumilio-Photos/{model}`)

## Workspace crates

| Crate | Role |
|---|---|
| `lumen-hub` | Inference server (daemon, services, models) |
| `lumen-cli` | Beta installer, config wizard, runtime launcher |
| `lumen-schema` | Config and result schemas |
| `lumnn` | ML runtime abstraction (ONNX Runtime, optional MNN and Candle) |
| `lumnn-mnn-sys` | MNN native bindings |
| `xtask` | Hub dist profile builds (`cargo xtask dist`) |

## Architecture layers

```
crates/lumen-hub/src/
  main.rs          CLI entry: load config → download models → build hub → serve gRPC
  daemon/          Transport: gRPC codec, batching, mDNS, Tonic server
  service/         Abstraction: ServiceHub → TaskRegistry → TaskHandler
  models/          Model layer: clip, siglip, ppocr, insightface
```

Inference backends are selected per model via `runtime` in config (`onnx` or `mnn`). Beta dist profiles bundle ONNX Runtime; some profiles also bundle MNN (for example Metal on macOS ARM64).

## Documentation map

- [Beta Quick Start](./beta-quick-start) — Install CLI, init, start hub
- [Architecture Overview](./architecture/overview) — Three-layer responsibilities
- [Request Lifecycle](./architecture/request-lifecycle) — gRPC request path
- [Task Input Contract](./architecture/task-input) — Raw vs tensor inputs
- [Batching Design](./architecture/batching) — Dynamic batching rules
- [Model Integration Pattern](./architecture/model-pattern) — Factory → Service → Pipeline → Task
- [Configuration](./configuration/lumen-config) — YAML config reference
