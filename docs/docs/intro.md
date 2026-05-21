---
sidebar_position: 1
---

# Lumen Hub

**Lumen Hub** is a unified multimodal inference gateway that receives inference requests, dispatches model execution, and returns standardized responses.

## Core Capabilities

- **Multi-model serving**: Host CLIP, SigLIP, OCR, and more in a single process
- **Unified protocol**: The `InferenceService` trait wraps different models behind a common interface — the gRPC transport layer is model-agnostic
- **Dynamic batching**: Automatically merges preprocessed tensor requests into batches to boost GPU/ONNX throughput
- **Service discovery**: Broadcasts service addresses via mDNS — clients don't need hardcoded IPs or ports
- **Configuration-driven**: A JSON-Schema-validated config system supporting standalone and federated deployment modes

## Architecture Layers

```
main.rs          CLI entry: parse args → load config → start service

daemon/          Transport: gRPC codec + dynamic batching + mDNS registration
service/         Abstraction: ServiceHub → TaskRegistry → TaskHandler
models/          Model layer: CLIP / SigLIP / OCR ... (concrete implementations)
```

## Documentation

- [Architecture Overview](./architecture/overview) — Responsibilities of the three layers
- [Request Lifecycle](./architecture/request-lifecycle) — End-to-end path of a gRPC request
- [Batching Design](./architecture/batching) — Trigger conditions and merge strategy for dynamic batching
- [Model Integration Pattern](./architecture/model-pattern) — Factory → Service → Pipeline → Task pattern
