---
sidebar_position: 1
---

# Configuration Overview

Lumen Hub is driven by a **YAML** config file, deserialized and validated by the `lumen-schema` crate.

## Config file location

| Tool | Default path |
|---|---|
| `lumen-cli init` | `~/.lumen/config.yaml` |
| `lumen-hub` | Requires `--config` (no default) |

```bash
lumen-hub --config ~/.lumen/config.yaml
lumen-hub --config /path/to/config.yaml --port 50051 --log-level info
```

## Top-level structure

```yaml
metadata:
  version: "0.1.0"
  region: other          # other | cn
  cache_dir: ~/.lumen/models

deployment:
  mode: hub              # hub | single
  services:              # required for hub mode
    - ocr
    - siglip
    - face

server:
  host: "0.0.0.0"
  port: 50051
  batching:
    enabled: true
    max_batch_size: 8
    queue_latency_ms: 2

services:
  siglip:
    enabled: true
    package: siglip
    models:
      default:
        model: siglip-so400m
        runtime: onnx
        precision: fp32
```

## Key fields

### metadata

| Field | Type | Description |
|---|---|---|
| `version` | string | Config format version (semver `x.y.z`) |
| `region` | string | `cn` uses [hf-mirror.com](https://hf-mirror.com); `other` uses [huggingface.co](https://huggingface.co) |
| `cache_dir` | string | Model cache directory (`~` expanded by hub) |

### deployment

| Field | Type | Description |
|---|---|---|
| `mode` | string | `hub` — multi-service gateway (only mode supported by `lumen-hub` today); `single` — reserved |
| `service` | string? | Service name when `mode: single` |
| `services` | string[]? | Enabled service names when `mode: hub` |

Service names in `deployment.services` are arbitrary keys. They must match top-level keys under `services` and are sent as `meta.service` on gRPC requests when multiple services are registered.

### server

| Field | Type | Default | Description |
|---|---|---|---|
| `port` | u16 | `50051` | gRPC port (hub requires `>= 1024`) |
| `host` | string | `"0.0.0.0"` | Bind address |
| `mdns` | object? | disabled | mDNS-SD config when `enabled: true` |
| `batching` | object | see defaults | [Batching config](./batching-config) |

### services

Each entry is keyed by the **service name** (for example `ocr`, `siglip`, `face`, `bioclip`):

| Field | Description |
|---|---|
| `enabled` | Whether the service loads at startup |
| `package` | Implementation package: `clip`, `siglip`, `ppocr`, `insightface` |
| `models` | Map of model aliases → `model`, `runtime`, `precision`, optional `dataset` (BioCLIP) |

Example packages:

| `package` | Typical service key | Models |
|---|---|---|
| `ppocr` | `ocr` | PP-OCRv5 |
| `siglip` | `siglip` | SigLIP SO400M |
| `insightface` | `face` | antelopev2, buffalo_* |
| `clip` | `bioclip` | BioCLIP-2 + dataset |

## Schema validation

- Rust: `LumenConfig::validate_config()` after `serde_yaml` parse
- JSON Schema: `schemas/config/lumen_config.schema.json`

## Key source

- `crates/lumen-schema/src/config/lumen_config.rs`
- `crates/lumen-hub/src/main.rs` — `load_config`, hub-only `deployment.mode`
