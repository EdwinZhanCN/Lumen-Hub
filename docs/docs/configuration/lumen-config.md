---
sidebar_position: 1
---

# Configuration Overview

Lumen Hub is driven by a JSON config file, validated against a schema defined in the `lumen-schema` crate.

## Config File Location

Default path: `~/.lumen/config.json`

Override with `--config`:

```bash
lumen-hub --config /path/to/config.json
```

## Top-Level Structure

```json
{
  "metadata": { ... },      // Version, region, cache directory
  "deployment": { ... },    // Deployment mode, service selection
  "server": { ... },        // Server binding, mDNS, batching config
  "services": { ... }       // Per-service model config
}
```

## Key Fields

### metadata

| Field | Type | Description |
|---|---|---|
| `version` | string | Config format version |
| `region` | string | Deployment region (e.g. `cn`) |
| `cache_dir` | string | Model cache directory |

### deployment

| Field | Type | Description |
|---|---|---|
| `mode` | string | `"single"` or `"federated"` — standalone or federated deployment |
| `service` | string | Service name to enable in standalone mode |

### server

| Field | Type | Default | Description |
|---|---|---|---|
| `port` | u16 | `50051` | gRPC port |
| `host` | string | `"0.0.0.0"` | Bind address |
| `mdns` | object? | `null` | mDNS config |
| `batching` | object | `{...}` | [Batching config](./batching-config) |

### services

Each service is indexed by name and contains an `enabled` flag and `models` config:

```json
{
  "services": {
    "clip": {
      "enabled": true,
      "package": "lumen_clip",
      "models": {
        "default": {
          "model": "ViT-B-32",
          "runtime": "onnx"
        }
      }
    }
  }
}
```

## Schema Validation

Config is validated by `jsonschema` on load. Schema definition file:

- `schemas/config/lumen_config.schema.json`

## Key Source

- `crates/lumen-schema/src/config/lumen_config.rs` — Config structure and validation logic
