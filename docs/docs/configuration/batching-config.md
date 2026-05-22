---
sidebar_position: 2
---

# Batching Config

`BatchingConfig` controls the daemon layer's dynamic request batching behaviour.

## Example

```yaml
server:
  batching:
    enabled: true
    max_batch_size: 8
    queue_latency_ms: 2
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Global batching toggle |
| `max_batch_size` | usize | `8` | Max requests merged per batch (minimum: 1) |
| `queue_latency_ms` | u64 | `2` | Max wait after first request enqueues, in ms (minimum: 1) |

## Flow

```
config.yaml
  → serde_yaml → LumenConfig
  → config.server.batching
  → HubGrpcService::new(hub, batching)
  → Batcher::new(batching)
```

## Tuning recommendations

| Scenario | `max_batch_size` | `queue_latency_ms` |
|---|---|---|
| High-throughput GPU | 16–32 | 5–10 |
| Low latency | 2–4 | 1 |
| CPU inference | 1–2 | 1 |
| Disable batching | — | `enabled: false` |

## Bypassing batching

Even with `enabled: true`, batching is skipped when:

1. Request input is not a tensor (`lumen.input.kind != "tensor"`)
2. Request requires preprocessing (`lumen.preprocess.skip != "true"`)
3. Task returns `None` from `batch_key()`

See [Task Input Contract](../architecture/task-input.md).

## Key source

- `crates/lumen-schema/src/config/lumen_config.rs` — `BatchingConfig`
- `crates/lumen-hub/src/daemon/batcher.rs`
- `crates/lumen-hub/src/daemon/grpc.rs` — `is_batch_wire_eligible()`
