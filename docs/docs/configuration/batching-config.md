---
sidebar_position: 2
---

# Batching Config

`BatchingConfig` controls the daemon layer's dynamic request batching behaviour.

## Example

```json
{
  "server": {
    "batching": {
      "enabled": true,
      "max_batch_size": 8,
      "queue_latency_ms": 2
    }
  }
}
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Global batching toggle |
| `max_batch_size` | usize | `8` | Max requests merged per batch (minimum: 1) |
| `queue_latency_ms` | u64 | `2` | Max wait after first request enqueues, in ms (minimum: 1) |

## Flow

```
LumenConfig::from_json_str(json)
  → config.server.batching = BatchingConfig
  → main.rs:server_config.batching
  → server_config.grpc_port(batching)
    → HubGrpcService::new(hub, batching)
      → Batcher::new(batching)
```

## Tuning Recommendations

| Scenario | `max_batch_size` | `queue_latency_ms` |
|---|---|---|
| High-throughput GPU | 16–32 | 5–10 ms |
| Low-latency required | 2–4 | 1 ms |
| CPU inference | 1–2 | 1 ms |
| Disable batching | — | Set `enabled: false` |

## Bypassing Batching

Even with `enabled: true`, batching is skipped when:

1. Request input is not a tensor (`lumen.input.kind != "tensor"`)
2. Request requires preprocessing (`lumen.preprocess.skip != "true"`)
3. Task doesn't implement `batch_key()` (returns `None`)

## Key Source

- `crates/lumen-schema/src/config/lumen_config.rs` — `BatchingConfig` definition
- `crates/lumen-hub/src/daemon/batcher.rs` — `Batcher::new(config)`
- `crates/lumen-hub/src/daemon/grpc.rs` — `is_batch_wire_eligible()`
