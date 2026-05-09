---
sidebar_position: 2
---

# 批处理配置

`BatchingConfig` 控制 daemon 层的动态请求批处理行为。

## 配置示例

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

## 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | `true` | 全局批处理开关 |
| `max_batch_size` | usize | `8` | 单批次最多合并的请求数，最小值为 1 |
| `queue_latency_ms` | u64 | `2` | 首个请求入队后最多等待的时间（毫秒），最小值为 1 |

## 生效流程

```
LumenConfig::from_json_str(json)
  → config.server.batching = BatchingConfig
  → main.rs:server_config.batching
  → server_config.grpc_port(batching)
    → HubGrpcService::new(hub, batching)
      → Batcher::new(batching)
```

## 调优建议

| 场景 | `max_batch_size` | `queue_latency_ms` |
|---|---|---|
| 高吞吐 GPU 推理 | 16-32 | 5-10ms |
| 低延迟要求 | 2-4 | 1ms |
| CPU 推理 | 1-2 | 1ms |
| 禁用批处理 | — | 设置 `enabled: false` |

## 覆盖批处理

即使 `enabled: true`，以下情况也不会批处理：

1. 请求不是张量输入（`lumen.input.kind != "tensor"`）
2. 请求需要预处理（`lumen.preprocess.skip != "true"`）
3. 任务未实现 `batch_key()`（返回 `None`）

## 关键代码

- `crates/lumen-schema/src/config/lumen_config.rs` — `BatchingConfig` 定义
- `crates/lumen-hub/src/daemon/batcher.rs` — `Batcher::new(config)`
- `crates/lumen-hub/src/daemon/grpc.rs` — `is_batch_wire_eligible()`
