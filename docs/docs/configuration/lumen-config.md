---
sidebar_position: 1
---

# 配置概览

Lumen Hub 使用 JSON 配置文件驱动所有行为，配置 schema 由 `lumen-schema` crate 定义和校验。

## 配置文件位置

默认路径：`~/.lumen/config.json`

可通过 `--config` 参数覆盖：

```bash
lumen-hub --config /path/to/config.json
```

## 顶级结构

```json
{
  "metadata": { ... },      // 版本、区域、缓存目录
  "deployment": { ... },    // 部署模式、服务选择
  "server": { ... },        // 服务器绑定、mDNS、批处理配置
  "services": { ... }       // 各服务的模型配置
}
```

## 关键配置项

### metadata

| 字段 | 类型 | 说明 |
|---|---|---|
| `version` | string | 配置格式版本 |
| `region` | string | 部署区域（如 `cn`） |
| `cache_dir` | string | 模型缓存目录 |

### deployment

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | string | `"single"` / `"federated"` — 单机或联合部署 |
| `service` | string | 单机模式下要启用的服务名 |

### server

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `port` | u16 | `50051` | gRPC 端口 |
| `host` | string | `"0.0.0.0"` | 绑定地址 |
| `mdns` | object? | `null` | mDNS 配置 |
| `batching` | object | `{...}` | [批处理配置](./batching-config) |

### services

每个服务通过服务名索引，包含 `enabled` 标志和 `models` 配置：

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

## Schema 校验

配置在加载时通过 `jsonschema` 校验。Schema 定义文件：

- `schemas/config/lumen_config.schema.json`

## 关键代码

- `crates/lumen-schema/src/config/lumen_config.rs` — 配置结构和校验逻辑
