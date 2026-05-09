---
sidebar_position: 1
---

# Lumen Hub

**Lumen Hub** 是一个统一的多模态推理网关，负责接收推理请求、调度模型执行、并返回标准化的响应。

## 核心能力

- **多模型服务**：在一个进程中同时承载 CLIP、SigLIP、FastVLM 等多个模型
- **协议统一**：通过 `InferenceService` trait 将不同模型封装为统一接口，上层 gRPC 传输不感知模型差异
- **动态批处理**：对预处理过的张量请求自动合并批次，提高 GPU/ONNX 推理吞吐
- **服务发现**：通过 mDNS 广播服务地址，客户端无需硬编码 IP 端口
- **配置驱动**：完整的 JSON Schema 校验的配置系统，支持单服务/联合部署模式

## 架构分层

```
main.rs          CLI 入口：解析参数 → 加载配置 → 启动服务

daemon/          传输层：gRPC 编解码 + 动态批处理 + mDNS 注册
service/         抽象层：ServiceHub → TaskRegistry → TaskHandler
models/          模型层：CLIP / SigLIP / FastVLM ...（具体实现）
```

## 文档导航

- [架构概览](./architecture/overview) — 三层架构的职责划分
- [请求生命周期](./architecture/request-lifecycle) — 一个 gRPC 请求从入站到返回的完整路径
- [批处理设计](./architecture/batching) — 动态批处理的触发条件和合并策略
- [模型集成模式](./architecture/model-pattern) — Factory → Service → Pipeline → Task 模式
