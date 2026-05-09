---
sidebar_position: 2
---

# SigLIP

Google SigLIP（Sigmoid Loss for Language-Image Pre-training）模型。

## 任务

| 任务名 | 输入 | 输出 | 支持批处理 |
|---|---|---|---|
| `image_embed` | 图像文件 (JPEG/PNG/WebP/AVIF) 或预处理张量 | L2 归一化嵌入向量 | ✅ 仅张量 |
| `text_embed` | 文本字符串 | L2 归一化嵌入向量 | ❌ |

## 实现

SigLIP 的实现结构与 CLIP 完全一致，差异仅在于：

- 模型权重/架构不同
- 预处理参数不同（从各自的 `model_info.json` 读取）
- 图像输入分辨率可能不同

## 批处理

`SiglipImageEmbedTask` 的批处理方式与 CLIP 相同——沿 NCHW 的第一个维度拼接张量。

## 关键文件

- `crates/lumen-hub/src/models/siglip/task.rs` — TaskHandler 实现
- `crates/lumen-hub/src/models/siglip/pipeline.rs` — 推理管线
- `crates/lumen-hub/src/models/siglip/nodes.rs` — L2NormalizeNode
- `crates/lumen-hub/src/models/siglip/service.rs` — InferenceService
- `crates/lumen-hub/src/models/siglip/factory.rs` — ModelFactory
