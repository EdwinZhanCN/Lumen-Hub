---
sidebar_position: 1
---

# CLIP

OpenAI CLIP 视觉-文本对齐模型。

## 任务

| 任务名 | 输入 | 输出 | 支持批处理 |
|---|---|---|---|
| `image_embed` | 图像文件 (JPEG/PNG/WebP/AVIF) 或预处理张量 | L2 归一化嵌入向量 | ✅ 仅张量 |
| `text_embed` | 文本字符串 | L2 归一化嵌入向量 | ❌ |

## 预处理

图像预处理参数由 `model_info.json` 中的 `task_metadata.tasks.<task>.preprocess` 定义：

- `resize` — 最长边缩放目标
- `center_crop` — 中心裁剪尺寸
- `mean` / `std` — 归一化参数

## 批处理

`ClipImageEmbedTask` 覆盖了 `batch_key()` 和 `handle_batch()`：

- `batch_key()` 返回模型 ID + 版本 + 张量形状 + 数据类型，确保兼容
- `handle_batch()` 沿 batch 维度（NCHW 的第一个维度）拼接多个 `[C, H, W]` 张量为 `[N, C, H, W]`

## 关键文件

- `crates/lumen-hub/src/models/clip/task.rs` — TaskHandler 实现
- `crates/lumen-hub/src/models/clip/pipeline.rs` — 推理管线
- `crates/lumen-hub/src/models/clip/nodes.rs` — L2NormalizeNode
- `crates/lumen-hub/src/models/clip/service.rs` — InferenceService
- `crates/lumen-hub/src/models/clip/factory.rs` — ModelFactory
