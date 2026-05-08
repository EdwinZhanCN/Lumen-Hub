# Lumen Hub CLIP Model

CLIP 双编码器推理服务：文本嵌入和图像嵌入。

## 模型仓库结构

```
{cache_dir}/{model_name}/
├── model_info.json              # ModelInfo schema（必需）
├── tokenizer.json               # HuggingFace tokenizer（文本模型必需）
├── onnx/
│   ├── text.fp32.onnx
│   ├── text.fp16.onnx
│   ├── vision.fp32.onnx
│   └── vision.fp16.onnx
└── rknn/
    ├── text.fp32.rknn
    └── vision.fp16.rknn
```

- 模型文件命名约定：`{component}.{precision}.{ext}`
- `component`: `text`（文本编码器）或 `vision`（图像编码器）
- `precision`: 遵循 HuggingFace 惯例 — `fp32`, `fp16`, `fp8`, `fp4`, `int8` 等
- `ext`: `onnx`（ONNX Runtime）或 `rknn`（RKNN Runtime）
- `model_info.json` 位于仓库根目录，是 CLIP 运行时元数据的唯一来源
- `tokenizer.json` 是文本任务运行所需的 tokenizer artifact

### model_info.json

完整最小示例见 [`model_info.example.json`](./model_info.example.json)。该文件是 CLIP
服务 `model_info.json` 示例的 SSOT；README 只描述字段契约，不维护重复 JSON 示例。

## task_metadata 字段

位于 `model_info.json` 的 `task_metadata` 字段，CLIP 服务解析以下结构：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `tasks` | `map<string, object>` | 是 | 任务定义，key 为任务名（如 `text_embed`, `image_embed`） |

`embedding_dim` 可以保留为附加信息，但当前运行时不消费该字段，因此不是必需元数据。

### 每个 task 的配置

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `component` | `string` | 是 | 模型组件名：`"text"` 或 `"vision"` |
| `input_names` | `[string]` | 是 | ONNX 模型输入节点名，按顺序排列。text 模型通常为 `["input_ids", "attention_mask"]`，vision 模型通常为 `["pixel_values"]` |
| `output_name` | `string` | 是 | ONNX 模型 forward-pass 输出的原始 embedding 节点名（在 L2 归一化之前），如 `"text_embeds"` 或 `"image_embeds"` |
| `preprocess` | `object` | vision task 必需 | 图像预处理元数据。text task 不需要 |

### vision preprocess 配置

`preprocess` 位于 `task_metadata.tasks.<image_task>.preprocess`。运行时只从这里读取图像预处理配置，不读取 HuggingFace `preprocessor_config.json`。

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `resize_shortest_edge` | `u32` | 是 | resize 时短边目标尺寸，必须大于 0 |
| `crop_size.width` | `u32` | 是 | 最终输入宽度，必须大于 0 |
| `crop_size.height` | `u32` | 是 | 最终输入高度，必须大于 0 |
| `do_resize` | `bool` | 是 | 是否先按短边 resize |
| `do_center_crop` | `bool` | 是 | 是否中心裁剪；为 false 时会直接 resize 到 `crop_size` |
| `do_rescale` | `bool` | 是 | 是否应用 `rescale_factor` |
| `do_normalize` | `bool` | 是 | 是否应用 `image_mean` / `image_std` |
| `rescale_factor` | `f32` | 是 | 像素缩放因子，例如 `1 / 255` |
| `image_mean` | `[f32; 3]` | 是 | RGB 三通道均值 |
| `image_std` | `[f32; 3]` | 是 | RGB 三通道标准差，不能为 0 |
| `resample` | `string` | 是 | `nearest`、`lanczos3`、`bilinear` 或 `bicubic` |
| `color_space` | `string` | 是 | 当前只支持 `rgb` |
| `layout` | `string` | 是 | 当前只支持 `nchw` |

## 提供的 Task

### text_embed（已实现）

| 属性 | 值 |
|------|-----|
| 输入 MIME | `text/plain` |
| 输出 MIME | `application/json;schema=embedding_v1` |
| 预处理 | HuggingFace `tokenizers` 库进行 tokenization → `input_ids` + `attention_mask` |
| 推理 | `text.{precision}.onnx` → L2 Normalize |
| 输出格式 | `EmbeddingV1` JSON `{"vector": [...], "dim": N, "model_id": "..."}` |

### image_embed（已实现）

| 属性 | 值 |
|------|-----|
| 输入 MIME | `image/jpeg`, `image/png`, `image/webp`, `image/avif` |
| 输出 MIME | `application/json;schema=embedding_v1` |
| 预处理 | `model_info.json` 的 `task_metadata.tasks.<image_task>.preprocess`：resize shortest edge + center crop + rescale + normalize |
| 推理 | `vision.{precision}.onnx` → L2 Normalize |
| 输出格式 | `EmbeddingV1` JSON |

## 加载流程

```
ServiceConfig.models[alias]
  │
  ├── 1. 加载 {cache_dir}/{model_name}/model_info.json
  │
  ├── 2. 解析 task_metadata.tasks (BTreeMap<String, ClipTaskConfig>)
  │
  ├── 3. For each task:
  │       ├── 按 component 分发
  │       ├── "text" → 创建 OrtNode({runtime}/text.{precision}.onnx)
  │       │         → 加载 tokenizer.json
  │       │         → 构建 pipeline: OrtNode → L2NormalizeNode
  │       │         → 注册 ClipTextEmbedTask
  │       └── "vision" → 创建 OrtNode({runtime}/vision.{precision}.onnx)
  │                   → 读取 task_metadata.tasks.<task>.preprocess
  │                   → 构建 pipeline: OrtNode → L2NormalizeNode
  │                   → 注册 ClipImageEmbedTask
  │
  └── 4. 返回 ClipService
```

## 路径硬编码说明

路径解析规则在当前版本中是硬编码的，不可配置：

| 资源 | 路径 | 说明 |
|------|------|------|
| 仓库元数据 | `{cache_dir}/{model}/model_info.json` | 固定文件名 |
| Tokenizer | `{cache_dir}/{model}/tokenizer.json` | 固定文件名 |
| 模型权重 | `{cache_dir}/{model}/{runtime}/{component}.{precision}.{ext}` | 三段式命名 |

runtime/component/precision 三元组直接映射到文件系统路径，这是被接受的硬编码耦合——如果需要其他命名约定，应当在 `ModelConfig` 或 `ClipTaskConfig` 中增加可配置字段。
