---
sidebar_position: 2
---

# SigLIP

SigLIP dual-encoder service for text and image embeddings.

## Repository Layout

```text
{cache_dir}/{model_name}/
├── model_info.json
├── tokenizer.json
└── onnx/
    ├── text.fp32.onnx
    ├── text.fp16.onnx
    ├── vision.fp32.onnx
    └── vision.fp16.onnx
```

| Path | Required | Purpose |
|------|----------|---------|
| `model_info.json` | yes | Runtime metadata SSOT |
| `tokenizer.json` | text task only | Tokenizer artifact |
| `{runtime}/{component}.{precision}.{ext}` | yes | Model artifact |

Complete example: [`crates/lumen-hub/tools/siglip/model_info.example.json`](https://github.com/EdwinZhanCN/Lumen-Hub/blob/main/crates/lumen-hub/tools/siglip/model_info.example.json)

## Runtime Metadata

Runtime reads `model_info.json.task_metadata` and consumes these fields:

| Field | Required | Purpose |
|------|----------|---------|
| `tasks` | yes | Task definitions keyed by task name |
| `tasks.<task>.component` | yes | `text` or `vision` |
| `tasks.<task>.input_names` | yes | Model input names |
| `tasks.<task>.output_name` | yes | Primary embedding output |
| `tasks.<task>.preprocess` | image task only | Image preprocessing contract |

`embedding_dim` and `hidden_output_name` may be present as descriptive metadata. Current semantic embedding runtime does not require them.

## Tasks

| Task | Input | Output | Uses |
|------|-------|--------|------|
| `semantic_text_embed` | `text/plain` | `application/json;schema=embedding_v1` | `tokenizer.json` + `text.{precision}.{ext}` |
| `semantic_image_embed` | `image/jpeg`, `image/png`, `image/webp`, `image/avif`, `application/octet-stream` | `application/json;schema=embedding_v1` | `vision.{precision}.{ext}` |

## Limits

- Image tensor input must match the declared preprocess contract.
- Image preprocess metadata is required for image tasks.
- Path convention is fixed to `{cache_dir}/{model_name}/{runtime}/{component}.{precision}.{ext}`.
