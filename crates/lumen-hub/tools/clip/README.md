# CLIP

CLIP dual-encoder service for text and image embeddings.

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

Complete example: [`model_info.example.json`](./model_info.example.json)

## Runtime Metadata

Runtime reads `model_info.json.task_metadata` and consumes these fields:

| Field | Required | Purpose |
|------|----------|---------|
| `tasks` | yes | Task definitions keyed by task name |
| `tasks.<task>.component` | yes | `text` or `vision` |
| `tasks.<task>.input_names` | yes | Model input names |
| `tasks.<task>.output_name` | yes | Primary embedding output |
| `tasks.<task>.preprocess` | image task only | Image preprocessing contract |

`embedding_dim` is optional descriptive metadata. Runtime does not require it.

## Tasks

| Task | Input | Output | Uses |
|------|-------|--------|------|
| `semantic_text_embed` | `text/plain` | `application/json;schema=embedding_v1` | `tokenizer.json` + `text.{precision}.{ext}` |
| `semantic_image_embed` | `image/jpeg`, `image/png`, `image/webp`, `image/avif`, `application/octet-stream` | `application/json;schema=embedding_v1` | `vision.{precision}.{ext}` |

## Limits

- Image tensor input must match the declared preprocess contract.
- Image preprocess metadata is required for image tasks.
- Path convention is fixed to `{cache_dir}/{model_name}/{runtime}/{component}.{precision}.{ext}`.

## Notes

- Runtime only supports the task metadata fields it actually consumes.
- If a different artifact naming scheme is needed, extend model config or task metadata instead of documenting exceptions here.
