---
sidebar_position: 3
---

# FastVLM

FastVLM service for image-conditioned text generation via a two-stage pipeline:
embeddings (vision + prompt) → decode (autoregressive generation).

## Repository Layout

```text
{cache_dir}/{model_name}/
├── model_info.json
├── tokenizer.json
└── onnx/
    ├── vision.int8.onnx
    ├── vision.fp16.onnx
    ├── embed.int8.onnx
    ├── embed.fp16.onnx
    ├── decoder.int8.onnx
    └── decoder.fp16.onnx
```

| Path | Required | Purpose |
|------|----------|---------|
| `model_info.json` | yes | Runtime metadata SSOT |
| `tokenizer.json` | yes | Prompt tokenizer artifact |
| `onnx/vision.{precision}.onnx` | yes | Vision encoder |
| `onnx/embed.{precision}.onnx` | yes | Prompt embedding model |
| `onnx/decoder.{precision}.onnx` | yes | Declared package component required by runtime validation |

Complete example: [`crates/lumen-hub/tools/fastvlm/model_info.example.json`](https://github.com/EdwinZhanCN/Lumen-Hub/blob/main/crates/lumen-hub/tools/fastvlm/model_info.example.json)

## Runtime Metadata

Runtime reads `model_info.json.task_metadata` and consumes these fields:

| Field | Required | Purpose |
|------|----------|---------|
| `tasks` | yes | Task definitions keyed by task name |
| `tasks.<task>.components` | yes | Required components for the task |
| `tasks.<task>.output_name` | yes | Output tensor name |
| `tasks.<task>.preprocess_id` | yes | Required tensor preprocess identifier |

Runtime also validates that the selected runtime declares `vision`, `embed`, and `decoder` components.

## Tasks

### `vlm_embeds`

Stage 1: image-conditioned prompt embedding.

| Property | Value |
|----------|-------|
| Input MIME | `application/octet-stream` (tensor) |
| Input Metadata | `lumen.input.kind=tensor`, `lumen.prompt` (required), `lumen.tensor.dtype`, `lumen.tensor.shape=[1,3,448,448]`, `lumen.tensor.layout=NCHW` |
| Preprocess ID | `fastvlm_image_preprocess_v1` |
| Output MIME | `application/octet-stream` (tensor) |
| Output Metadata | `lumen.output.kind=tensor`, `lumen.output.tensor.dtype=fp32\|fp16`, `lumen.output.tensor.shape=[1,S,896]`, `lumen.output.tensor.layout=BSH` |
| Components | `vision.{precision}.onnx` + `embed.{precision}.onnx` + `tokenizer.json` |
| Batchable | yes (same prompt → shared prompt embeds; different images → batched vision) |

### `vlm_decode`

Stage 2: autoregressive text generation from merged embeddings.

| Property | Value |
|----------|-------|
| Input MIME | `application/octet-stream` (tensor) |
| Input Metadata | `lumen.input.kind=tensor`, `lumen.generation.max_tokens` (optional, default `128`), `lumen.tensor.dtype=fp32\|fp16`, `lumen.tensor.shape=[1,S,896]`, `lumen.tensor.layout=BSH` |
| Preprocess ID | `fastvlm_image_preprocess_v1` |
| Output MIME | `application/json;schema=text_generation_v1` |
| Output Schema | `text_generation_v1` |
| Components | `embed.{precision}.onnx` + `decoder.{precision}.onnx` + `tokenizer.json` |
| KV Cache | 24 layers × 2 heads × dim 64, initial past length 0 |
| Limits | `model_max_length=8192`, `default_max_new_tokens=128` |
| Batchable | no (single sequence generation) |

### Generation Parameters

| Metadata Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `lumen.generation.max_tokens` | usize | `128` | Maximum new tokens to generate |

Defaults: `temperature=0.7`, `top_p=0.9`.

## Limits

- Image tensor shape must be `[1, 3, 448, 448]`.
- Merged embeddings tensor must be rank 3 `[B, S, H]` with `B=1`, `H=896`.
- Merged embeddings sequence length `S` must be `>0` and `≤8192`.
- `decoder` component is required for the package even when only `vlm_embeds` is used.

## Notes

- Path convention is fixed to `{cache_dir}/{model_name}/{runtime}/{component}.{precision}.{ext}`.
- Task names are fixed to `vlm_embeds` and `vlm_decode`. Only one FastVLM model per service is supported.
- Merged embeddings are deterministic for the same image + prompt; clients are encouraged to cache them.
- If a different artifact naming scheme is needed, extend model config or task metadata instead of documenting exceptions here.
