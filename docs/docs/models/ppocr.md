---
sidebar_position: 5
---

# PP-OCR

PP-OCR service for end-to-end text detection and recognition.

## Repository Layout

```text
{cache_dir}/{model_name}/
├── model_info.json
├── ppocrv5_dict.txt
└── onnx/
    ├── detection.fp32.onnx
    └── recognition.fp32.onnx
```

| Path | Required | Purpose |
|------|----------|---------|
| `model_info.json` | yes | Runtime metadata SSOT |
| `ppocrv5_dict.txt` | recognition task only | Character vocabulary |
| `onnx/detection.{precision}.onnx` | yes | Detection model |
| `onnx/recognition.{precision}.onnx` | yes | Recognition model |

Complete example: [`crates/lumen-hub/tools/ppocr/model_info.example.json`](https://github.com/EdwinZhanCN/Lumen-Hub/blob/main/crates/lumen-hub/tools/ppocr/model_info.example.json)

## Runtime Metadata

Runtime reads `model_info.json.task_metadata` and consumes these fields:

| Field | Required | Purpose |
|------|----------|---------|
| `tasks` | yes | Task definitions keyed by task name |
| `tasks.<task>.detection.component` | yes | Detection artifact name |
| `tasks.<task>.detection.input_name` | yes | Detection input name |
| `tasks.<task>.detection.output_name` | yes | Detection output name |
| `tasks.<task>.recognition.component` | yes | Recognition artifact name |
| `tasks.<task>.recognition.input_name` | yes | Recognition input name |
| `tasks.<task>.recognition.output_name` | yes | Recognition output name |
| `tasks.<task>.recognition.image_shape` | yes | Static recognition input shape |
| `tasks.<task>.recognition.character_dict_path` | yes | Vocabulary file path |

Other detection and recognition fields in the example file are part of the runtime preprocessing and postprocessing contract.

## Tasks

| Task | Input | Output | Uses |
|------|-------|--------|------|
| `ocr` | `image/jpeg`, `image/png`, `image/webp`, `image/avif`, `application/octet-stream` | `application/json;schema=ocr_v1` | `detection.{precision}.onnx` + `recognition.{precision}.onnx` + vocabulary |

Tensor input skips detection preprocess only. See [Task Input Contract](../architecture/task-input.md).

## Limits

- Input is image-only for the raw path. Tensor input skips detection preprocess only.
- Recognition preprocessing normalizes crops to the configured static shape, for example `[3, 48, 320]`.
- Current runtime support is ONNX only.
