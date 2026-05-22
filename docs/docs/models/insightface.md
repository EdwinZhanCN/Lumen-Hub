---
sidebar_position: 3
---

# InsightFace

InsightFace service for face detection (SCRFD) and recognition (ArcFace). Typical config uses service key `face` with `package: insightface`.

## Repository layout

```text
{cache_dir}/{model_name}/
├── model_info.json
└── onnx/
    ├── detection.fp32.onnx
    └── recognition.fp32.onnx
```

| Path | Required | Purpose |
|------|----------|---------|
| `model_info.json` | yes | Runtime metadata SSOT |
| `onnx/detection.{precision}.onnx` | yes | SCRFD detector |
| `onnx/recognition.{precision}.onnx` | yes | ArcFace embedder |

Complete example: [`crates/lumen-hub/tools/insightface/model_info.example.json`](https://github.com/EdwinZhanCN/Lumen-Hub/blob/main/crates/lumen-hub/tools/insightface/model_info.example.json)

## Runtime metadata

| Field | Required | Purpose |
|------|----------|---------|
| `tasks` | yes | Task definitions |
| `tasks.face_recognition.pack` | yes | Pack spec: `antelopev2`, `buffalo_l`, `buffalo_m`, `buffalo_s`, `buffalo_sc` |
| `tasks.face_recognition.detection.*` | yes | Detection ONNX names and output indices |
| `tasks.face_recognition.recognition.*` | yes | Recognition ONNX names |

Pack geometry (input size, mean/std, letterbox, NMS) is resolved from built-in pack specs in code when `pack` is set.

## Tasks

| Task | Input | Output | Notes |
|------|-------|--------|-------|
| `face_recognition` | `image/*` or detection tensor | `application/json;schema=face_v1` | Raw path runs full detect + recognize |

Tensor path uses `lumen.preprocess.id: insightface_det_v1` and letterbox/source metadata. See [Task Input Contract](../architecture/task-input.md).

## Limits

- Multi-face images return multiple detections in one response
- Batching is not supported (variable face counts per image)
- Beta hub presets use model `antelopev2`
