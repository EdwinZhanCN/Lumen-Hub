---
sidebar_position: 6
---

# Task Request Examples

Complete request examples for every Lumen Hub task. All requests are sent over the gRPC `Infer` bidi stream as one or more `InferRequest` chunks.

Contract details: [Task Input Contract](./task-input.md) · Batching: [Batching Design](./batching.md)

## gRPC envelope

Each chunk uses this shape (single-chunk requests omit `seq` / `total` or set `seq=0`):

```json
{
  "correlation_id": "req-001",
  "task": "<task_name>",
  "payload_mime": "<mime>",
  "payload": "<bytes>",
  "meta": {
    "service": "<service_name>"
  },
  "seq": 0,
  "total": 1
}
```

| Field | Required | Notes |
|---|---|---|
| `task` | yes | Registered task name (see below) |
| `payload_mime` | yes | Input MIME type |
| `payload` | yes | Raw bytes (image, text UTF-8, or tensor) |
| `meta.service` | when multiple services are registered | e.g. `clip`, `siglip`, `ocr`, `face`, `bioclip` |
| `correlation_id` | recommended | Tracing / response correlation |

When `meta.service` is omitted and exactly one service is registered, the hub routes to that service automatically.

---

## Shared tensor metadata

All tensor inputs (`application/octet-stream`) require:

```json
{
  "lumen.input.kind": "tensor",
  "lumen.preprocess.skip": "true",
  "lumen.preprocess.id": "<preprocess_id>",
  "lumen.tensor.dtype": "fp32",
  "lumen.tensor.shape": "[1,3,224,224]",
  "lumen.tensor.layout": "NCHW",
  "lumen.tensor.format": "contiguous",
  "lumen.tensor.byte_order": "little"
}
```

| `lumen.preprocess.id` | Task |
|---|---|
| `clip_image_preprocess_v1` | CLIP / BioCLIP image |
| `siglip_image_preprocess_v1` | SigLIP image |
| `ppocr_det_v1` | PPOCR detection input |
| `insightface_det_v1` | InsightFace detection input |

Det-skip tasks also require source geometry (Face additionally requires letterbox):

```json
{
  "lumen.source.width": "1920",
  "lumen.source.height": "1080",
  "lumen.letterbox.scale": "0.3333333",
  "lumen.letterbox.pad_x": "0",
  "lumen.letterbox.pad_y": "140"
}
```

(`lumen.letterbox.*` is required for `face_recognition` only.)

---

## `semantic_text_embed` (CLIP / SigLIP)

**Output:** `application/json;schema=embedding_v1`

### Raw — text

```json
{
  "correlation_id": "text-001",
  "task": "semantic_text_embed",
  "payload_mime": "text/plain",
  "payload": "a photo of a cat",
  "meta": {
    "service": "clip"
  }
}
```

Tensor input is not supported yet.

---

## `semantic_image_embed` (CLIP / SigLIP)

**Output:** `application/json;schema=embedding_v1` · **Batching:** yes (tensor path)

### Raw — image

```json
{
  "correlation_id": "img-001",
  "task": "semantic_image_embed",
  "payload_mime": "image/jpeg",
  "payload": "<jpeg bytes>",
  "meta": {
    "service": "clip"
  }
}
```

Supported raw MIME types: `image/jpeg`, `image/png`, `image/webp`, `image/avif`.

### Tensor — full preprocess skip (CLIP, 224×224 fp32)

```json
{
  "correlation_id": "img-002",
  "task": "semantic_image_embed",
  "payload_mime": "application/octet-stream",
  "payload": "<602112 bytes: 1×3×224×224 fp32 little-endian>",
  "meta": {
    "service": "clip",
    "lumen.input.kind": "tensor",
    "lumen.preprocess.skip": "true",
    "lumen.preprocess.id": "clip_image_preprocess_v1",
    "lumen.tensor.dtype": "fp32",
    "lumen.tensor.shape": "[1,3,224,224]",
    "lumen.tensor.layout": "NCHW",
    "lumen.tensor.format": "contiguous",
    "lumen.tensor.byte_order": "little"
  }
}
```

### Tensor — SigLIP (same shape contract, different preprocess id)

```json
{
  "correlation_id": "img-003",
  "task": "semantic_image_embed",
  "payload_mime": "application/octet-stream",
  "payload": "<602112 bytes: 1×3×224×224 fp32 little-endian>",
  "meta": {
    "service": "siglip",
    "lumen.input.kind": "tensor",
    "lumen.preprocess.skip": "true",
    "lumen.preprocess.id": "siglip_image_preprocess_v1",
    "lumen.tensor.dtype": "fp32",
    "lumen.tensor.shape": "[1,3,224,224]",
    "lumen.tensor.layout": "NCHW",
    "lumen.tensor.format": "contiguous",
    "lumen.tensor.byte_order": "little"
  }
}
```

Use `lumen.tensor.dtype=fp16` and half the payload size when the loaded model precision is fp16.

### Batching note

Batching requires the tensor path above plus `server.batching.enabled=true`. Only requests with identical `service`, `task`, model, dtype, shape tail, and `lumen.preprocess.id` are merged. Raw images are never batched.

---

## `bioclip_classify` (CLIP + dataset)

**Output:** `application/json;schema=labels_v1` · **Batching:** yes (tensor path, same as CLIP image)

### Raw — image

```json
{
  "correlation_id": "cls-001",
  "task": "bioclip_classify",
  "payload_mime": "image/png",
  "payload": "<png bytes>",
  "meta": {
    "service": "bioclip",
    "top_k": "10"
  }
}
```

Optional Top-K meta keys (any one): `TopK`, `topK`, `top_k`, `top-k`, `lumen.top_k`. Default is `5`.

### Tensor — full preprocess skip

```json
{
  "correlation_id": "cls-002",
  "task": "bioclip_classify",
  "payload_mime": "application/octet-stream",
  "payload": "<602112 bytes: 1×3×224×224 fp32 little-endian>",
  "meta": {
    "service": "bioclip",
    "top_k": "10",
    "lumen.input.kind": "tensor",
    "lumen.preprocess.skip": "true",
    "lumen.preprocess.id": "clip_image_preprocess_v1",
    "lumen.tensor.dtype": "fp32",
    "lumen.tensor.shape": "[1,3,224,224]",
    "lumen.tensor.layout": "NCHW",
    "lumen.tensor.format": "contiguous",
    "lumen.tensor.byte_order": "little"
  }
}
```

---

## `ocr` (PPOCR)

**Output:** `application/json;schema=ocr_v1` · **Batching:** no

### Raw — image

```json
{
  "correlation_id": "ocr-001",
  "task": "ocr",
  "payload_mime": "image/png",
  "payload": "<png bytes>",
  "meta": {
    "service": "ocr"
  }
}
```

### Tensor — detection preprocess skip

Client must supply a detection-ready NCHW tensor and original image dimensions. Shape must be `[1,3,H,W]` where `H` and `W` are multiples of 32.

```json
{
  "correlation_id": "ocr-002",
  "task": "ocr",
  "payload_mime": "application/octet-stream",
  "payload": "<1×3×736×1280 fp32 little-endian bytes>",
  "meta": {
    "service": "ocr",
    "lumen.input.kind": "tensor",
    "lumen.preprocess.skip": "true",
    "lumen.preprocess.id": "ppocr_det_v1",
    "lumen.tensor.dtype": "fp32",
    "lumen.tensor.shape": "[1,3,736,1280]",
    "lumen.tensor.layout": "NCHW",
    "lumen.tensor.format": "contiguous",
    "lumen.tensor.byte_order": "little",
    "lumen.source.width": "1920",
    "lumen.source.height": "1080"
  }
}
```

Server still runs DB post-processing, per-box recognition, and CTC decode. Output boxes are in original-image coordinates.

---

## `face_recognition` (InsightFace)

**Output:** `application/json;schema=face_v1` · **Batching:** no

### Raw — image

```json
{
  "correlation_id": "face-001",
  "task": "face_recognition",
  "payload_mime": "image/png",
  "payload": "<png bytes>",
  "meta": {
    "service": "face"
  }
}
```

### Tensor — detection preprocess skip

Client performs letterbox, normalize, and NCHW packing. Shape must match pack `detection.input_size` (e.g. `[1,3,640,640]` for antelopev2). Letterbox metadata is required so bbox / landmarks map back to the original image.

```json
{
  "correlation_id": "face-002",
  "task": "face_recognition",
  "payload_mime": "application/octet-stream",
  "payload": "<4915200 bytes: 1×3×640×640 fp32 little-endian>",
  "meta": {
    "service": "face",
    "lumen.input.kind": "tensor",
    "lumen.preprocess.skip": "true",
    "lumen.preprocess.id": "insightface_det_v1",
    "lumen.tensor.dtype": "fp32",
    "lumen.tensor.shape": "[1,3,640,640]",
    "lumen.tensor.layout": "NCHW",
    "lumen.tensor.format": "contiguous",
    "lumen.tensor.byte_order": "little",
    "lumen.source.width": "1920",
    "lumen.source.height": "1080",
    "lumen.letterbox.scale": "0.3333333",
    "lumen.letterbox.pad_x": "0",
    "lumen.letterbox.pad_y": "140"
  }
}
```

Server still runs SCRFD post-processing, NMS, face alignment, and ArcFace recognition.

---

## Quick reference

| Task | Service examples | Raw `payload_mime` | Tensor `preprocess.id` | Batching |
|---|---|---|---|---|
| `semantic_text_embed` | `clip`, `siglip` | `text/plain` | — | no |
| `semantic_image_embed` | `clip`, `siglip` | `image/*` | `clip_image_preprocess_v1` / `siglip_image_preprocess_v1` | yes |
| `bioclip_classify` | `bioclip` | `image/*` | `clip_image_preprocess_v1` | yes |
| `ocr` | `ocr` | `image/*` | `ppocr_det_v1` + source meta | no |
| `face_recognition` | `face` | `image/*` | `insightface_det_v1` + source/letterbox meta | no |

## Deprecated

Do not use `application/vnd.lumen.tensor+json` or JSON pixel payloads for `face_recognition`. Use `application/octet-stream` with the metadata above.
