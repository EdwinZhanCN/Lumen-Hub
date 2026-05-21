---
sidebar_position: 5
---

# Task Input Contract

All Lumen Hub tasks accept two input paths. Tensor inputs always use `application/octet-stream` with `lumen.*` metadata.

## Path A — Raw (default)

```text
payload_mime: image/jpeg | image/png | image/webp | image/avif
              text/plain                         (semantic_text_embed only)
meta:         service = <name>                   (required when multiple services are registered)
batching:     never
```

## Path B — Tensor (preprocessed)

```text
payload_mime: application/octet-stream
meta:
  lumen.input.kind         = tensor
  lumen.preprocess.skip    = true
  lumen.preprocess.id      = <see table below>
  lumen.tensor.dtype       = fp32 | fp16
  lumen.tensor.shape       = JSON array, e.g. [1,3,224,224]
  lumen.tensor.layout      = NCHW
  lumen.tensor.format      = contiguous
  lumen.tensor.byte_order  = little
  service                  = <name>               (required when multiple services are registered)
batching:     embedding tasks only (see batching.md)
```

### `lumen.preprocess.id` registry

| ID | Task | Shape | Skip semantics |
|---|---|---|---|
| `clip_image_preprocess_v1` | `semantic_image_embed`, `bioclip_classify` | Fixed from `model_info.preprocess` | Full image preprocess |
| `siglip_image_preprocess_v1` | `semantic_image_embed` | Fixed from `model_info.preprocess` | Full image preprocess |
| `ppocr_det_v1` | `ocr` | Dynamic `[1,3,H,W]`, H and W multiples of 32 | Detection preprocess only |
| `insightface_det_v1` | `face_recognition` | Fixed from pack `detection.input_size` | Detection preprocess only |

### Source / letterbox metadata (det skip tasks)

When skipping detection preprocess, clients must supply geometry metadata so boxes and landmarks map back to the original image:

| Key | Required for | Description |
|---|---|---|
| `lumen.source.width` | `ocr`, `face_recognition` | Original image width in pixels |
| `lumen.source.height` | `ocr`, `face_recognition` | Original image height in pixels |
| `lumen.letterbox.scale` | `face_recognition` | Letterbox scale factor |
| `lumen.letterbox.pad_x` | `face_recognition` | Horizontal padding in pixels |
| `lumen.letterbox.pad_y` | `face_recognition` | Vertical padding in pixels |

For `ocr`, `ratio_h = tensor_H / source.height` and `ratio_w = tensor_W / source.width`.

## Task summary

| Task | Raw | Tensor | Batching |
|---|---|---|---|
| `semantic_text_embed` | `text/plain` | not yet | no |
| `semantic_image_embed` | `image/*` | full skip | yes |
| `bioclip_classify` | `image/*` | full skip (`clip_image_preprocess_v1`) | yes |
| `ocr` | `image/*` | det skip (`ppocr_det_v1`) + source meta | no |
| `face_recognition` | `image/*` | det skip (`insightface_det_v1`) + source/letterbox meta | no |

## Deprecated

- `application/vnd.lumen.tensor+json` for `face_recognition` — use Path B instead.

Complete copy-paste examples: [Task Request Examples](./task-request-examples.md).
