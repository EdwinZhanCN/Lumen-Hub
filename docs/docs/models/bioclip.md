---
sidebar_position: 2
---

# BioCLIP

BioCLIP zero-shot image classification using a CLIP vision encoder plus a precomputed text embedding dataset (`.npy` + labels JSON, optional HNSW index).

Uses the **`clip` package** with a `dataset` field on the model config. Typical service key: `bioclip`.

## Config example

```yaml
services:
  bioclip:
    enabled: true
    package: clip
    models:
      default:
        model: bioclip-2
        runtime: onnx
        precision: fp16
        dataset: TreeOfLife200M
```

## Repository layout

```text
{cache_dir}/bioclip-2/
├── model_info.json
├── onnx/
│   └── vision.fp16.onnx
└── datasets/
    ├── TreeOfLife200M.json
    ├── TreeOfLife200M.npy
    └── TreeOfLife200M.bin    # optional HNSW index
```

| Path | Required | Purpose |
|------|----------|---------|
| `model_info.json` | yes | Must declare a vision task in `task_metadata` |
| `onnx/vision.{precision}.onnx` | yes | Shared CLIP vision encoder convention |
| `datasets/{dataset}.json` | yes | Label list |
| `datasets/{dataset}.npy` | yes | Text embedding matrix |
| `datasets/{dataset}.bin` | no | HNSW index for faster Top-K |

Tooling to build dataset packs: `crates/lumen-hub/tools/clip/scripts/`.

## Task

| Task | Input | Output | Batching |
|------|-------|--------|----------|
| `bioclip_classify` | `image/*` or tensor (`clip_image_preprocess_v1`) | `application/json;schema=labels_v1` | yes (image tensors) |

Request metadata can include `TopK` / `topk` for result count.

## Limits

- No text encoder at inference time — classification is dot-product against frozen dataset embeddings
- Dataset name in config must match files under `datasets/`
- Included in `lumen-cli init` presets `curious` and `brave`, not `calm`
