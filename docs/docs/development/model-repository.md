---
sidebar_position: 3
---

# Model Repository Spec

Beta model downloads use public Hugging Face repositories under:

```text
Lumilio-Photos/{model}
```

`{model}` comes from `services.<service>.models.<alias>.model` in `lumen-config`.

## Required Layout

```text
model_info.json
onnx/<component>.<precision>.onnx
mnn/<component>.<precision>.mnn
mnn-llm/config.json
mnn-llm/<runtime package files>
rknn/<component>.<precision>.rknn
<root metadata files>
datasets/<dataset>.json
datasets/<dataset>.npy
datasets/<dataset>.bin
```

`model_info.json` is downloaded first and validated with the Lumen `ModelInfo` schema. The requested runtime and precision must exist in `model_info.runtimes`.

## Runtime Artifacts

For a config model like:

```json
{
  "model": "siglip-so400m",
  "runtime": "onnx",
  "precision": "fp32"
}
```

Lumen reads `model_info.runtimes.onnx.components` and downloads:

```text
onnx/<component>.fp32.onnx
```

For `runtime: "mnn"`, the path is `mnn/<component>.<precision>.mnn`.

For `runtime: "mnn-llm"`, Lumen treats `{model}/mnn-llm/` as an MNN LLM
package directory. The runtime entrypoint is:

```text
mnn-llm/config.json
```

The model should declare this runtime with a package marker, for example:

```json
{
  "runtimes": {
    "mnn-llm": {
      "available": true,
      "components": ["config"],
      "precisions": ["mixed"]
    }
  }
}
```

Unlike ordinary tensor runtimes, `mnn-llm` does not derive artifact filenames
from component and precision. Lumen downloads the root-level metadata files and
all non-hidden files directly under `mnn-llm/`; the MNN LLM runtime interprets
the package files through `mnn-llm/config.json`.

## Root Files

Lumen downloads all root-level repository files except `.npy` and `.bin`.

If the model config sets `dataset`, Lumen also downloads every root-level file matching:

```text
datasets/{dataset}.*
```

This dataset rule includes `.json`, `.npy`, and `.bin`. Root-level `{dataset}.*` files are not downloaded.

Existing files in the cache are skipped. New files are written to a temporary file and atomically renamed into place.
