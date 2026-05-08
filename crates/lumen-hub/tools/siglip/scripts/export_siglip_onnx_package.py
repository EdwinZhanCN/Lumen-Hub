#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch>=2.2.0",
#   "transformers>=4.40.0",
#   "huggingface-hub>=0.23.0",
#   "onnx>=1.16.0",
#   "onnxconverter-common>=1.14.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
# ]
# ///

"""Export a Hugging Face SiglipModel into the Lumen SigLIP ONNX layout.

The generated package is rooted at ``--output-dir`` and contains:

    model_info.json
    tokenizer.json
    config/tokenizer/preprocessor artifacts copied from the HF repo
    onnx/text.fp32.onnx      # text_features + last_hidden_state
    onnx/text.fp16.onnx      # text_features + last_hidden_state
    onnx/vision.fp32.onnx    # image_features + last_hidden_state
    onnx/vision.fp16.onnx    # image_features + last_hidden_state

RKNN artifacts are intentionally not produced by this exporter.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
import torch
from huggingface_hub import snapshot_download
from onnxconverter_common import float16
from torch import nn
from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel


TEXT_INPUT_NAMES = ["input_ids", "attention_mask"]
VISION_INPUT_NAMES = ["pixel_values"]
TEXT_OUTPUT_NAME = "text_features"
VISION_OUTPUT_NAME = "image_features"
HIDDEN_OUTPUT_NAME = "last_hidden_state"
DEFAULT_VERSION = "1.0.0"
DEFAULT_ONNX_OPSET = 17

HF_METADATA_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.vocab",
    "merges.txt",
    "vocab.*",
    "README.md",
)
HF_METADATA_IGNORE_PATTERNS = (
    "*.bin",
    "*.safetensors",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.tflite",
    "*.h5",
    "*.msgpack",
    "*.npz",
)


class _TextOutputsWrapper(nn.Module):
    """Expose SigLIP text features plus token hidden states."""

    def __init__(self, model: SiglipModel) -> None:
        super().__init__()
        self.text_model = model.text_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.pooler_output, outputs.last_hidden_state


class _VisionOutputsWrapper(nn.Module):
    """Expose SigLIP image features plus patch-token hidden states."""

    def __init__(self, model: SiglipModel) -> None:
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        return outputs.pooler_output, outputs.last_hidden_state


def _safe_model_name(repo_id: str) -> str:
    return repo_id.rstrip("/").rsplit("/", maxsplit=1)[-1].lower().replace("_", "-")


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _first_positive_int(*values: Any, default: int) -> int:
    for value in values:
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    return default


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _download_and_copy_hf_metadata(
    *,
    repo_id: str,
    revision: str | None,
    output_dir: Path,
    hf_cache_dir: Path | None,
    token: str | None,
    local_files_only: bool,
) -> None:
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=list(HF_METADATA_ALLOW_PATTERNS),
            ignore_patterns=list(HF_METADATA_IGNORE_PATTERNS),
        )
    )

    for source in snapshot.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(snapshot)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _save_required_hf_artifacts(
    *,
    repo_id: str,
    revision: str | None,
    output_dir: Path,
    hf_cache_dir: Path | None,
    token: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
    model: SiglipModel,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=True,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        repo_id,
        revision=revision,
        cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    tokenizer.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    tokenizer_json = output_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise RuntimeError(
            f"{tokenizer_json} was not created. SigLIP text tasks require a fast "
            "Hugging Face tokenizer that can save tokenizer.json."
        )

    return tokenizer, image_processor


def _normalise_resample_name(value: Any) -> str:
    if value is None:
        return "bicubic"

    name = getattr(value, "name", None)
    if name:
        value = name

    text = str(value).lower()
    numeric_names = {
        "0": "nearest",
        "1": "lanczos3",
        "2": "bilinear",
        "3": "bicubic",
    }
    if text in numeric_names:
        return numeric_names[text]
    if "nearest" in text:
        return "nearest"
    if "lanczos" in text:
        return "lanczos3"
    if "bilinear" in text:
        return "bilinear"
    if "bicubic" in text:
        return "bicubic"
    return text


def _float_triplet(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        return default
    parsed = [float(item) for item in value]
    if len(parsed) != 3:
        return default
    return parsed


def _vision_default_image_size(model: SiglipModel) -> int:
    vision_config = getattr(model.config, "vision_config", None)
    if vision_config is None:
        return 224
    return _first_positive_int(
        getattr(vision_config, "image_size", None),
        getattr(vision_config, "image_size_h", None),
        default=224,
    )


def _build_image_preprocess(
    *,
    output_dir: Path,
    image_processor: Any,
    model: SiglipModel,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    processor_dict = getattr(image_processor, "to_dict", None)
    if callable(processor_dict):
        cfg.update(processor_dict())

    preprocessor_config = output_dir / "preprocessor_config.json"
    if preprocessor_config.is_file():
        cfg.update(_load_json(preprocessor_config))

    default_image_size = _vision_default_image_size(model)
    size = _as_mapping(cfg.get("size"))
    crop_size = _as_mapping(cfg.get("crop_size"))

    size_height = size.get("height")
    size_width = size.get("width")
    size_shortest = size.get("shortest_edge")
    crop_height = crop_size.get("height")
    crop_width = crop_size.get("width")

    if isinstance(cfg.get("size"), int):
        size_height = cfg["size"]
        size_width = cfg["size"]
    if isinstance(cfg.get("crop_size"), int):
        crop_height = cfg["crop_size"]
        crop_width = cfg["crop_size"]

    resolved_crop_height = _first_positive_int(
        crop_height,
        size_height,
        size_shortest,
        default=default_image_size,
    )
    resolved_crop_width = _first_positive_int(
        crop_width,
        size_width,
        size_shortest,
        default=resolved_crop_height,
    )
    resize_shortest_edge = _first_positive_int(
        size_shortest,
        min(resolved_crop_height, resolved_crop_width),
        default=default_image_size,
    )

    image_mean = _float_triplet(cfg.get("image_mean"), [0.5, 0.5, 0.5])
    image_std = _float_triplet(cfg.get("image_std"), [0.5, 0.5, 0.5])
    if any(std == 0.0 for std in image_std):
        raise RuntimeError("preprocessor_config.json produced an invalid zero image_std value")

    return {
        "resize_shortest_edge": resize_shortest_edge,
        "crop_size": {
            "width": resolved_crop_width,
            "height": resolved_crop_height,
        },
        "do_resize": bool(cfg.get("do_resize", True)),
        "do_center_crop": bool(cfg.get("do_center_crop", False)),
        "do_rescale": bool(cfg.get("do_rescale", True)),
        "do_normalize": bool(cfg.get("do_normalize", True)),
        "rescale_factor": float(cfg.get("rescale_factor", 1.0 / 255.0)),
        "image_mean": image_mean,
        "image_std": image_std,
        "resample": _normalise_resample_name(cfg.get("resample")),
        "color_space": "rgb",
        "layout": "nchw",
    }


def _infer_sequence_length(model: SiglipModel, override: int | None) -> int:
    if override is not None:
        return override
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        max_positions = _positive_int(getattr(text_config, "max_position_embeddings", None))
        if max_positions is not None:
            return max_positions
    return 64


def _infer_embedding_dim(model: SiglipModel, sequence_length: int) -> int | None:
    for module_name in ("text_projection", "visual_projection", "vision_projection"):
        projection = getattr(model, module_name, None)
        out_features = _positive_int(getattr(projection, "out_features", None))
        if out_features is not None:
            return out_features

    for config in (
        model.config,
        getattr(model.config, "text_config", None),
        getattr(model.config, "vision_config", None),
    ):
        if config is None:
            continue
        for attr in ("projection_size", "projection_dim", "hidden_size"):
            value = _positive_int(getattr(config, attr, None))
            if value is not None:
                return value

    try:
        with torch.inference_mode():
            input_ids = torch.ones((1, sequence_length), dtype=torch.long)
            attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
            output = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return int(output.pooler_output.shape[-1])
    except Exception:
        return None


def _export_text_onnx(
    *,
    model: SiglipModel,
    output_path: Path,
    opset: int,
    batch_size: int,
    sequence_length: int,
) -> None:
    wrapper = _TextOutputsWrapper(model).eval()
    input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_path),
        input_names=TEXT_INPUT_NAMES,
        output_names=[TEXT_OUTPUT_NAME, HIDDEN_OUTPUT_NAME],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            TEXT_OUTPUT_NAME: {0: "batch_size"},
            HIDDEN_OUTPUT_NAME: {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )


def _export_vision_onnx(
    *,
    model: SiglipModel,
    output_path: Path,
    opset: int,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    wrapper = _VisionOutputsWrapper(model).eval()
    pixel_values = torch.zeros((batch_size, num_channels, height, width), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (pixel_values,),
        str(output_path),
        input_names=VISION_INPUT_NAMES,
        output_names=[VISION_OUTPUT_NAME, HIDDEN_OUTPUT_NAME],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            VISION_OUTPUT_NAME: {0: "batch_size"},
            HIDDEN_OUTPUT_NAME: {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )


def _convert_to_fp16(src_path: Path, dst_path: Path) -> None:
    model = onnx.load(str(src_path), load_external_data=True)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save_model(converted, str(dst_path), save_as_external_data=False)


def _graph_output_names(path: Path) -> list[str]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    return [output.name for output in model.graph.output]


def _feature_output_name(path: Path, expected_feature: str) -> str:
    output_names = _graph_output_names(path)
    expected_outputs = [expected_feature, HIDDEN_OUTPUT_NAME]
    if output_names != expected_outputs:
        raise RuntimeError(
            f"{path} must expose exactly {expected_outputs!r}, got {output_names!r}"
        )
    for output_name in output_names:
        if "pooler" in output_name.lower():
            raise RuntimeError(f"{path} exposes forbidden pooler output {output_name!r}")
    return output_names[0]


def _write_model_info(
    *,
    output_dir: Path,
    model_name: str,
    version: str,
    description: str,
    repo_id: str,
    revision: str | None,
    embedding_dim: int | None,
    text_output_name: str,
    vision_output_name: str,
    preprocess: dict[str, Any],
    precisions: list[str],
) -> Path:
    source: dict[str, Any] = {
        "format": "huggingface",
        "repo_id": repo_id,
    }
    if revision is not None:
        source["revision"] = revision

    task_metadata: dict[str, Any] = {
        "tasks": {
            "semantic_text_embed": {
                "component": "text",
                "input_names": TEXT_INPUT_NAMES,
                "output_name": text_output_name,
                "hidden_output_name": HIDDEN_OUTPUT_NAME,
            },
            "semantic_image_embed": {
                "component": "vision",
                "input_names": VISION_INPUT_NAMES,
                "output_name": vision_output_name,
                "hidden_output_name": HIDDEN_OUTPUT_NAME,
                "preprocess": preprocess,
            },
        }
    }
    if embedding_dim is not None:
        task_metadata = {"embedding_dim": embedding_dim, **task_metadata}

    model_info = {
        "name": model_name,
        "version": version,
        "description": description,
        "model_type": "siglip",
        "source": source,
        "runtimes": {
            "onnx": {
                "available": True,
                "components": ["text", "vision"],
                "precisions": precisions,
            }
        },
        "task_metadata": task_metadata,
    }

    path = output_dir / "model_info.json"
    _write_json(path, model_info)
    return path


def _resolve_output_dir(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.cache_dir is None:
        parser.error("one of --output-dir/--output or --cache-dir is required")
    model_name = args.model_name or _safe_model_name(args.repo_id)
    return args.cache_dir / model_name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face SigLIP repo id.")
    parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help="Model package directory to create.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Compatibility alias: write to CACHE_DIR/MODEL_NAME when --output-dir is absent.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        help="Optional Hugging Face download cache directory.",
    )
    parser.add_argument("--model-name", help="model_info.json name. Default: output dir name.")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--description")
    parser.add_argument("--revision")
    parser.add_argument("--token", help="Hugging Face token for private repos.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--skip-fp16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dir = _resolve_output_dir(args, parser)
    model_name = args.model_name or output_dir.name or _safe_model_name(args.repo_id)
    hf_cache_dir = args.hf_cache_dir or args.cache_dir

    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] Loading SiglipModel: {args.repo_id}")
    model = SiglipModel.from_pretrained(
        args.repo_id,
        revision=args.revision,
        cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).eval()

    print("[2/8] Copying Hugging Face metadata/config artifacts")
    _download_and_copy_hf_metadata(
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=output_dir,
        hf_cache_dir=hf_cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )
    _, image_processor = _save_required_hf_artifacts(
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=output_dir,
        hf_cache_dir=hf_cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        model=model,
    )

    preprocess = _build_image_preprocess(
        output_dir=output_dir,
        image_processor=image_processor,
        model=model,
    )
    sequence_length = _infer_sequence_length(model, args.sequence_length)
    height = args.height or int(preprocess["crop_size"]["height"])
    width = args.width or int(preprocess["crop_size"]["width"])

    text_fp32 = onnx_dir / "text.fp32.onnx"
    vision_fp32 = onnx_dir / "vision.fp32.onnx"
    text_fp16 = onnx_dir / "text.fp16.onnx"
    vision_fp16 = onnx_dir / "vision.fp16.onnx"

    print("[3/8] Exporting onnx/text.fp32.onnx")
    _export_text_onnx(
        model=model,
        output_path=text_fp32,
        opset=args.opset,
        batch_size=args.batch_size,
        sequence_length=sequence_length,
    )

    print("[4/8] Exporting onnx/vision.fp32.onnx")
    _export_vision_onnx(
        model=model,
        output_path=vision_fp32,
        opset=args.opset,
        batch_size=args.batch_size,
        num_channels=args.num_channels,
        height=height,
        width=width,
    )

    precisions = ["fp32"]
    if not args.skip_fp16:
        print("[5/8] Creating fp16 ONNX files")
        _convert_to_fp16(text_fp32, text_fp16)
        _convert_to_fp16(vision_fp32, vision_fp16)
        precisions.append("fp16")
    else:
        print("[5/8] Skipping fp16 ONNX files")

    print("[6/8] Checking graph output names")
    text_output_name = _feature_output_name(text_fp32, TEXT_OUTPUT_NAME)
    vision_output_name = _feature_output_name(vision_fp32, VISION_OUTPUT_NAME)
    if not args.skip_fp16:
        _feature_output_name(text_fp16, TEXT_OUTPUT_NAME)
        _feature_output_name(vision_fp16, VISION_OUTPUT_NAME)

    print("[7/8] Writing model_info.json")
    embedding_dim = _infer_embedding_dim(model, sequence_length)
    description = args.description or f"SigLIP deployment assets for {args.repo_id}."
    model_info_path = _write_model_info(
        output_dir=output_dir,
        model_name=model_name,
        version=args.version,
        description=description,
        repo_id=args.repo_id,
        revision=args.revision,
        embedding_dim=embedding_dim,
        text_output_name=text_output_name,
        vision_output_name=vision_output_name,
        preprocess=preprocess,
        precisions=precisions,
    )

    print("[8/8] Done")
    print(f"package={output_dir}")
    print(f"model_info={model_info_path}")
    print(f"text_output_name={text_output_name}")
    print(f"vision_output_name={vision_output_name}")
    print(f"hidden_output_name={HIDDEN_OUTPUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
