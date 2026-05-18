#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch>=2.2.0",
#   "transformers>=4.40.0",
#   "huggingface-hub>=0.23.0",
#   "onnx>=1.16.0",
#   "onnxconverter-common>=1.14.0",
#   "onnxruntime>=1.18.0",
#   "open_clip_torch>=2.24.0",
#   "timm>=0.9.16",
# ]
# ///

"""Export CLIP assets into fixed-shape ONNX and generate repository metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import onnx
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from onnx import TensorProto
from onnxconverter_common import float16
from torch import nn
from transformers import CLIPModel

try:
    import open_clip as _open_clip  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    open_clip: Any = SimpleNamespace()
else:  # pragma: no cover - exercised when optional dependency is installed
    open_clip = _open_clip

try:
    from timm.utils import reparameterize_model as _reparameterize_model
except ImportError:  # pragma: no cover - optional dependency
    reparameterize_model: Any = None
else:  # pragma: no cover - exercised when optional dependency is installed
    reparameterize_model = _reparameterize_model


CLIP_HF_ALLOW_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "*.bin",
    "*.safetensors",
)
DEFAULT_ONNX_OPSET = 17


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a stable pretty-printed JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_shape(value: str) -> tuple[int, ...]:
    """Parse a comma-separated tensor shape."""
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(
            f"shape must be a comma-separated list of positive integers, got {value!r}"
        )
    return dims


def download_hf_artifacts(
        *,
        repo_id: str,
        revision: str,
        output_dir: Path,
        allow_patterns: list[str],
) -> None:
    """Download selected Hugging Face files into `output_dir/upstream`."""
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )
    )
    upstream_dir = output_dir / "upstream"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot.rglob("*"):
        if not source.is_file():
            continue
        target = upstream_dir / source.relative_to(snapshot)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_model_metadata(
        *,
        model_root: Path,
        task: str,
        model_name: str,
        description: str,
        source_format: str,
        source_repo_id: str,
        runtimes: dict[str, Any],
        task_metadata: dict[str, Any],
        version: str,
) -> Path:
    """Write the Lumen `model_info.json` metadata file."""
    model_info = {
        "name": model_name,
        "version": version,
        "description": description,
        "model_type": task,
        "source": {
            "format": source_format,
            "repo_id": source_repo_id,
        },
        "runtimes": runtimes,
        "task_metadata": task_metadata,
    }
    path = model_root / "model_info.json"
    write_json(path, model_info)
    return path


def validate_clip_assets(*, model_root: Path) -> dict[str, Any]:
    """Validate the exported CLIP package enough to catch broken artifacts."""
    required = [
        model_root / "model_info.json",
        model_root / "onnx" / "vision.fp32.onnx",
        model_root / "onnx" / "vision.fp16.onnx",
        model_root / "onnx" / "text.fp32.onnx",
        model_root / "onnx" / "text.fp16.onnx",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CLIP export artifacts: {missing}")

    checked_models: list[str] = []
    for path in required:
        if path.suffix != ".onnx":
            continue
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        checked_models.append(str(path))

    return {
        "ok": True,
        "checked_models": checked_models,
    }


def generate_model_card(*, model_root: Path, template: str) -> Path:
    """Generate a small README for the exported package."""
    model_info = read_json(model_root / "model_info.json")
    path = model_root / "README.md"
    path.write_text(
        "\n".join(
            [
                f"# {model_info['name']}",
                "",
                str(model_info.get("description", "")),
                "",
                f"- Type: `{template}`",
                f"- Source: `{model_info['source']['repo_id']}`",
                "- Runtime: `onnx`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    """Resolved CLIP upstream source information."""

    format: str
    repo_id: str
    revision: str | None = None
    pretrained: str | None = None
    reparameterize: bool = False


class _HFVisionWrapper(nn.Module):
    """Expose `get_image_features()` for Hugging Face CLIP export."""

    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return projected image features."""
        return self.model.get_image_features(pixel_values=pixel_values)


class _HFTextWrapper(nn.Module):
    """Expose `get_text_features()` for Hugging Face CLIP export."""

    def __init__(self, model: CLIPModel) -> None:
        super().__init__()
        self.model = model

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected text features."""
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class _OpenCLIPVisionWrapper(nn.Module):
    """Expose `encode_image()` for OpenCLIP export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return projected image features."""
        return self.model.encode_image(pixel_values)


class _OpenCLIPTextWrapper(nn.Module):
    """Expose `encode_text()` for OpenCLIP export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return projected text features."""
        return self.model.encode_text(input_ids)


def _read_source_spec(model_root: Path) -> _SourceSpec:
    """Resolve CLIP source metadata from `upstream/source.json`."""
    source_metadata = read_json(model_root / "upstream" / "source.json")
    export_options = source_metadata.get("export_options", {})
    return _SourceSpec(
        format=str(source_metadata["format"]),
        repo_id=str(source_metadata["repo_id"]),
        revision=source_metadata.get("revision"),
        pretrained=source_metadata.get("pretrained"),
        reparameterize=bool(export_options.get("reparameterize", False)),
    )


def _maybe_download_hf_upstream(model_root: Path, source_spec: _SourceSpec) -> None:
    """Download missing CLIP Hugging Face artifacts into `upstream/`."""
    if source_spec.format != "huggingface":
        return

    upstream_dir = model_root / "upstream"
    has_weight = any(upstream_dir.glob("*.safetensors")) or any(upstream_dir.glob("*.bin"))
    required_metadata = (
        upstream_dir / "config.json",
        upstream_dir / "tokenizer.json",
        upstream_dir / "tokenizer_config.json",
        upstream_dir / "preprocessor_config.json",
        upstream_dir / "special_tokens_map.json",
    )
    if has_weight and all(path.exists() for path in required_metadata):
        return

    download_hf_artifacts(
        repo_id=source_spec.repo_id,
        revision=source_spec.revision or "main",
        output_dir=model_root,
        allow_patterns=list(CLIP_HF_ALLOW_PATTERNS),
    )


def _resolve_hf_source(model_root: Path) -> tuple[str | Path, str | None, bool]:
    """Return the Hugging Face CLIP source and whether it should be loaded locally."""
    upstream_dir = model_root / "upstream"
    if any(upstream_dir.glob(pattern) for pattern in ("*.safetensors", "*.bin")):
        return upstream_dir, None, True

    source_metadata = _read_source_spec(model_root)
    return (
        str(source_metadata.repo_id),
        source_metadata.revision or "main",
        False,
    )


def _load_hf_clip_model(model_root: Path) -> CLIPModel:
    """Load a Hugging Face CLIPModel from local artifacts or the upstream repo."""
    source, revision, local_files_only = _resolve_hf_source(model_root)
    model = CLIPModel.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    model.eval()
    return model


def _load_openclip_model(model_root: Path) -> nn.Module:
    """Load an OpenCLIP model using its factory APIs."""
    spec = _read_source_spec(model_root)
    if not hasattr(open_clip, "create_model_and_transforms"):
        raise RuntimeError(
            "OpenCLIP export requires the optional 'open_clip_torch' dependency."
        )
    model, _, _ = open_clip.create_model_and_transforms(
        spec.repo_id,
        pretrained=spec.pretrained,
    )
    model.eval()
    if spec.reparameterize:
        if reparameterize_model is None:
            raise RuntimeError(
                "OpenCLIP reparameterization requires the optional 'timm' dependency."
            )
        model = reparameterize_model(model)
        model.eval()
    return model


def _load_clip_model(model_root: Path) -> tuple[str, nn.Module]:
    """Load a CLIP-family model and return `(source_format, model)`."""
    source_spec = _read_source_spec(model_root)
    if source_spec.format == "huggingface":
        return source_spec.format, _load_hf_clip_model(model_root)
    if source_spec.format == "openclip":
        return source_spec.format, _load_openclip_model(model_root)
    raise ValueError(
        f"Unsupported CLIP source format '{source_spec.format}'. "
        "Expected 'huggingface' or 'openclip'."
    )


def _export_vision_model(
        *,
        source_format: str,
        model: nn.Module,
        output_path: Path,
        opset: int,
        image_shape: tuple[int, ...],
) -> None:
    """Export the CLIP vision branch to ONNX."""
    if source_format == "huggingface":
        wrapper = _HFVisionWrapper(cast(CLIPModel, model)).eval()
    else:
        wrapper = _OpenCLIPVisionWrapper(model).eval()
    dummy = torch.zeros(image_shape, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def _export_text_model(
        *,
        source_format: str,
        model: nn.Module,
        output_path: Path,
        opset: int,
        text_shape: tuple[int, ...],
) -> None:
    """Export the CLIP text branch to ONNX."""
    if source_format == "huggingface":
        wrapper = _HFTextWrapper(cast(CLIPModel, model)).eval()
        input_ids = torch.zeros(text_shape, dtype=torch.long)
        attention_mask = torch.ones(text_shape, dtype=torch.long)
        args: tuple[torch.Tensor, ...] = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
    else:
        wrapper = _OpenCLIPTextWrapper(model).eval()
        input_ids = torch.zeros(text_shape, dtype=torch.long)
        args = (input_ids,)
        input_names = ["input_ids"]

    torch.onnx.export(
        wrapper,
        args,
        str(output_path),
        input_names=input_names,
        output_names=["text_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def _tensor_elem_types(model: onnx.ModelProto) -> dict[str, int]:
    """Return tensor element types recorded in graph inputs, outputs, and values."""
    elem_types: dict[str, int] = {}
    values = (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    )
    for value in values:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            elem_types[value.name] = int(tensor_type.elem_type)
    return elem_types


def _fix_fp16_cast_targets(model: onnx.ModelProto) -> int:
    """Retarget stale Cast-to-float nodes introduced by ONNX fp16 conversion."""
    elem_types = _tensor_elem_types(model)
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or len(node.output) != 1:
            continue
        if elem_types.get(node.output[0]) != TensorProto.FLOAT16:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
                fixed += 1
    return fixed


def _convert_to_fp16(src_path: Path, dst_path: Path) -> int:
    """Convert an FP32 ONNX file into FP16."""
    model = onnx.load(str(src_path), load_external_data=True)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    _fix_fp16_cast_targets(converted)
    onnx.save_model(converted, str(dst_path), save_as_external_data=False)
    return int(converted.ir_version)


def _default_model_name(source_spec: _SourceSpec) -> str:
    """Derive a default model name from the upstream identifier."""
    return source_spec.repo_id.rsplit("/", maxsplit=1)[-1].lower()


def _default_description(model_name: str) -> str:
    """Build a default model description."""
    return f"CLIP deployment assets for {model_name}."


def _build_clip_runtimes() -> dict[str, Any]:
    """Return the CLIP runtime inventory."""
    return {
        "onnx": {
            "available": True,
            "components": ["vision", "text"],
            "precisions": ["fp32", "fp16"],
        }
    }


def _build_image_preprocess(preprocessor_config_path: Path) -> dict[str, Any]:
    """Build image preprocess metadata from HuggingFace preprocessor_config.json."""
    cfg = read_json(preprocessor_config_path)
    size = cfg.get("size", {})
    preprocess: dict[str, Any] = {
        "resample": "bicubic",
        "color_space": "rgb",
        "layout": "nchw",
    }
    if isinstance(size, dict) and "shortest_edge" in size:
        preprocess["resize_shortest_edge"] = size["shortest_edge"]
    for key in (
            "do_resize",
            "do_center_crop",
            "do_rescale",
            "do_normalize",
            "rescale_factor",
            "image_mean",
            "image_std",
            "crop_size",
    ):
        if key in cfg:
            preprocess[key] = cfg[key]
    return preprocess


def _build_clip_task_metadata(
        *,
        model_root: Path,
        source_format: str,
        model: nn.Module,
        image_shape: tuple[int, ...],
        text_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Build task-specific metadata for CLIP assets."""
    tokenizer_path = model_root / "upstream" / "tokenizer.json"

    text_input_names = (
        ["input_ids", "attention_mask"]
        if source_format == "huggingface"
        else ["input_ids"]
    )
    image_embed_task: dict[str, Any] = {
        "component": "vision",
        "input_names": ["pixel_values"],
        "output_name": "image_features",
    }
    preprocessor_config_path = model_root / "upstream" / "preprocessor_config.json"
    if preprocessor_config_path.exists():
        image_embed_task["preprocess"] = _build_image_preprocess(
            preprocessor_config_path
        )

    task_metadata: dict[str, Any] = {
        "tasks": {
            "text_embed": {
                "component": "text",
                "input_names": text_input_names,
                "output_name": "text_features",
            },
            "image_embed": image_embed_task,
        },
        "tokenizer": {
            "mode": "tokenizers-json" if tokenizer_path.exists() else "runtime-bound",
        },
    }
    if tokenizer_path.exists():
        task_metadata["tokenizer"]["file"] = "upstream/tokenizer.json"

    if source_format == "huggingface" and isinstance(model, CLIPModel):
        task_metadata["projection_dim"] = int(model.config.projection_dim)
        task_metadata["embedding_dim"] = int(model.config.projection_dim)
        task_metadata["vision_config"] = {
            "image_size": int(model.config.vision_config.image_size),
            "patch_size": int(model.config.vision_config.patch_size),
        }
        task_metadata["text_config"] = {
            "max_position_embeddings": int(
                model.config.text_config.max_position_embeddings
            ),
            "vocab_size": int(model.config.text_config.vocab_size),
        }

    return task_metadata


def export_clip_to_onnx(
        *,
        model_root: Path,
        model_name: str | None = None,
        description: str | None = None,
        version: str = "1.0.0",
        opset: int = DEFAULT_ONNX_OPSET,
        image_shape: tuple[int, ...] = (1, 3, 224, 224),
        text_shape: tuple[int, ...] = (1, 77),
) -> dict[str, Any]:
    """Build the full CLIP asset package with a single entrypoint."""
    model_root = model_root.resolve()
    source_spec = _read_source_spec(model_root)
    _maybe_download_hf_upstream(model_root, source_spec)

    onnx_dir = model_root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    source_format, model = _load_clip_model(model_root)

    vision_fp32 = onnx_dir / "vision.fp32.onnx"
    text_fp32 = onnx_dir / "text.fp32.onnx"
    vision_fp16 = onnx_dir / "vision.fp16.onnx"
    text_fp16 = onnx_dir / "text.fp16.onnx"

    _export_vision_model(
        source_format=source_format,
        model=model,
        output_path=vision_fp32,
        opset=opset,
        image_shape=image_shape,
    )
    _export_text_model(
        source_format=source_format,
        model=model,
        output_path=text_fp32,
        opset=opset,
        text_shape=text_shape,
    )
    _convert_to_fp16(vision_fp32, vision_fp16)
    _convert_to_fp16(text_fp32, text_fp16)

    resolved_model_name = model_name or _default_model_name(source_spec)
    resolved_description = description or _default_description(resolved_model_name)
    task_metadata = _build_clip_task_metadata(
        model_root=model_root,
        source_format=source_format,
        model=model,
        image_shape=image_shape,
        text_shape=text_shape,
    )
    model_info_path = write_model_metadata(
        model_root=model_root,
        task="clip",
        model_name=resolved_model_name,
        description=resolved_description,
        source_format=source_format,
        source_repo_id=source_spec.repo_id,
        runtimes=_build_clip_runtimes(),
        task_metadata=task_metadata,
        version=version,
    )
    validation = validate_clip_assets(model_root=model_root)
    readme_path = generate_model_card(model_root=model_root, template="clip")

    return {
        "model_info": model_info_path,
        "readme": readme_path,
        "validation": validation,
        "artifacts": {
            "vision_fp32": vision_fp32,
            "vision_fp16": vision_fp16,
            "text_fp32": text_fp32,
            "text_fp16": text_fp16,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--description")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--opset", type=int, default=DEFAULT_ONNX_OPSET)
    parser.add_argument("--image-shape", default="1,3,224,224")
    parser.add_argument("--text-shape", default="1,77")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    export_clip_to_onnx(
        model_root=args.model_root.resolve(),
        model_name=args.model_name,
        description=args.description,
        version=args.version,
        opset=args.opset,
        image_shape=parse_shape(args.image_shape),
        text_shape=parse_shape(args.text_shape),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
