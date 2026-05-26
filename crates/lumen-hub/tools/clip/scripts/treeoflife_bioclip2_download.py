#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub>=0.23.0",
#   "numpy>=1.26.0",
#   "tqdm>=4.66.0",
# ]
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from numpy.lib.format import open_memmap
from tqdm import tqdm

DEFAULT_REPO_ID = "imageomics/TreeOfLife-200M"
DEFAULT_NPY_BASENAME = "txt_emb_species.npy"
DEFAULT_JSON_BASENAME = "txt_emb_species.json"
DEFAULT_EMBEDDING_DIM = 768


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 64) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_file_by_basename(
    repo_id: str,
    basename: str,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> str:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)

    matches = [f for f in files if Path(f).name == basename]

    if not matches:
        raise FileNotFoundError(
            f"Could not find {basename!r} in {repo_id}. "
            f"Try checking the HuggingFace repo file tree manually."
        )

    matches.sort(key=lambda p: (0 if "embeddings" in p.lower() else 1, len(p)))
    return matches[0]


def download_hf_file(
    repo_id: str,
    filename: str,
    out_dir: Path,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=filename,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def check_colmajor_norms(
    arr: np.ndarray,
    sample_cols: int = 10_000,
) -> dict[str, float | int | str]:
    """
    For original BioCLIP2 layout: [dim, num_labels].
    Each column is one embedding.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    n = min(sample_cols, arr.shape[1])
    sample = np.asarray(arr[:, :n], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=0)

    return {
        "layout": "colmajor_dim_by_labels",
        "sample_cols": int(n),
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
    }


def check_rowmajor_norms(
    arr: np.ndarray,
    sample_rows: int = 10_000,
) -> dict[str, float | int | str]:
    """
    For output layout: [num_labels, dim].
    Each row is one embedding.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")

    n = min(sample_rows, arr.shape[0])
    sample = np.asarray(arr[:n], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)

    return {
        "layout": "rowmajor_labels_by_dim",
        "sample_rows": int(n),
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
    }


def convert_colmajor_npy_to_rowmajor_fp16(
    src_npy: Path,
    dst_npy: Path,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    chunk_cols: int = 65_536,
    normalize: bool = True,
    force: bool = False,
) -> dict:
    if dst_npy.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {dst_npy}. Use --force to overwrite."
        )

    src = np.load(src_npy, mmap_mode="r")

    if src.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={src.shape}")

    if src.shape[0] != embedding_dim:
        raise ValueError(
            f"Expected source shape [{embedding_dim}, num_labels], got {src.shape}. "
            "This script is specifically for BioCLIP2 txt_emb_species.npy col-major layout."
        )

    dim, num_labels = src.shape

    if force and dst_npy.exists():
        dst_npy.unlink()

    # Output is [num_labels, dim], each row is contiguous.
    dst = open_memmap(
        dst_npy,
        mode="w+",
        dtype=np.float16,
        shape=(num_labels, dim),
    )

    norm_stats_before = check_colmajor_norms(src)

    for start in tqdm(
        range(0, num_labels, chunk_cols),
        desc="Normalize columns, transpose, save fp16",
    ):
        end = min(start + chunk_cols, num_labels)

        # Source block: [768, block_labels]
        block = np.asarray(src[:, start:end], dtype=np.float32)

        if normalize:
            # Each column is one species/text embedding.
            norms = np.linalg.norm(block, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            block = block / norms

        # Output block: [block_labels, 768]
        dst[start:end] = block.T.astype(np.float16)

    dst.flush()

    converted = np.load(dst_npy, mmap_mode="r")
    norm_stats_after = check_rowmajor_norms(converted)

    return {
        "source": str(src_npy),
        "output": str(dst_npy),
        "source_shape": [int(dim), int(num_labels)],
        "output_shape": [int(num_labels), int(dim)],
        "source_layout": "colmajor_dim_by_labels",
        "output_layout": "rowmajor_labels_by_dim",
        "embedding_dim": int(dim),
        "num_labels": int(num_labels),
        "source_dtype": str(src.dtype),
        "output_dtype": "float16",
        "normalize": bool(normalize),
        "chunk_cols": int(chunk_cols),
        "norm_stats_before_sample": norm_stats_before,
        "norm_stats_after_sample": norm_stats_after,
        "source_size_bytes": int(src_npy.stat().st_size),
        "output_size_bytes": int(dst_npy.stat().st_size),
    }


from typing import Any

def normalize_label_item(item: Any, idx: int) -> dict[str, Any]:
    RANK_KEYS = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    if isinstance(item, dict):
        obj = dict(item)
        if "class_name" in obj and "class" not in obj:
            obj["class"] = obj["class_name"]
        if "taxon_class" in obj and "class" not in obj:
            obj["class"] = obj["taxon_class"]
        species = (
            obj.get("scientific_name")
            or obj.get("name")
            or obj.get("canonical_name")
            or obj.get("species")
            or ""
        )
        if species and "scientific_name" not in obj:
            obj["scientific_name"] = str(species)
        obj.setdefault("common_name", "")
        return obj

    if isinstance(item, (list, tuple)):
        common_name = ""
        ranks_raw: Any = item

        if len(item) >= 1 and isinstance(item[0], (list, tuple)):
            ranks_raw = item[0]
            if len(item) >= 2 and item[1] is not None:
                common_name = str(item[1])
        else:
            ranks_raw = item

        if not isinstance(ranks_raw, (list, tuple)):
            raise ValueError(
                f"Unsupported nested label format at index {idx}: {item!r}"
            )

        obj: dict[str, Any] = {}
        for key, value in zip(RANK_KEYS, list(ranks_raw)[:7]):
            obj[key] = "" if value is None else str(value)

        for key in RANK_KEYS:
            obj.setdefault(key, "")

        genus = obj.get("genus", "").strip()
        species_epithet = obj.get("species", "").strip()

        if genus and species_epithet:
            scientific_name = f"{genus} {species_epithet}"
        else:
            scientific_name = ""
            for key in reversed(RANK_KEYS):
                value = obj.get(key, "").strip()
                if value:
                    scientific_name = value
                    break

        obj["scientific_name"] = scientific_name
        obj["name"] = scientific_name
        obj["common_name"] = common_name
        return obj

    if isinstance(item, str):
        return {
            "kingdom": "",
            "phylum": "",
            "class": "",
            "order": "",
            "family": "",
            "genus": "",
            "species": item,
            "scientific_name": item,
            "name": item,
            "common_name": "",
        }

    raise ValueError(
        f"Unsupported label item type at index {idx}: {type(item)}; value={item!r}"
    )


def load_labels(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        if "labels" in raw and isinstance(raw["labels"], list):
            items = raw["labels"]
        elif "data" in raw and isinstance(raw["data"], list):
            items = raw["data"]
        else:
            try:
                pairs = sorted(raw.items(), key=lambda kv: int(kv[0]))
                items = [v for _, v in pairs]
            except Exception as e:
                raise ValueError(
                    "Unsupported labels json structure. Expected list, "
                    "{'labels': [...]}, {'data': [...]}, or numeric-key dict."
                ) from e
    else:
        raise ValueError(f"Unsupported labels json root type: {type(raw)}")

    return [normalize_label_item(item, i) for i, item in enumerate(items)]


def normalize_and_save_label_json(src_json: Path, dst_json: Path, force: bool = False) -> None:
    if dst_json.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {dst_json}. Use --force to overwrite."
        )

    if src_json.resolve() == dst_json.resolve():
        return

    print(f"Normalizing labels from {src_json} to {dst_json}...")
    normalized = load_labels(src_json)

    # Output clean standard taxonomy keys + common_name
    output_labels = []
    for item in normalized:
        output_labels.append({
            "kingdom": item.get("kingdom", "").strip(),
            "phylum": item.get("phylum", "").strip(),
            "class": item.get("class", "").strip(),
            "order": item.get("order", "").strip(),
            "family": item.get("family", "").strip(),
            "genus": item.get("genus", "").strip(),
            "species": item.get("species", "").strip(),
            "common_name": item.get("common_name", "").strip(),
        })

    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(output_labels, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download BioCLIP2 TreeOfLife text embeddings from HuggingFace, "
            "convert txt_emb_species.npy [768, num_species] to row-major fp16 "
            "[num_species, 768], and save txt_emb_species.json."
        )
    )

    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HuggingFace revision / branch / commit.",
    )
    parser.add_argument(
        "--out-dir",
        default="bioclip2_tol_embeddings",
        help="Output directory.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Expected embedding dimension. Default: {DEFAULT_EMBEDDING_DIM}",
    )
    parser.add_argument(
        "--chunk-cols",
        type=int,
        default=65_536,
        help="Species/text-label columns processed per chunk during conversion.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before saving fp16.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute sha256 for output files. This can take a while.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    raw_dir = out_dir / "raw"
    converted_dir = out_dir / "converted"

    raw_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo:       {args.repo_id}")
    print(f"Output dir: {out_dir}")

    print("\nFinding files in HuggingFace dataset...")
    npy_repo_path = find_file_by_basename(
        repo_id=args.repo_id,
        basename=DEFAULT_NPY_BASENAME,
        revision=args.revision,
    )
    json_repo_path = find_file_by_basename(
        repo_id=args.repo_id,
        basename=DEFAULT_JSON_BASENAME,
        revision=args.revision,
    )

    print(f"Found npy:  {npy_repo_path}")
    print(f"Found json: {json_repo_path}")

    print("\nDownloading txt_emb_species.npy...")
    raw_npy = download_hf_file(
        repo_id=args.repo_id,
        filename=npy_repo_path,
        out_dir=raw_dir,
        revision=args.revision,
    )

    print("\nDownloading txt_emb_species.json...")
    raw_json = download_hf_file(
        repo_id=args.repo_id,
        filename=json_repo_path,
        out_dir=raw_dir,
        revision=args.revision,
    )

    dst_npy = converted_dir / "txt_emb_species.rowmajor.norm.f16.npy"
    dst_json = converted_dir / "txt_emb_species.json"
    manifest_path = converted_dir / "manifest.rowmajor.norm.f16.json"

    print("\nConverting col-major npy -> row-major fp16 npy...")
    conversion_info = convert_colmajor_npy_to_rowmajor_fp16(
        src_npy=raw_npy,
        dst_npy=dst_npy,
        embedding_dim=args.embedding_dim,
        chunk_cols=args.chunk_cols,
        normalize=not args.no_normalize,
        force=args.force,
    )

    print("\nNormalizing txt_emb_species.json...")
    normalize_and_save_label_json(raw_json, dst_json, force=args.force)

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "raw_npy_repo_path": npy_repo_path,
        "raw_json_repo_path": json_repo_path,
        "files": {
            "raw_npy": str(raw_npy),
            "raw_json": str(raw_json),
            "rowmajor_fp16_npy": str(dst_npy),
            "label_json": str(dst_json),
        },
        "conversion": conversion_info,
    }

    if args.hash:
        print("\nComputing sha256...")
        manifest["sha256"] = {
            "rowmajor_fp16_npy": sha256_file(dst_npy),
            "label_json": sha256_file(dst_json),
        }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Row-major FP16 embeddings: {dst_npy}")
    print(f"Labels JSON:              {dst_json}")
    print(f"Manifest:                 {manifest_path}")
    print("\nNorm stats after conversion sample:")
    print(json.dumps(conversion_info["norm_stats_after_sample"], indent=2))


if __name__ == "__main__":
    main()
