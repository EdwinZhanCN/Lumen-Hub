#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26.0",
#   "hnswlib>=0.8.0",
#   "tqdm>=4.66.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import hnswlib
import numpy as np
from tqdm import tqdm


def l2_normalize_block(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def check_row_norms(
    x: np.ndarray,
    sample_rows: int = 10_000,
) -> dict[str, float | int]:
    n = min(sample_rows, x.shape[0])
    sample = np.asarray(x[:n], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)

    return {
        "sample_rows": int(n),
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
    }


def iter_row_blocks(
    x: np.ndarray,
    block_rows: int,
):
    rows = x.shape[0]
    for start in range(0, rows, block_rows):
        end = min(start + block_rows, rows)
        block = np.asarray(x[start:end], dtype=np.float32)
        yield start, end, block


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HNSW index from row-major BioCLIP2 text embeddings."
    )

    parser.add_argument(
        "--emb",
        required=True,
        help="Path to row-major npy, shape [num_labels, 768].",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output HNSW index path, e.g. txt_emb_species.ip.hnsw.bin.",
    )
    parser.add_argument(
        "--space",
        default="ip",
        choices=["ip", "cosine", "l2"],
        help=(
            "HNSW distance space. For normalized CLIP/BioCLIP embeddings, "
            "use ip or cosine. Default: ip."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Expected embedding dimension. Default: 768.",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=65_536,
        help="Rows added per chunk.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=32,
        help="HNSW M. Higher = better recall, more memory. Common: 16/32/48.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW ef_construction. Higher = better recall, slower build. Common: 100-400.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=80,
        help="Default search ef saved in metadata. Must usually be > k. Common: 50-300.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads for hnswlib. 0 means hnswlib default/all available.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help=(
            "Do not normalize vectors before adding. Your file already looks normalized, "
            "but keeping normalization enabled is safer."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output index/meta files.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a small self-query test after saving.",
    )

    args = parser.parse_args()

    emb_path = Path(args.emb).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    if out_path.exists() and not args.force:
        raise FileExistsError(f"Output index already exists: {out_path}. Use --force.")
    if meta_path.exists() and not args.force:
        raise FileExistsError(f"Output meta already exists: {meta_path}. Use --force.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.load(emb_path, mmap_mode="r")

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D row-major matrix [num_labels, dim], got {x.shape}"
        )

    num_labels, dim = x.shape

    if dim != args.dim:
        raise ValueError(
            f"Expected dim={args.dim}, got shape={x.shape}. "
            "Did you accidentally pass the original col-major [768, N] file?"
        )

    norm_stats_before = check_row_norms(x)

    print(f"Embedding:       {emb_path}")
    print(f"Raw shape:       {x.shape}")
    print(f"Raw dtype:       {x.dtype}")
    print(f"Num labels:      {num_labels}")
    print(f"Dim:             {dim}")
    print(f"Norm sample:     {norm_stats_before}")
    print(f"Space:           {args.space}")
    print(f"M:               {args.m}")
    print(f"ef_construction: {args.ef_construction}")
    print(f"ef_search:       {args.ef_search}")
    print(f"Block rows:      {args.block_rows}")
    print(f"Normalize add:   {not args.no_normalize}")

    index = hnswlib.Index(space=args.space, dim=dim)

    index.init_index(
        max_elements=num_labels,
        ef_construction=args.ef_construction,
        M=args.m,
    )

    if args.threads != 0:
        index.set_num_threads(args.threads)

    t0 = time.time()
    added = 0

    total_blocks = (num_labels + args.block_rows - 1) // args.block_rows

    for start, end, block in tqdm(
        iter_row_blocks(x, args.block_rows),
        total=total_blocks,
        desc="Adding embeddings to HNSW",
    ):
        if not args.no_normalize:
            block = l2_normalize_block(block)

        ids = np.arange(start, end, dtype=np.uint64)
        index.add_items(block, ids)
        added += end - start

    index.set_ef(args.ef_search)

    print("Saving HNSW index...")
    index.save_index(str(out_path))

    elapsed = time.time() - t0

    meta = {
        "embedding_path": str(emb_path),
        "index_path": str(out_path),
        "shape": [int(num_labels), int(dim)],
        "layout": "rowmajor_labels_by_dim",
        "source_dtype": str(x.dtype),
        "index_input_dtype": "float32",
        "space": args.space,
        "M": int(args.m),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
        "block_rows": int(args.block_rows),
        "normalize_on_add": not args.no_normalize,
        "norm_stats_before_sample": norm_stats_before,
        "added": int(added),
        "build_seconds": float(elapsed),
    }

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Index:         {out_path}")
    print(f"Meta:          {meta_path}")
    print(f"Build seconds: {elapsed:.2f}")

    if args.quick_test:
        print("\nRunning quick self-query test...")
        index2 = hnswlib.Index(space=args.space, dim=dim)
        index2.load_index(str(out_path), max_elements=num_labels)
        index2.set_ef(max(args.ef_search, 20))

        q = np.asarray(x[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)

        labels, distances = index2.knn_query(q, k=5)

        print("labels:", labels[0].tolist())
        print("distances:", distances[0].tolist())

        if args.space in {"ip", "cosine"}:
            scores = 1.0 - distances[0]
            print("scores:", scores.tolist())


if __name__ == "__main__":
    main()
