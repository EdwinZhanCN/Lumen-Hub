#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26.0",
#   "tqdm>=4.66.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

RANK_KEYS = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


DEFAULT_DOMAINS = [
    "plant",
    "bird",
    "insect",
    "mammal",
    "fish",
    "flower",
    "tree",
    "reptile",
    "amphibian",
]


FISH_CLASSES = {
    "Actinopteri",
    "Actinopterygii",
    "Chondrichthyes",
    "Elasmobranchii",
    "Holocephali",
    "Sarcopterygii",
    "Myxini",
    "Petromyzonti",
    "Cephalaspidomorphi",
}


FLOWER_HINTS = {
    "Magnoliophyta",
    "Magnoliopsida",
    "Liliopsida",
    "Angiospermae",
    "angiosperm",
    "flowering",
}


TREE_NAME_HINTS = {
    "tree",
    "oak",
    "pine",
    "maple",
    "spruce",
    "fir",
    "cedar",
    "birch",
    "willow",
    "poplar",
    "eucalyptus",
    "sequoia",
    "redwood",
    "cypress",
    "elm",
    "ash",
    "beech",
    "chestnut",
    "walnut",
    "quercus",
    "pinus",
    "acer",
    "picea",
    "abies",
    "cedrus",
    "betula",
    "salix",
    "populus",
    "eucalyptus",
    "sequoiadendron",
    "sequoia",
    "cupressus",
    "ulmus",
    "fraxinus",
    "fagus",
    "castanea",
    "juglans",
}


def norm_domain(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def get_field(obj: dict[str, Any], *names: str) -> str:
    for name in names:
        if name in obj and obj[name] is not None:
            return str(obj[name])

    lower_map = {str(k).lower(): k for k in obj.keys()}

    for name in names:
        key = lower_map.get(name.lower())
        if key is not None and obj[key] is not None:
            return str(obj[key])

    return ""


def normalize_label_item(item: Any, idx: int) -> dict[str, Any]:
    """
    Normalize one txt_emb_species.json item into a dict.

    TreeOfLife-200M observed format:

      [
        [kingdom, phylum, class, order, family, genus, species],
        common_name
      ]

    Example:

      [
        ["Animalia", "", "", "", "", "Acanthodes", "pristis"],
        ""
      ]

    Output:

      {
        "kingdom": "Animalia",
        "phylum": "",
        "class": "",
        "order": "",
        "family": "",
        "genus": "Acanthodes",
        "species": "pristis",
        "scientific_name": "Acanthodes pristis",
        "common_name": "",
        "_global_id": 0,
        "_raw_label": ...
      }
    """
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
        obj["_global_id"] = idx
        obj["_raw_label"] = item
        return obj

    if isinstance(item, (list, tuple)):
        common_name = ""
        ranks_raw: Any = item

        # TreeOfLife-200M format:
        #   [ [7-rank taxonomy], common_name ]
        if len(item) >= 1 and isinstance(item[0], (list, tuple)):
            ranks_raw = item[0]

            if len(item) >= 2 and item[1] is not None:
                common_name = str(item[1])
        else:
            # Fallback: direct 7-rank list.
            ranks_raw = item

        if not isinstance(ranks_raw, (list, tuple)):
            raise ValueError(
                f"Unsupported nested label format at index {idx}: {item!r}"
            )

        obj: dict[str, Any] = {}

        for key, value in zip(RANK_KEYS, list(ranks_raw)[:7]):
            obj[key] = "" if value is None else str(value)

        # Fill missing rank keys.
        for key in RANK_KEYS:
            obj.setdefault(key, "")

        if len(ranks_raw) > 7:
            obj["_extra_rank_fields"] = list(ranks_raw[7:])

        genus = obj.get("genus", "").strip()
        species_epithet = obj.get("species", "").strip()

        if genus and species_epithet:
            scientific_name = f"{genus} {species_epithet}"
        else:
            # Fall back to the most specific non-empty rank.
            scientific_name = ""
            for key in reversed(RANK_KEYS):
                value = obj.get(key, "").strip()
                if value:
                    scientific_name = value
                    break

        obj["scientific_name"] = scientific_name
        obj["name"] = scientific_name
        obj["common_name"] = common_name
        obj["_global_id"] = idx
        obj["_raw_label"] = item

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
            "_global_id": idx,
            "_raw_label": item,
        }

    raise ValueError(
        f"Unsupported label item type at index {idx}: {type(item)}; value={item!r}"
    )


def load_labels(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

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


def classify_domains(label: dict[str, Any]) -> set[str]:
    kingdom = get_field(label, "kingdom")
    phylum = get_field(label, "phylum")
    class_name = get_field(label, "class", "class_name", "taxon_class")
    order = get_field(label, "order")
    family = get_field(label, "family")
    genus = get_field(label, "genus")
    species = get_field(label, "species", "scientific_name", "name")
    scientific_name = get_field(label, "scientific_name", "name")
    common_name = get_field(label, "common_name", "vernacular_name")

    k_low = kingdom.lower()
    p_low = phylum.lower()
    c_low = class_name.lower()

    text = " ".join(
        [
            kingdom,
            phylum,
            class_name,
            order,
            family,
            genus,
            species,
            scientific_name,
            common_name,
        ]
    )
    text_low = text.lower()

    domains: set[str] = set()

    if k_low == "plantae":
        domains.add("plant")

        # "flower" is heuristic, because it is not a strict Linnaean rank.
        if (
            any(h.lower() in text_low for h in FLOWER_HINTS)
            or "magnoliophyta" in p_low
            or "angiosperm" in text_low
        ):
            domains.add("flower")

        # "tree" is heuristic, not a strict taxonomic group.
        if any(h in text_low for h in TREE_NAME_HINTS):
            domains.add("tree")

    if k_low == "animalia":
        if c_low == "aves":
            domains.add("bird")

        if c_low == "mammalia":
            domains.add("mammal")

        if c_low == "insecta":
            domains.add("insect")

        if c_low == "reptilia":
            domains.add("reptile")

        if c_low == "amphibia":
            domains.add("amphibian")

        if class_name in FISH_CLASSES or c_low in {x.lower() for x in FISH_CLASSES}:
            domains.add("fish")

        if any(
            x in text_low
            for x in [
                "actinopterygii",
                "actinopteri",
                "chondrichthyes",
                "elasmobranchii",
                "holocephali",
                "sarcopterygii",
                "myxini",
                "petromyzonti",
            ]
        ):
            domains.add("fish")

    return domains


def select_ids(
    labels: list[dict[str, Any]],
    wanted_domains: set[str],
) -> tuple[np.ndarray, list[dict[str, Any]], Counter]:
    ids: list[int] = []
    selected_labels: list[dict[str, Any]] = []
    domain_counts: Counter = Counter()
    kingdom_counts: Counter = Counter()
    class_counts: Counter = Counter()

    for i, label in enumerate(labels):
        kingdom = get_field(label, "kingdom") or "<empty>"
        class_name = get_field(label, "class", "class_name", "taxon_class") or "<empty>"

        kingdom_counts[kingdom] += 1
        class_counts[class_name] += 1

        domains = classify_domains(label)
        matched = domains & wanted_domains

        if matched:
            enriched = dict(label)
            enriched["_global_id"] = i
            enriched["_domains"] = sorted(domains)
            enriched["_matched_domains"] = sorted(matched)

            ids.append(i)
            selected_labels.append(enriched)

            for d in matched:
                domain_counts[d] += 1

    print("\nTop kingdoms in source labels:")
    for k, v in kingdom_counts.most_common(20):
        print(f"  {k}: {v}")

    print("\nTop classes in source labels:")
    for k, v in class_counts.most_common(30):
        print(f"  {k}: {v}")

    return np.asarray(ids, dtype=np.int64), selected_labels, domain_counts


def check_row_norms(arr: np.ndarray, sample_rows: int = 10_000) -> dict[str, Any]:
    n = min(sample_rows, arr.shape[0])
    sample = np.asarray(arr[:n], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)

    return {
        "sample_rows": int(n),
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
    }


def copy_embedding_rows(
    emb_path: Path,
    out_path: Path,
    ids: np.ndarray,
    block_rows: int,
    force: bool,
) -> dict[str, Any]:
    if out_path.exists() and not force:
        raise FileExistsError(f"Output exists: {out_path}. Use --force.")

    x = np.load(emb_path, mmap_mode="r")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got {x.shape}")

    n, dim = x.shape

    if ids.size == 0:
        raise ValueError("No labels matched the requested domains.")

    if ids.min() < 0 or ids.max() >= n:
        raise ValueError(
            f"Selected ids out of range. ids range=({ids.min()}, {ids.max()}), "
            f"embedding rows={n}"
        )

    if force and out_path.exists():
        out_path.unlink()

    y = open_memmap(
        out_path,
        mode="w+",
        dtype=x.dtype,
        shape=(ids.size, dim),
    )

    # Advanced indexing with a large id list can allocate a lot, so copy in blocks.
    for start in tqdm(range(0, ids.size, block_rows), desc="Copying subset embeddings"):
        end = min(start + block_rows, ids.size)
        block_ids = ids[start:end]
        y[start:end] = np.asarray(x[block_ids], dtype=x.dtype)

    y.flush()

    z = np.load(out_path, mmap_mode="r")
    norm_sample = check_row_norms(z)

    return {
        "source_shape": [int(n), int(dim)],
        "output_shape": [int(ids.size), int(dim)],
        "dtype": str(x.dtype),
        "norm_sample": norm_sample,
        "source_size_bytes": int(emb_path.stat().st_size),
        "output_size_bytes": int(out_path.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a smaller BioCLIP2 domain pack from row-major species "
            "embeddings and TreeOfLife-200M txt_emb_species.json labels."
        )
    )

    parser.add_argument(
        "--emb",
        required=True,
        help="Row-major embedding npy, shape [num_labels, 768].",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="txt_emb_species.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for domain pack.",
    )
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help=f"Comma-separated domains. Default: {','.join(DEFAULT_DOMAINS)}",
    )
    parser.add_argument(
        "--name",
        default="core_domains",
        help="Pack name prefix.",
    )
    parser.add_argument(
        "--block-rows",
        type=int,
        default=65_536,
        help="Rows copied per chunk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files.",
    )

    args = parser.parse_args()

    emb_path = Path(args.emb).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_domains = {norm_domain(x) for x in args.domains.split(",") if x.strip()}

    print(f"Embedding: {emb_path}")
    print(f"Labels:    {labels_path}")
    print(f"Out dir:   {out_dir}")
    print(f"Domains:   {sorted(wanted_domains)}")

    labels = load_labels(labels_path)
    print(f"\nLoaded labels: {len(labels)}")

    x = np.load(emb_path, mmap_mode="r")
    print(f"Embedding shape: {x.shape}, dtype={x.dtype}")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got {x.shape}")

    if len(labels) != x.shape[0]:
        raise ValueError(
            f"Label count does not match embedding rows: "
            f"labels={len(labels)}, emb_rows={x.shape[0]}"
        )

    ids, selected_labels, domain_counts = select_ids(labels, wanted_domains)

    print(f"\nSelected labels: {ids.size}")
    print("Matched domain counts:")
    for k, v in domain_counts.most_common():
        print(f"  {k}: {v}")

    prefix = args.name

    out_emb = out_dir / f"txt_emb_species.{prefix}.rowmajor.norm.f16.npy"
    out_labels = out_dir / f"txt_emb_species.{prefix}.json"
    out_ids = out_dir / f"txt_emb_species.{prefix}.global_ids.npy"
    out_manifest = out_dir / f"txt_emb_species.{prefix}.manifest.json"

    if not args.force:
        for p in [out_emb, out_labels, out_ids, out_manifest]:
            if p.exists():
                raise FileExistsError(f"Output exists: {p}. Use --force.")

    np.save(out_ids, ids)

    copy_info = copy_embedding_rows(
        emb_path=emb_path,
        out_path=out_emb,
        ids=ids,
        block_rows=args.block_rows,
        force=args.force,
    )

    clean_labels = []
    for item in selected_labels:
        clean_labels.append({
            "kingdom": item.get("kingdom", "").strip(),
            "phylum": item.get("phylum", "").strip(),
            "class": item.get("class", "").strip(),
            "order": item.get("order", "").strip(),
            "family": item.get("family", "").strip(),
            "genus": item.get("genus", "").strip(),
            "species": item.get("species", "").strip(),
            "common_name": item.get("common_name", "").strip(),
        })

    out_labels.write_text(
        json.dumps(clean_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "source_embedding": str(emb_path),
        "source_labels": str(labels_path),
        "domains_requested": sorted(wanted_domains),
        "selected_count": int(ids.size),
        "domain_counts": dict(domain_counts),
        "files": {
            "embedding": str(out_emb),
            "labels": str(out_labels),
            "global_ids": str(out_ids),
            "manifest": str(out_manifest),
        },
        "copy": copy_info,
        "notes": [
            "Output embedding is row-major [selected_labels, dim].",
            "global_ids maps subset row index to original TreeOfLife/BioCLIP label index.",
            "flower and tree are heuristic domains, not strict taxonomic ranks.",
            "TreeOfLife-200M txt_emb_species.json items are normalized from [[7-rank taxonomy], common_name].",
        ],
    }

    out_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Embedding:  {out_emb}")
    print(f"Labels:     {out_labels}")
    print(f"Global IDs: {out_ids}")
    print(f"Manifest:   {out_manifest}")
    print("\nNorm sample:")
    print(json.dumps(copy_info["norm_sample"], indent=2))


if __name__ == "__main__":
    main()
