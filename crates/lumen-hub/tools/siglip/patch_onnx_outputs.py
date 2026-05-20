#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnx>=1.16.0",
# ]
# ///

"""
Patches a SigLIP (or similar) ONNX model:
  - Removes specified outputs (e.g., ``last_hidden_state``).
  - Renames outputs (e.g., ``pooler_output`` → ``text_features`` / ``image_features``).

Usage:
  # Vision: keep only pooler_output as image_features
  uv run tools/siglip/patch_onnx_outputs.py vision.fp32.onnx \
      --remove last_hidden_state \
      --rename pooler_output:image_features

  # Text: keep only pooler_output as text_features
  uv run tools/siglip/patch_onnx_outputs.py text.fp32.onnx \
      --remove last_hidden_state \
      --rename pooler_output:text_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def patch_onnx(
    model_path: str,
    remove: list[str],
    rename: dict[str, str],
    output_path: str | None = None,
) -> None:
    model = onnx.load(model_path)
    graph = model.graph

    # ── 1. 打印当前输出 ────────────────────────────────────────
    output_info = {}
    for o in graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        output_info[o.name] = dims
    print(f"当前输出: {output_info}")

    # ── 2. 删除指定输出 ────────────────────────────────────────
    orig_count = len(graph.output)
    new_outputs = [o for o in graph.output if o.name not in remove]
    del graph.output[:]
    graph.output.extend(new_outputs)
    removed = orig_count - len(graph.output)
    if removed:
        print(f"已移除 {removed} 个输出: {remove}")

    # 检查需要重命名的输出是否还在
    for old_name in rename:
        if old_name not in {o.name for o in graph.output}:
            print(f"警告: 输出 `{old_name}` 已不存在，跳过重命名")

    # ── 3. 重命名输出 ──────────────────────────────────────────
    #     需要改两处:
    #     a) graph.output 中 ValueInfoProto 的 name
    #     b) graph.node 中产生该输出的节点的 output 名称
    for output in graph.output:
        if output.name in rename:
            new_name = rename[output.name]
            print(f"输出 {output.name} → {new_name}")
            output.name = new_name

    for node in graph.node:
        new_outputs = list(node.output)
        changed = False
        for i, out_name in enumerate(node.output):
            if out_name in rename:
                new_outputs[i] = rename[out_name]
                changed = True
        if changed:
            del node.output[:]
            node.output.extend(new_outputs)
            print(f"  节点 {node.op_type} (name={node.name}) output 已同步更新")

    # ── 4. 验证 ────────────────────────────────────────────────
    final_names = [o.name for o in graph.output]
    print(f"修复后输出: {final_names}")

    for old_name in remove:
        if old_name in final_names:
            raise RuntimeError(f"输出 `{old_name}` 未被移除")

    for old_name, new_name in rename.items():
        if old_name in final_names:
            raise RuntimeError(f"输出 `{old_name}` 未被重命名")
        if new_name not in final_names:
            raise RuntimeError(f"新输出名 `{new_name}` 不在最终输出中")

    if not final_names:
        raise RuntimeError("模型没有任何输出")

    # ── 5. 保存 ────────────────────────────────────────────────
    dest = Path(output_path if output_path else model_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dest))
    print(f"已保存到: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Patches an ONNX model: remove specified outputs and rename others. "
            "Useful for fixing SigLIP/CLIP models where you only want pooler_output "
            "and want to drop last_hidden_state."
        )
    )
    parser.add_argument("model", type=str, help="Path to the ONNX model file")
    parser.add_argument(
        "--remove",
        type=str,
        action="append",
        default=[],
        help="Output name to remove (can be specified multiple times)",
    )
    parser.add_argument(
        "--rename",
        type=str,
        action="append",
        default=[],
        help="Rename in OLD:NEW format, e.g. pooler_output:image_features",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: overwrite in-place)",
    )
    args = parser.parse_args()

    rename_dict: dict[str, str] = {}
    for r in args.rename:
        if ":" not in r:
            parser.error(f"--rename 格式应为 OLD:NEW，收到 `{r}`")
        old, new = r.split(":", 1)
        rename_dict[old] = new

    patch_onnx(args.model, args.remove, rename_dict, args.out)


if __name__ == "__main__":
    main()
