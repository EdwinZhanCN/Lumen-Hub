"""Conversion recipes: the single source of truth for what gets built.

A recipe (recipes/<model>.json) fixes, per model:
  * the precision tag of the whole artifact set,
  * every component, its input dtype, its static entry points (shapes),
    which source-graph outputs are kept, and whether its weights are quantized,
  * whether CPU targets compile with data tiling,
  * the fidelity gates checked by `verify`.

Schema (version 1):

{
  "schema": 1,
  "model": "<model repo name>",
  "precision": "fp32" | "w8a32",
  "cpu_data_tiling": true | false,
  "components": [
    {
      "name": "<component>",
      "input_dtype": "float32" | "int64",
      "entries": {"kind": "fixed", "shapes": {"main": [1, 3, 224, 224]}}
               | {"kind": "long_side", "long_side": 960, "multiple": 32,
                  "min_short_side": 64, "batch": 1, "channels": 3},
      "keep_outputs": [<source output index>, ...],
      "output_shapes": [[<dim or "$i">, ...], ...],  # one per kept output;
                                                    # "$i" = input dim i of the entry
      "quantize": true | false,
      "min_quantized_cosine": <float>          # required iff quantize is true
    }
  ],
  "model_info_checks": [                      # optional
    {"path": ["task_metadata", "tasks", "ocr", "recognition", "image_shape"],
     "equals": [3, 48, 320], "absent_means_equal": true}
  ]
  "absent_means_equal": true declares that the hub's serde default for an
  absent key equals "equals", so a missing key passes.
}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .constants import PRECISION_FP32, PRECISION_W8A32, PRECISIONS

SYMBOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class RecipeError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Component:
    name: str
    input_dtype: str
    entries: tuple[Entry, ...]
    keep_outputs: tuple[int, ...]
    output_shapes: tuple[tuple[int | str, ...], ...]
    quantize: bool
    min_quantized_cosine: float | None


@dataclass(frozen=True)
class Recipe:
    path: Path
    model: str
    precision: str
    cpu_data_tiling: bool
    components: tuple[Component, ...]
    model_info_checks: tuple[tuple[tuple[str, ...], object, bool], ...]

    def component(self, name: str) -> Component:
        for component in self.components:
            if component.name == name:
                return component
        raise RecipeError(f"recipe {self.model} has no component {name!r}")


def resolve_output_shape(shape: tuple[int | str, ...], input_shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(input_shape[int(d[1:])] if isinstance(d, str) else d for d in shape)


def long_side_shapes(long_side: int, multiple: int, min_short_side: int, batch: int, channels: int) -> list[Entry]:
    """All (H, W) with max(H, W) == long_side and the other side a multiple of
    `multiple` in [min_short_side, long_side]; landscape and portrait.

    The hub's OCR detection preprocessing produces exactly these shapes (see
    the migration plan, "OCR detection input shapes")."""
    if long_side % multiple or min_short_side % multiple:
        raise RecipeError("long_side and min_short_side must be multiples of `multiple`")
    if not multiple <= min_short_side <= long_side:
        raise RecipeError("min_short_side must be in [multiple, long_side]")
    shapes: list[tuple[int, int]] = []
    for short in range(min_short_side, long_side + 1, multiple):
        shapes.append((long_side, short))
        if short != long_side:
            shapes.append((short, long_side))
    shapes.sort()
    return [Entry(f"main_h{h}_w{w}", (batch, channels, h, w)) for h, w in shapes]


def _entries(raw: dict, where: str) -> tuple[Entry, ...]:
    kind = raw.get("kind")
    if kind == "fixed":
        shapes = raw.get("shapes")
        if not isinstance(shapes, dict) or not shapes:
            raise RecipeError(f"{where}: fixed entries need a non-empty 'shapes' map")
        out = []
        for name, shape in shapes.items():
            if not SYMBOL_RE.match(name):
                raise RecipeError(f"{where}: invalid entry name {name!r}")
            if not isinstance(shape, list) or not shape or not all(isinstance(d, int) and d > 0 for d in shape):
                raise RecipeError(f"{where}: entry {name} needs a list of positive dims")
            out.append(Entry(name, tuple(shape)))
        return tuple(out)
    if kind == "long_side":
        for key in ("long_side", "multiple", "min_short_side", "batch", "channels"):
            if not isinstance(raw.get(key), int) or raw[key] <= 0:
                raise RecipeError(f"{where}: long_side entries need positive int {key!r}")
        return tuple(
            long_side_shapes(raw["long_side"], raw["multiple"], raw["min_short_side"], raw["batch"], raw["channels"])
        )
    raise RecipeError(f"{where}: unknown entries kind {kind!r}")


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("schema") != 1:
        raise RecipeError(f"{path}: unsupported recipe schema {raw.get('schema')!r}")
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise RecipeError(f"{path}: missing model")
    precision = raw.get("precision")
    if precision not in PRECISIONS:
        raise RecipeError(f"{path}: precision must be one of {PRECISIONS}")
    cpu_data_tiling = raw.get("cpu_data_tiling")
    if not isinstance(cpu_data_tiling, bool):
        raise RecipeError(f"{path}: cpu_data_tiling must be a bool")
    components = []
    seen = set()
    for index, comp in enumerate(raw.get("components") or []):
        where = f"{path}: components[{index}]"
        name = comp.get("name")
        if not isinstance(name, str) or not COMPONENT_RE.match(name) or name in seen:
            raise RecipeError(f"{where}: invalid or duplicate component name {name!r}")
        seen.add(name)
        input_dtype = comp.get("input_dtype")
        if input_dtype not in ("float32", "int64"):
            raise RecipeError(f"{where}: input_dtype must be float32 or int64")
        keep = comp.get("keep_outputs")
        if not isinstance(keep, list) or not keep or not all(isinstance(i, int) and i >= 0 for i in keep) or len(set(keep)) != len(keep):
            raise RecipeError(f"{where}: keep_outputs must be a non-empty list of distinct indices")
        output_shapes = comp.get("output_shapes")
        if not isinstance(output_shapes, list) or len(output_shapes) != len(keep):
            raise RecipeError(f"{where}: output_shapes must list one shape per kept output")
        parsed_shapes = []
        for shape in output_shapes:
            if not isinstance(shape, list) or not shape:
                raise RecipeError(f"{where}: invalid output shape {shape!r}")
            for dim in shape:
                ok = (isinstance(dim, int) and dim > 0) or (isinstance(dim, str) and re.fullmatch(r"\$[0-9]+", dim))
                if not ok:
                    raise RecipeError(f"{where}: invalid output dim {dim!r}")
            parsed_shapes.append(tuple(shape))
        quantize = comp.get("quantize")
        if not isinstance(quantize, bool):
            raise RecipeError(f"{where}: quantize must be a bool")
        if precision == PRECISION_FP32 and quantize:
            raise RecipeError(f"{where}: fp32 recipes cannot quantize components")
        gate = comp.get("min_quantized_cosine")
        if quantize != (gate is not None):
            raise RecipeError(f"{where}: min_quantized_cosine is required iff quantize is true")
        if gate is not None and not (isinstance(gate, (int, float)) and 0.0 < gate <= 1.0):
            raise RecipeError(f"{where}: min_quantized_cosine must be in (0, 1]")
        components.append(
            Component(
                name=name,
                input_dtype=input_dtype,
                entries=_entries(comp.get("entries") or {}, where),
                keep_outputs=tuple(keep),
                output_shapes=tuple(parsed_shapes),
                quantize=quantize,
                min_quantized_cosine=float(gate) if gate is not None else None,
            )
        )
    if not components:
        raise RecipeError(f"{path}: recipe has no components")
    if precision == PRECISION_W8A32 and not any(c.quantize for c in components):
        raise RecipeError(f"{path}: w8a32 recipe must quantize at least one component")
    checks = []
    for check in raw.get("model_info_checks") or []:
        key_path = check.get("path")
        if not isinstance(key_path, list) or not key_path or "equals" not in check:
            raise RecipeError(f"{path}: invalid model_info_checks entry {check!r}")
        absent_ok = check.get("absent_means_equal", False)
        if not isinstance(absent_ok, bool):
            raise RecipeError(f"{path}: absent_means_equal must be a bool")
        checks.append((tuple(key_path), check["equals"], absent_ok))
    return Recipe(
        path=path,
        model=model,
        precision=precision,
        cpu_data_tiling=cpu_data_tiling,
        components=tuple(components),
        model_info_checks=tuple(checks),
    )


def check_model_info(recipe: Recipe, model_info: dict) -> None:
    """Fails when model_info.json disagrees with a value the recipe relies on."""
    for key_path, expected, absent_ok in recipe.model_info_checks:
        node: object = model_info
        missing = False
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                missing = True
                break
            node = node[key]
        if missing:
            if absent_ok:
                continue
            raise RecipeError(f"model_info.json lacks {'.'.join(key_path)} required by {recipe.path.name}")
        if node != expected:
            raise RecipeError(
                f"model_info.json {'.'.join(key_path)} = {node!r} but recipe {recipe.path.name} expects {expected!r}"
            )
