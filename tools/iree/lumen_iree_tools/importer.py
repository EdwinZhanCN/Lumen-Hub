"""Imports one or more static-shape ONNX graphs into ONE MLIR module.

Every entry graph becomes a public function named after the entry
(`main`, `main_h960_w736`, ...). Initializers with >= PARAM_MIN_ELEMENTS
elements become `#stream.parameter.named<"model"::"<key>">` globals backed by
the component's .irpa archive; identical tensors (same name, dtype, shape and
bytes) are shared by all entries so weights are stored and loaded once. A
tensor whose name repeats with different contents gets the key
"<name>__<entry>".

This drives IREE's own ONNX importer (iree.compiler.tools.import_onnx) and is
pinned to the IREE release in constants.IREE_VERSION.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from iree.compiler.ir import Context, Location, RankedTensorType
from iree.compiler.tools.import_onnx.importer_externalization_overrides import (
    ELEM_TYPE_TO_SIGNLESS_IR_TYPE,
    IREENodeImporter,
    ParamData,
    onnx_importer,
)

from .constants import PARAM_MIN_ELEMENTS, PARAM_SCOPE


class ImportError_(RuntimeError):
    pass


def _digest(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


class _Params:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.digests: dict[str, str] = {}


def _make_importer_class(params: _Params, entry: str, data_dir: Path):
    class EntryImporter(IREENodeImporter):
        def create_tensor_global(self, t):  # noqa: N802 - IREE API name
            if not isinstance(t, onnx.TensorProto):
                raise ImportError_("graph inputs are never externalized")
            array = numpy_helper.to_array(t, base_dir=str(data_dir))
            digest = _digest(array)
            key = self.sanitize_name(t.name)
            if key in params.digests and params.digests[key] != digest:
                key = f"{key}__{entry}"
            if key in params.digests:
                if params.digests[key] != digest:
                    raise ImportError_(f"parameter key collision for {key}")
                with self._m.context, Location.unknown():
                    return key, RankedTensorType.get(
                        tuple(t.dims), ELEM_TYPE_TO_SIGNLESS_IR_TYPE[t.data_type]()
                    )
            params.arrays[key] = array
            params.digests[key] = digest
            renamed = onnx.TensorProto()
            renamed.CopyFrom(t)
            renamed.name = key
            symbol, tensor_type = super().create_tensor_global(renamed)
            if symbol != key:
                raise ImportError_(f"global symbol {symbol} != parameter key {key}")
            return symbol, tensor_type

    return EntryImporter


def import_entries(entries: list[tuple[str, Path]], out_mlir: Path, out_irpa: Path) -> int:
    """entries: [(function_name, prepared_or_quantized_onnx_path)].
    Writes the MLIR module and the parameter archive; returns the parameter count."""
    import iree.runtime as rt

    if not entries:
        raise ImportError_("no entries to import")
    names = [name for name, _ in entries]
    if len(set(names)) != len(names):
        raise ImportError_(f"duplicate entry names: {names}")

    context = Context()
    params = _Params()
    module_op = None
    for name, path in entries:
        path = Path(path)
        inferred = path.with_name(path.stem + ".inferred.onnx")
        onnx.shape_inference.infer_shapes_path(str(path), str(inferred))
        model = onnx.load_model(str(inferred), load_external_data=False)
        model.graph.name = name
        info = onnx_importer.ModelInfo(model)
        if module_op is None:
            module_op = info.create_module(context=context).operation
        param_data = ParamData(
            param_bit_threshold=None,
            num_elements_threshold=PARAM_MIN_ELEMENTS,
            params_scope=PARAM_SCOPE,
            data_dir=str(path.parent),
            param_path=str(out_irpa),
            input_index_threshold=None,
        )
        importer = IREENodeImporter.define_function(info.main_graph, module_op, param_data)
        # define_function always instantiates IREENodeImporter; swap in the
        # subclass that shares parameters across entries.
        importer.__class__ = _make_importer_class(params, name, path.parent)
        importer.import_all()
        inferred.unlink()

    module_op.verify()
    out_mlir.write_text(module_op.get_asm(assume_verified=True))

    index = rt.ParameterIndex()
    for key in sorted(params.arrays):
        index.add_buffer(key, np.ascontiguousarray(params.arrays[key]))
    index.create_archive_file(str(out_irpa))
    return len(params.arrays)
