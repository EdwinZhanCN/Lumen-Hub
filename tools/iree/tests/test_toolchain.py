"""Self-tests for the Lumen IREE toolchain.

Run from tools/iree with the pinned virtualenv active:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from lumen_iree_tools import qa_fixture
from lumen_iree_tools.constants import ALL_TARGETS, TARGETS, TARGETS_BY_PRECISION, irpa_name, vmfb_name
from lumen_iree_tools.quantize import MIN_QUANT_ELEMENTS, quantize_model, quantize_tensor
from lumen_iree_tools.recipes import RecipeError, load_recipe, long_side_shapes, resolve_output_shape

TOOLS = Path(__file__).resolve().parent.parent


def _write_recipe(tmp: Path, body: dict) -> Path:
    path = tmp / "r.json"
    path.write_text(json.dumps(body))
    return path


BASE_COMPONENT = {
    "name": "net",
    "input_dtype": "float32",
    "entries": {"kind": "fixed", "shapes": {"main": [1, 64]}},
    "keep_outputs": [0],
    "output_shapes": [[1, 16]],
    "quantize": False,
}


class RecipeTests(unittest.TestCase):
    def test_all_committed_recipes_load(self):
        names = sorted(p.stem for p in (TOOLS / "recipes").glob("*.json"))
        self.assertEqual(
            names,
            ["antelopev2", "bioclip-2", "pp-ocrv6-small", "qa-tiny", "siglip2-base-patch16-224", "siglip2-so400m-patch14-384"],
        )
        for path in (TOOLS / "recipes").glob("*.json"):
            recipe = load_recipe(path)
            self.assertEqual(recipe.model, path.stem)

    def test_precision_policy(self):
        expected = {
            "antelopev2": "fp32",
            "pp-ocrv6-small": "fp32",
            "qa-tiny": "fp32",
            "siglip2-base-patch16-224": "w8a32",
            "siglip2-so400m-patch14-384": "w8a32",
            "bioclip-2": "w8a32",
        }
        for model, precision in expected.items():
            self.assertEqual(load_recipe(TOOLS / "recipes" / f"{model}.json").precision, precision)

    def test_long_side_shapes_cover_hub_resize_rule(self):
        entries = long_side_shapes(960, 32, 64, 1, 3)
        self.assertEqual(len(entries), 57)
        shapes = {e.shape[2:] for e in entries}
        self.assertIn((960, 960), shapes)
        self.assertIn((960, 64), shapes)
        self.assertIn((64, 960), shapes)
        self.assertNotIn((960, 32), shapes)  # IREE 3.12 cannot compile the 32-px buckets
        self.assertTrue(all(max(s) == 960 and min(s) % 32 == 0 and min(s) >= 64 for s in shapes))
        self.assertIn("main_h960_w736", {e.name for e in entries})

    def test_output_shape_placeholders(self):
        self.assertEqual(resolve_output_shape((1, 1, "$2", "$3"), (1, 3, 960, 544)), (1, 1, 960, 544))

    def test_rejects_ambiguous_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bad_fp32_quant = {"schema": 1, "model": "m", "precision": "fp32", "cpu_data_tiling": False,
                              "components": [dict(BASE_COMPONENT, quantize=True, min_quantized_cosine=0.99)]}
            with self.assertRaises(RecipeError):
                load_recipe(_write_recipe(tmp, bad_fp32_quant))
            missing_gate = {"schema": 1, "model": "m", "precision": "w8a32", "cpu_data_tiling": False,
                            "components": [dict(BASE_COMPONENT, quantize=True)]}
            with self.assertRaises(RecipeError):
                load_recipe(_write_recipe(tmp, missing_gate))
            unknown_precision = {"schema": 1, "model": "m", "precision": "w8a8", "cpu_data_tiling": False,
                                 "components": [BASE_COMPONENT]}
            with self.assertRaises(RecipeError):
                load_recipe(_write_recipe(tmp, unknown_precision))


class QuantizeTests(unittest.TestCase):
    def test_quantize_tensor_is_symmetric_per_channel(self):
        w = np.array([[1.0, -0.5], [0.0, 0.0], [-2.0, 4.0]], dtype=np.float32)
        q, s = quantize_tensor(w, axis=0)
        np.testing.assert_array_equal(q, [[127, -64], [0, 0], [-64, 127]])  # rint(-63.5) = -64 (half-even)
        np.testing.assert_allclose(s, [1.0 / 127, 1.0, 4.0 / 127])

    def test_rule_selects_weights_and_axes(self):
        k, n, v = 128, 64, 1024
        rng = np.random.default_rng(0)
        f = lambda *shape: rng.standard_normal(shape).astype(np.float32)
        nodes = [
            helper.make_node("Gather", ["table", "ids"], ["emb"], axis=0),
            helper.make_node("MatMul", ["emb", "w_mm"], ["a"]),
            helper.make_node("Gemm", ["a2d", "w_gemm"], ["b"], transB=1),
            helper.make_node("MatMul", ["emb", "w_shared"], ["c"]),
            helper.make_node("Add", ["c", "w_shared_vec"], ["d"]),
            helper.make_node("MatMul", ["emb", "w_small"], ["e"]),
        ]
        inits = [
            numpy_helper.from_array(f(v, k), "table"),
            numpy_helper.from_array(f(k, n), "w_mm"),
            numpy_helper.from_array(f(n, k), "w_gemm"),
            numpy_helper.from_array(f(k, n), "w_shared"),
            numpy_helper.from_array(f(k, n), "w_shared_vec"),
            numpy_helper.from_array(f(k, 8), "w_small"),
        ]
        # w_shared_vec is reused as a non-weight Add operand -> must stay float.
        nodes.append(helper.make_node("Add", ["e", "w_shared_vec"], ["g"]))
        graph = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 4]),
             helper.make_tensor_value_info("a2d", TensorProto.FLOAT, [1, k])],
            [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in ("b", "d", "g")],
            inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        quantized = quantize_model(model)
        self.assertEqual(quantized, ["table", "w_gemm", "w_mm", "w_shared"])
        self.assertLess(k * 8, MIN_QUANT_ELEMENTS)  # w_small is below the size floor
        axes = {n.input[0][: -len("__w8q")]: n.attribute[0].i for n in model.graph.node if n.op_type == "DequantizeLinear"}
        self.assertEqual(axes, {"table": 0, "w_gemm": 0, "w_mm": 1, "w_shared": 1})


class QaFixtureTests(unittest.TestCase):
    def test_deterministic_value_matches_rust_formula(self):
        self.assertEqual(qa_fixture.deterministic_value(0, 0), np.float32(-0.05))
        # usize::MAX / 2 wraps in the Rust implementation; the Python port must too.
        row = (1 << 64) // 2 - 1
        h = (((row * 31) % (1 << 64)) + 5 * 17) % (1 << 64) % 197
        expected = np.float32((np.float32(h) / np.float32(197.0) - np.float32(0.5)) * np.float32(0.1))
        self.assertEqual(qa_fixture.deterministic_value(row, 5), expected)


class OnnxPrepTests(unittest.TestCase):
    def test_negative_concat_axis_on_shape_vector(self):
        # Paddle exports build Reshape targets with Concat(axis=-1) over 1-D
        # shape pieces (PP-OCR classifiers); symbolic shape inference rejects
        # that unless the axis is normalized.
        from lumen_iree_tools.onnx_prep import prepare_entry

        weight = np.linspace(-1.0, 1.0, 48 * 4, dtype=np.float32).reshape(48, 4)
        nodes = [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Slice", ["shape", "zero", "one"], ["batch"]),
            helper.make_node("Concat", ["batch", "minus_one"], ["target"], axis=-1),
            helper.make_node("Reshape", ["x", "target"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["y"]),
        ]
        graph = helper.make_graph(
            nodes,
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3, 4, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [
                numpy_helper.from_array(np.array([0], dtype=np.int64), "zero"),
                numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
                numpy_helper.from_array(np.array([-1], dtype=np.int64), "minus_one"),
                numpy_helper.from_array(weight, "w"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src.onnx"
            onnx.save_model(model, str(src))
            prepared = prepare_entry(src, tmp / "work", "main", "float32", (1, 3, 4, 4), (0,), ((1, 4),))
            import onnxruntime

            x = np.linspace(-2.0, 2.0, 48, dtype=np.float32).reshape(1, 3, 4, 4)
            session = onnxruntime.InferenceSession(str(prepared), providers=["CPUExecutionProvider"])
            (y,) = session.run(None, {"x": x})
            np.testing.assert_allclose(y, x.reshape(1, 48) @ weight, rtol=1e-5, atol=1e-5)


class ConvertEndToEndTests(unittest.TestCase):
    def test_tiny_convert_and_verify_on_host(self):
        from lumen_iree_tools.pipeline import convert
        from lumen_iree_tools.verify import host_cpu_target

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "net.onnx"
            qa_fixture.build_onnx(src)
            recipe = load_recipe(TOOLS / "recipes" / "qa-tiny.json")
            host = host_cpu_target()
            out = convert(recipe, {"net": src}, tmp / "out", [host], model_info_path=None, keep_work=False)
            iree = out / "iree"
            self.assertTrue((iree / irpa_name("net", "fp32")).is_file())
            self.assertTrue((iree / vmfb_name("net", "fp32", host)).is_file())
            build = json.loads((iree / "BUILD.fp32.json").read_text())
            parity = build["components"]["net"]["verification"]["entries"]["main"]["parity_min_cosine"]
            self.assertGreaterEqual(parity, 0.9999)

            # Byte-reproducible and free of build-machine paths.
            again = convert(recipe, {"net": src}, tmp / "again", [host], model_info_path=None, keep_work=False)
            for name in (irpa_name("net", "fp32"), vmfb_name("net", "fp32", host)):
                first = (iree / name).read_bytes()
                self.assertEqual(first, (again / "iree" / name).read_bytes(), name)
                self.assertNotIn(str(tmp).encode(), first, name)

    def test_targets_are_the_published_set(self):
        self.assertEqual(ALL_TARGETS, ("cpu-x86_64", "cpu-aarch64", "cuda-sm_75", "metal-macos"))
        self.assertEqual({t.driver for t in TARGETS.values()}, {"local-task", "cuda", "metal"})
        self.assertEqual(TARGETS_BY_PRECISION["fp32"], ALL_TARGETS)
        self.assertNotIn("metal-macos", TARGETS_BY_PRECISION["w8a32"])


if __name__ == "__main__":
    unittest.main()
