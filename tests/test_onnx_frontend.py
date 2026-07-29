"""
tests/test_onnx_frontend.py – Tests for the ONNX MatMul frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

# Try importing the ONNX frontend
try:
    import onnx_frontend
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _ONNX_AVAILABLE,
    reason="onnx_frontend not importable (onnx package may not be installed)",
)


class TestOnnxFrontend:
    """Verify that onnx_frontend.parse_onnx_to_hls behaves correctly."""

    def test_module_importable(self):
        import onnx_frontend
        assert hasattr(onnx_frontend, "parse_onnx_to_hls"), (
            "onnx_frontend must expose parse_onnx_to_hls()"
        )

    def test_only_matmul_nodes_extracted(self, tmp_path):
        """parse_onnx_to_hls must only extract MatMul nodes."""
        try:
            import onnx
            import numpy as np
        except ImportError:
            pytest.skip("onnx/numpy not available")

        # Build minimal single-MatMul ONNX graph
        A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [16, 16])
        B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [16, 16])
        C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [16, 16])
        relu_out = onnx.helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [16, 16])

        matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"])
        relu_node = onnx.helper.make_node("Relu", ["C"], ["D"])  # should be ignored

        graph = onnx.helper.make_graph([matmul_node, relu_node], "test", [A, B], [relu_out])
        model = onnx.helper.make_model(graph)
        model_path = str(tmp_path / "test_model.onnx")
        onnx.save(model, model_path)

        nodes = onnx_frontend.parse_onnx_to_hls(model_path)
        assert len(nodes) == 1, f"Expected 1 MatMul node, got {len(nodes)}"

    def test_node_limit_respected(self, tmp_path):
        """node_limit must cap the number of extracted MatMul nodes."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not available")

        inputs = [onnx.helper.make_tensor_value_info(f"x{i}", onnx.TensorProto.FLOAT, [8, 8])
                  for i in range(5)]
        outputs = [onnx.helper.make_tensor_value_info(f"y{i}", onnx.TensorProto.FLOAT, [8, 8])
                   for i in range(4)]
        weights = [onnx.helper.make_tensor_value_info(f"w{i}", onnx.TensorProto.FLOAT, [8, 8])
                   for i in range(4)]

        nodes = [
            onnx.helper.make_node("MatMul", [f"x{i}", f"w{i}"], [f"y{i}"])
            for i in range(4)
        ]
        # Wire outputs as next inputs (chain)
        graph = onnx.helper.make_graph(nodes, "test4", inputs + weights, outputs)
        model = onnx.helper.make_model(graph)
        model_path = str(tmp_path / "test_model4.onnx")
        onnx.save(model, model_path)

        result = onnx_frontend.parse_onnx_to_hls(model_path, node_limit=2)
        assert len(result) == 2, f"node_limit=2 should give 2 nodes, got {len(result)}"
