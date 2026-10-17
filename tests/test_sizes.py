"""total_size_mb and inference_memory_mb must describe the model, not the running sum over its nodes."""
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from omnynet_export.analyzer import peak_activation_bytes, weight_bytes_total


def _chain(shared_weight_uses: int, width: int = 64):
    """x -> [MatMul w] -> relu ... : `shared_weight_uses` MatMuls all read the SAME initializer w (width x width float32)."""
    w = numpy_helper.from_array(np.ones((width, width), dtype=np.float32), "w")
    nodes, prev = [], "x"
    for i in range(shared_weight_uses):
        nodes.append(helper.make_node("MatMul", [prev, "w"], [f"m{i}"]))
        prev = f"m{i}"
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, width])],
        initializer=[w],
    )
    return onnx.shape_inference.infer_shapes(helper.make_model(g))


def test_a_weight_used_by_many_nodes_is_counted_once():
    one, ten = _chain(1), _chain(10)
    assert weight_bytes_total(one) == weight_bytes_total(ten) == 64 * 64 * 4
    # control: the old per-node sum would have grown with the number of uses (10 uses -> 10x), so the two models would differ
    assert sum(64 * 64 * 4 for _ in range(10)) != weight_bytes_total(ten)


def test_weights_are_counted_by_dtype():
    w16 = numpy_helper.from_array(np.ones((8, 8), dtype=np.float16), "a")
    w8 = numpy_helper.from_array(np.ones((8, 8), dtype=np.int8), "b")
    g = helper.make_graph([], "g", [], [], initializer=[w16, w8])
    assert weight_bytes_total(helper.make_model(g)) == 8 * 8 * 2 + 8 * 8 * 1


def test_the_peak_is_what_is_alive_together_not_the_sum_over_all_nodes():
    chain = _chain(20)
    per_tensor = 1 * 64 * 4
    peak = peak_activation_bytes(chain)
    # in a chain only the tensor being read and the one being written are alive (the graph output stays alive to the end): a handful of tensors, never 20
    assert 0 < peak <= 3 * per_tensor, peak
    # control: the old cumulative sum over 20 nodes would be 20 tensors
    assert 20 * per_tensor > 3 * per_tensor


def test_an_unknown_shape_counts_zero_instead_of_failing():
    g = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    assert peak_activation_bytes(helper.make_model(g)) == 0


def test_the_exported_metadata_matches_the_weights(tmp_path):
    from omnynet_export.exporter import ExportConfig, enrich_onnx

    model = _chain(5, width=256)  # 256 KiB of weights
    src = tmp_path / "m.onnx"
    onnx.save(model, src)
    result = enrich_onnx(src, tmp_path / "m.omny", config=ExportConfig(validate=False, check_compatibility=False))
    assert result.success, result.error
    meta = result.metadata.model
    assert meta.total_size_mb == 1, meta  # 256 KiB of weights rounds up to 1 MB, not 5x that
    assert meta.inference_memory_mb >= meta.total_size_mb
    assert 100 <= meta.inference_memory_mb <= 110, meta  # 1.5 x 0.25 MB + the 100 MB runtime base, plus a few KB of activations
