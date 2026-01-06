"""
Analyze ONNX models to find safe cut points for distributed inference.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import onnx
from onnx import ModelProto, GraphProto, NodeProto, TensorProto

from .metadata import CutPoint


@dataclass
class NodeAnalysis:
    """Analysis of a single ONNX node."""

    name: str
    op_type: str
    index: int
    inputs: list[str]
    outputs: list[str]
    output_shapes: list[list[int]]
    output_dtypes: list[str]
    weight_bytes: int = 0
    activation_bytes: int = 0
    cumulative_bytes: int = 0
    is_safe_cut: bool = False
    cut_quality: float = 0.0  # 0-1, higher is better


def dtype_to_bytes(dtype: int) -> int:
    """Convert ONNX dtype to bytes per element."""
    dtype_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.INT8: 1,
        TensorProto.UINT8: 1,
        TensorProto.BOOL: 1,
        TensorProto.BFLOAT16: 2,
    }
    return dtype_sizes.get(dtype, 4)


def dtype_to_str(dtype: int) -> str:
    """Convert ONNX dtype to string."""
    dtype_names = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.DOUBLE: "float64",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.INT8: "int8",
        TensorProto.UINT8: "uint8",
        TensorProto.BOOL: "bool",
        TensorProto.BFLOAT16: "bfloat16",
    }
    return dtype_names.get(dtype, "float32")


def get_tensor_shape(
    graph: GraphProto, tensor_name: str
) -> tuple[Optional[list[int]], int]:
    """
    Get the shape and dtype of a tensor from the graph.

    Returns:
        (shape, dtype) - shape may be None if not found
    """
    # Check value_info
    for vi in graph.value_info:
        if vi.name == tensor_name:
            if vi.type.HasField("tensor_type"):
                tt = vi.type.tensor_type
                shape = []
                if tt.HasField("shape"):
                    for dim in tt.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)  # Dynamic dimension
                return shape, tt.elem_type

    # Check inputs
    for inp in graph.input:
        if inp.name == tensor_name:
            if inp.type.HasField("tensor_type"):
                tt = inp.type.tensor_type
                shape = []
                if tt.HasField("shape"):
                    for dim in tt.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)
                return shape, tt.elem_type

    # Check outputs
    for out in graph.output:
        if out.name == tensor_name:
            if out.type.HasField("tensor_type"):
                tt = out.type.tensor_type
                shape = []
                if tt.HasField("shape"):
                    for dim in tt.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)
                return shape, tt.elem_type

    # Check initializers
    for init in graph.initializer:
        if init.name == tensor_name:
            return list(init.dims), init.data_type

    return None, TensorProto.FLOAT


def calculate_tensor_bytes(shape: list[int], dtype: int) -> int:
    """Calculate the size of a tensor in bytes."""
    if not shape or any(d <= 0 for d in shape):
        return 0

    num_elements = 1
    for d in shape:
        num_elements *= d

    return num_elements * dtype_to_bytes(dtype)


def get_initializer_bytes(graph: GraphProto, name: str) -> int:
    """Get the size of an initializer (weight) in bytes."""
    for init in graph.initializer:
        if init.name == name:
            return calculate_tensor_bytes(list(init.dims), init.data_type)
    return 0


def analyze_graph(model: ModelProto) -> list[NodeAnalysis]:
    """
    Analyze all nodes in the ONNX graph.

    Returns list of NodeAnalysis sorted by execution order.
    """
    graph = model.graph

    # Get all initializer names (weights)
    initializer_names = {init.name for init in graph.initializer}

    # Analyze each node
    analyses = []
    cumulative_bytes = 0

    for idx, node in enumerate(graph.node):
        # Calculate weight bytes (inputs that are initializers)
        weight_bytes = 0
        for inp in node.input:
            if inp in initializer_names:
                weight_bytes += get_initializer_bytes(graph, inp)

        # Get output shapes and calculate activation bytes
        output_shapes = []
        output_dtypes = []
        activation_bytes = 0

        for out in node.output:
            shape, dtype = get_tensor_shape(graph, out)
            if shape:
                output_shapes.append(shape)
                output_dtypes.append(dtype_to_str(dtype))
                activation_bytes += calculate_tensor_bytes(shape, dtype)
            else:
                output_shapes.append([])
                output_dtypes.append("float32")

        cumulative_bytes += weight_bytes + activation_bytes

        # Determine if this is a safe cut point
        is_safe = is_safe_cut_point(node, output_shapes)
        cut_quality = calculate_cut_quality(node, output_shapes, idx, len(graph.node))

        analysis = NodeAnalysis(
            name=node.name or f"node_{idx}",
            op_type=node.op_type,
            index=idx,
            inputs=list(node.input),
            outputs=list(node.output),
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            cumulative_bytes=cumulative_bytes,
            is_safe_cut=is_safe,
            cut_quality=cut_quality,
        )
        analyses.append(analysis)

    return analyses


def is_safe_cut_point(node: NodeProto, output_shapes: list[list[int]]) -> bool:
    """
    Determine if cutting after this node is safe.

    A cut point is safe if:
    - It has exactly one output (simplifies tensor transfer)
    - The output shape is fully defined (no dynamic dims except batch)
    - It's not a control flow node
    """
    # Must have outputs
    if not node.output or not output_shapes:
        return False

    # Prefer single output nodes
    if len(node.output) != 1:
        return False

    # Check output shape is defined
    shape = output_shapes[0] if output_shapes else []
    if not shape:
        return False

    # Allow dynamic batch dimension (first dim), but others must be concrete
    for i, dim in enumerate(shape):
        if i == 0:
            continue  # Batch dimension can be dynamic
        if dim <= 0:
            return False

    # Avoid cutting inside control flow
    if node.op_type in ("If", "Loop", "Scan"):
        return False

    return True


def calculate_cut_quality(
    node: NodeProto,
    output_shapes: list[list[int]],
    index: int,
    total_nodes: int,
) -> float:
    """
    Calculate a quality score for this cut point (0-1, higher is better).

    Factors:
    - Smaller output tensors are better (less data to transfer)
    - Layer boundaries are better
    - Even distribution through the model is better
    """
    if not output_shapes or not output_shapes[0]:
        return 0.0

    score = 0.5  # Base score

    # Prefer layer boundary ops
    layer_ops = {
        "Conv", "ConvTranspose", "Gemm", "MatMul",
        "BatchNormalization", "LayerNormalization",
        "Add", "Relu", "Sigmoid", "Softmax",
        "GlobalAveragePool", "AveragePool", "MaxPool",
    }
    if node.op_type in layer_ops:
        score += 0.2

    # Prefer smaller output tensors
    shape = output_shapes[0]
    if all(d > 0 for d in shape):
        tensor_size = 1
        for d in shape:
            tensor_size *= d
        # Normalize: smaller is better
        if tensor_size < 1_000_000:
            score += 0.2
        elif tensor_size < 10_000_000:
            score += 0.1

    # Prefer positions that divide model evenly
    position_ratio = (index + 1) / total_nodes
    # Good positions: 0.25, 0.33, 0.5, 0.67, 0.75
    good_positions = [0.25, 0.33, 0.5, 0.67, 0.75]
    min_dist = min(abs(position_ratio - p) for p in good_positions)
    if min_dist < 0.05:
        score += 0.1

    return min(score, 1.0)


def find_cut_points(
    model: ModelProto,
    max_shard_size_mb: int = 1200,
    min_shards: int = 2,
    max_shards: int = 8,
) -> list[CutPoint]:
    """
    Find optimal cut points for sharding the model.

    Args:
        model: ONNX model
        max_shard_size_mb: Maximum memory per shard in MB
        min_shards: Minimum number of shards to create
        max_shards: Maximum number of shards to create

    Returns:
        List of CutPoint objects
    """
    analyses = analyze_graph(model)

    if not analyses:
        return []

    total_bytes = analyses[-1].cumulative_bytes
    total_mb = total_bytes / (1024 * 1024)

    # Calculate target number of shards
    target_shards = max(min_shards, int(np.ceil(total_mb / max_shard_size_mb)))
    target_shards = min(target_shards, max_shards)

    # Find safe cut points
    safe_cuts = [a for a in analyses if a.is_safe_cut]

    if not safe_cuts:
        # Fallback: use any nodes with defined shapes
        safe_cuts = [a for a in analyses if a.output_shapes and a.output_shapes[0]]

    if not safe_cuts:
        return []

    # Select cut points to divide model into target_shards pieces
    # We need (target_shards - 1) cut points
    num_cuts = target_shards - 1

    # Score-based selection
    safe_cuts_sorted = sorted(safe_cuts, key=lambda a: a.cut_quality, reverse=True)

    # Also consider even distribution
    target_bytes_per_shard = total_bytes / target_shards
    selected_cuts = []

    for cut_num in range(num_cuts):
        target_cumulative = target_bytes_per_shard * (cut_num + 1)

        # Find the safe cut closest to target
        best_cut = None
        best_distance = float("inf")

        for cut in safe_cuts:
            # Skip if already selected
            if cut in selected_cuts:
                continue

            # Skip if too close to already selected cuts
            too_close = False
            for selected in selected_cuts:
                if abs(cut.index - selected.index) < len(analyses) / (max_shards * 2):
                    too_close = True
                    break

            if too_close:
                continue

            distance = abs(cut.cumulative_bytes - target_cumulative)
            # Weight by cut quality
            distance = distance / (cut.cut_quality + 0.1)

            if distance < best_distance:
                best_distance = distance
                best_cut = cut

        if best_cut:
            selected_cuts.append(best_cut)

    # Sort by index and create CutPoint objects
    selected_cuts.sort(key=lambda a: a.index)

    cut_points = []
    prev_bytes = 0

    for i, cut in enumerate(selected_cuts):
        shard_bytes = cut.cumulative_bytes - prev_bytes

        cp = CutPoint(
            id=f"cut_{i + 1}",
            after_node=cut.name,
            tensor_name=cut.outputs[0] if cut.outputs else "",
            shape=cut.output_shapes[0] if cut.output_shapes else [],
            dtype=cut.output_dtypes[0] if cut.output_dtypes else "float32",
            cumulative_memory_mb=int(cut.cumulative_bytes / (1024 * 1024)),
            shard_memory_mb=int(shard_bytes / (1024 * 1024)),
        )
        cut_points.append(cp)
        prev_bytes = cut.cumulative_bytes

    return cut_points


def calculate_shard_configs(
    cut_points: list[CutPoint],
    total_memory_mb: int,
) -> list[tuple[int, list[int], list[str]]]:
    """
    Calculate valid shard configurations from cut points.

    Returns list of (num_shards, memory_per_shard, cut_point_ids)
    """
    if not cut_points:
        return [(1, [total_memory_mb], [])]

    configs = []

    # Configuration with all cut points
    num_shards = len(cut_points) + 1
    memories = []
    prev_mem = 0
    for cp in cut_points:
        memories.append(cp.cumulative_memory_mb - prev_mem)
        prev_mem = cp.cumulative_memory_mb
    memories.append(total_memory_mb - prev_mem)

    configs.append((
        num_shards,
        memories,
        [cp.id for cp in cut_points],
    ))

    # Also generate configs with fewer shards (subsets of cut points)
    if len(cut_points) >= 2:
        # Half the cut points
        half_cuts = cut_points[::2]
        if len(half_cuts) < len(cut_points):
            num_shards = len(half_cuts) + 1
            memories = []
            prev_mem = 0
            for cp in half_cuts:
                memories.append(cp.cumulative_memory_mb - prev_mem)
                prev_mem = cp.cumulative_memory_mb
            memories.append(total_memory_mb - prev_mem)
            configs.append((
                num_shards,
                memories,
                [cp.id for cp in half_cuts],
            ))

    return configs
