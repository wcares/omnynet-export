"""
Runtime validation for cut points - actually split and test the model.
"""

import numpy as np
import onnx
from onnx import ModelProto, helper, TensorProto
from typing import Optional
from dataclasses import dataclass
import onnxruntime as ort

from .metadata import CutPoint


@dataclass
class ValidationResult:
    """Result of validating a cut point."""

    cut_point_id: str
    is_valid: bool
    error: Optional[str] = None
    max_diff: float = 0.0
    mean_diff: float = 0.0


@dataclass
class ShardValidationResult:
    """Result of validating a sharding configuration."""

    num_shards: int
    cut_point_ids: list[str]
    is_valid: bool
    shard_results: list[ValidationResult]
    error: Optional[str] = None


def extract_subgraph(
    model: ModelProto,
    start_node_idx: int,
    end_node_idx: int,
    input_names: list[str],
    output_names: list[str],
) -> ModelProto:
    """
    Extract a subgraph from the model between node indices.

    Args:
        model: Full ONNX model
        start_node_idx: Index of first node to include (or -1 for start)
        end_node_idx: Index of last node to include (or -1 for end)
        input_names: Names of inputs to this subgraph
        output_names: Names of outputs from this subgraph

    Returns:
        New ONNX model containing just the subgraph
    """
    graph = model.graph
    nodes = list(graph.node)

    if end_node_idx == -1:
        end_node_idx = len(nodes) - 1

    # Get nodes in range
    subgraph_nodes = nodes[start_node_idx:end_node_idx + 1]

    # Collect all initializers used by these nodes
    node_inputs = set()
    for node in subgraph_nodes:
        node_inputs.update(node.input)

    initializers = []
    for init in graph.initializer:
        if init.name in node_inputs:
            initializers.append(init)

    # Build input specs
    inputs = []
    for name in input_names:
        # Find the tensor info
        found = False
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.name == name:
                inputs.append(vi)
                found = True
                break

        if not found:
            # Create a placeholder input
            inputs.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT, None
            ))

    # Build output specs
    outputs = []
    for name in output_names:
        found = False
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.name == name:
                outputs.append(vi)
                found = True
                break

        if not found:
            outputs.append(helper.make_tensor_value_info(
                name, TensorProto.FLOAT, None
            ))

    # Create new graph
    new_graph = helper.make_graph(
        subgraph_nodes,
        f"subgraph_{start_node_idx}_{end_node_idx}",
        inputs,
        outputs,
        initializers,
    )

    # Create new model with same opset as original
    opset_imports = list(model.opset_import)
    new_model = helper.make_model(new_graph, opset_imports=opset_imports)

    # Set IR version - cap at 8 for max onnxruntime compatibility
    new_model.ir_version = min(model.ir_version, 8)

    return new_model


def split_at_cut_point(
    model: ModelProto,
    cut_point: CutPoint,
) -> tuple[ModelProto, ModelProto]:
    """
    Split model into two parts at the cut point.

    Returns:
        (shard1, shard2) - Two ONNX models
    """
    graph = model.graph
    nodes = list(graph.node)

    # Find the cut node index
    cut_node_idx = -1
    for idx, node in enumerate(nodes):
        if node.name == cut_point.after_node:
            cut_node_idx = idx
            break

    if cut_node_idx == -1:
        raise ValueError(f"Cut point node not found: {cut_point.after_node}")

    # Get original inputs (excluding initializers)
    init_names = {init.name for init in graph.initializer}
    original_inputs = [inp.name for inp in graph.input if inp.name not in init_names]
    original_outputs = [out.name for out in graph.output]

    # Build set of tensors produced by shard1 (nodes 0 to cut_node_idx)
    shard1_nodes = nodes[0:cut_node_idx + 1]
    shard1_outputs = set()
    for node in shard1_nodes:
        shard1_outputs.update(node.output)

    # Build set of tensors consumed by shard2 (nodes cut_node_idx+1 to end)
    shard2_nodes = nodes[cut_node_idx + 1:]
    shard2_inputs_needed = set()
    shard2_outputs_produced = set()
    for node in shard2_nodes:
        shard2_inputs_needed.update(node.input)
        shard2_outputs_produced.update(node.output)

    # Find all external inputs shard2 needs:
    # - Not produced within shard2 itself
    # - Not an initializer
    # - Either produced by shard1 OR is an original model input
    shard2_external_inputs = []
    for inp in shard2_inputs_needed:
        if inp in shard2_outputs_produced:
            continue  # Produced within shard2
        if inp in init_names:
            continue  # Is an initializer (weights)
        if inp in shard1_outputs or inp in original_inputs:
            shard2_external_inputs.append(inp)

    # Deduplicate while preserving order
    seen = set()
    shard2_external_inputs_deduped = []
    for inp in shard2_external_inputs:
        if inp not in seen:
            seen.add(inp)
            shard2_external_inputs_deduped.append(inp)
    shard2_external_inputs = shard2_external_inputs_deduped

    # Shard 1: from start to cut node (inclusive)
    # Outputs: the cut tensor + any other tensors shard2 needs from shard1
    shard1_output_names = list(shard2_external_inputs)  # All tensors shard2 needs
    if cut_point.tensor_name not in shard1_output_names:
        shard1_output_names.insert(0, cut_point.tensor_name)

    shard1 = extract_subgraph(
        model,
        start_node_idx=0,
        end_node_idx=cut_node_idx,
        input_names=original_inputs,
        output_names=shard1_output_names,
    )

    # Shard 2: from cut node + 1 to end
    # Inputs: all external tensors needed
    shard2 = extract_subgraph(
        model,
        start_node_idx=cut_node_idx + 1,
        end_node_idx=-1,
        input_names=shard2_external_inputs,
        output_names=original_outputs,
    )

    return shard1, shard2


def run_inference(
    model: ModelProto,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run inference on an ONNX model."""
    # Serialize to bytes for onnxruntime
    model_bytes = model.SerializeToString()

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Suppress warnings

    session = ort.InferenceSession(
        model_bytes,
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Get input names from session
    input_names = [inp.name for inp in session.get_inputs()]

    # Filter inputs to only those the model expects
    feed = {}
    for name in input_names:
        if name in inputs:
            feed[name] = inputs[name]
        else:
            raise ValueError(f"Missing input: {name}")

    # Run inference
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, feed)

    return dict(zip(output_names, outputs))


def validate_cut_point(
    model: ModelProto,
    cut_point: CutPoint,
    sample_inputs: dict[str, np.ndarray],
    tolerance: float = 1e-4,
) -> ValidationResult:
    """
    Validate a single cut point by actually splitting and running inference.

    Args:
        model: Full ONNX model
        cut_point: Cut point to validate
        sample_inputs: Sample inputs for inference
        tolerance: Maximum allowed difference

    Returns:
        ValidationResult
    """
    try:
        # Run full model to get expected output
        expected_outputs = run_inference(model, sample_inputs)

        # Split the model
        shard1, shard2 = split_at_cut_point(model, cut_point)

        # Run shard 1
        shard1_outputs = run_inference(shard1, sample_inputs)

        # Run shard 2 with ALL of shard 1's outputs as inputs
        # (handles skip connections and other cross-shard dependencies)
        shard2_outputs = run_inference(shard2, shard1_outputs)

        # Compare outputs
        max_diff = 0.0
        mean_diff = 0.0
        total_elements = 0

        for name, expected in expected_outputs.items():
            if name not in shard2_outputs:
                return ValidationResult(
                    cut_point_id=cut_point.id,
                    is_valid=False,
                    error=f"Missing output: {name}",
                )

            actual = shard2_outputs[name]
            diff = np.abs(expected - actual)
            max_diff = max(max_diff, float(np.max(diff)))
            mean_diff += float(np.sum(diff))
            total_elements += diff.size

        mean_diff = mean_diff / total_elements if total_elements > 0 else 0.0

        is_valid = max_diff <= tolerance

        return ValidationResult(
            cut_point_id=cut_point.id,
            is_valid=is_valid,
            max_diff=max_diff,
            mean_diff=mean_diff,
            error=None if is_valid else f"Output mismatch: max_diff={max_diff:.6f}",
        )

    except Exception as e:
        return ValidationResult(
            cut_point_id=cut_point.id,
            is_valid=False,
            error=str(e),
        )


def validate_sharding(
    model: ModelProto,
    cut_points: list[CutPoint],
    sample_inputs: dict[str, np.ndarray],
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> ShardValidationResult:
    """
    Validate a full sharding configuration with multiple cut points.

    Args:
        model: Full ONNX model
        cut_points: List of cut points defining the shards
        sample_inputs: Sample inputs for inference
        tolerance: Maximum allowed difference
        verbose: Print progress

    Returns:
        ShardValidationResult
    """
    if not cut_points:
        # No sharding, just validate model runs
        try:
            run_inference(model, sample_inputs)
            return ShardValidationResult(
                num_shards=1,
                cut_point_ids=[],
                is_valid=True,
                shard_results=[],
            )
        except Exception as e:
            return ShardValidationResult(
                num_shards=1,
                cut_point_ids=[],
                is_valid=False,
                shard_results=[],
                error=str(e),
            )

    results = []
    all_valid = True

    for i, cp in enumerate(cut_points):
        if verbose:
            print(f"  Validating cut point {i+1}/{len(cut_points)}: {cp.id}...")

        result = validate_cut_point(model, cp, sample_inputs, tolerance)
        results.append(result)

        if not result.is_valid:
            all_valid = False
            if verbose:
                print(f"    FAILED: {result.error}")
        elif verbose:
            print(f"    OK (max_diff={result.max_diff:.6f})")

    return ShardValidationResult(
        num_shards=len(cut_points) + 1,
        cut_point_ids=[cp.id for cp in cut_points],
        is_valid=all_valid,
        shard_results=results,
    )


def find_valid_cut_points(
    model: ModelProto,
    candidate_cut_points: list[CutPoint],
    sample_inputs: dict[str, np.ndarray],
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> list[CutPoint]:
    """
    Find which cut points actually work by testing each one.

    Args:
        model: Full ONNX model
        candidate_cut_points: Candidate cut points to test
        sample_inputs: Sample inputs for inference
        tolerance: Maximum allowed difference
        verbose: Print progress

    Returns:
        List of validated cut points that actually work
    """
    valid_cuts = []

    for i, cp in enumerate(candidate_cut_points):
        if verbose:
            print(f"Testing cut point {i+1}/{len(candidate_cut_points)}: {cp.id} ({cp.after_node})...")

        result = validate_cut_point(model, cp, sample_inputs, tolerance)

        if result.is_valid:
            valid_cuts.append(cp)
            if verbose:
                print(f"  VALID (max_diff={result.max_diff:.6f})")
        else:
            if verbose:
                print(f"  INVALID: {result.error}")

    return valid_cuts
