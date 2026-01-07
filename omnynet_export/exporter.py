"""
Main export functionality - convert PyTorch models or enrich ONNX to .omny format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import tempfile
import os

import numpy as np
import onnx
from onnx import ModelProto

from .metadata import (
    OmnyMetadata,
    ModelInfo,
    TensorSpec,
    ShardingConstraints,
    ShardConfig,
    ExportInfo,
)
from .analyzer import find_cut_points, analyze_graph, calculate_shard_configs
from .format import embed_metadata, save_omny


@dataclass
class ExportConfig:
    """Configuration for export."""

    min_vram_mb: int = 1500
    max_shard_size_mb: int = 1200
    min_shards: int = 2
    max_shards: int = 8
    opset_version: int = 17
    model_name: Optional[str] = None
    model_architecture: str = "unknown"
    validate: bool = False  # Run actual inference to validate cut points
    validation_tolerance: float = 1e-4  # Max allowed output difference


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    output_path: Optional[Path]
    metadata: Optional[OmnyMetadata]
    cut_points: list
    allowed_shards: list[int]
    error: Optional[str] = None
    validated: bool = False  # Whether cut points were validated
    validation_errors: list[str] = None  # Any validation errors

    def summary(self) -> str:
        if not self.success:
            return f"Export failed: {self.error}"

        lines = [
            f"Export successful: {self.output_path}",
            f"Cut points: {len(self.cut_points)}",
            f"Allowed shards: {self.allowed_shards}",
        ]

        if self.metadata:
            lines.append(f"Model: {self.metadata.model.name}")
            lines.append(f"Total size: {self.metadata.model.total_size_mb} MB")
            lines.append(f"Inference memory: {self.metadata.model.inference_memory_mb} MB")

        return "\n".join(lines)


def export_model(
    model: Union[str, Path, Any],  # Path to .pt file or PyTorch model
    output: Union[str, Path],
    sample_input: dict[str, Any],
    config: Optional[Union[dict, ExportConfig]] = None,
) -> ExportResult:
    """
    Export a PyTorch model to .omny format.

    Args:
        model: Path to PyTorch checkpoint (.pt/.pth) or model instance
        output: Output path for .omny file
        sample_input: Dict of input_name -> tensor/array for tracing
        config: Export configuration

    Returns:
        ExportResult with export details
    """
    try:
        import torch
    except ImportError:
        return ExportResult(
            success=False,
            output_path=None,
            metadata=None,
            cut_points=[],
            allowed_shards=[],
            error="PyTorch not installed. Run: pip install torch",
        )

    # Parse config
    if config is None:
        cfg = ExportConfig()
    elif isinstance(config, dict):
        cfg = ExportConfig(**config)
    else:
        cfg = config

    output_path = Path(output)

    try:
        # Load model if path provided
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            if not model_path.exists():
                return ExportResult(
                    success=False,
                    output_path=None,
                    metadata=None,
                    cut_points=[],
                    allowed_shards=[],
                    error=f"Model file not found: {model_path}",
                )

            # Try to load as checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    pytorch_model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    # Need model architecture - can't load just state_dict
                    return ExportResult(
                        success=False,
                        output_path=None,
                        metadata=None,
                        cut_points=[],
                        allowed_shards=[],
                        error="Checkpoint contains only state_dict. Please provide full model or use enrich_onnx with pre-converted ONNX.",
                    )
                else:
                    pytorch_model = checkpoint
            else:
                pytorch_model = checkpoint

            if cfg.model_name is None:
                cfg.model_name = model_path.stem
        else:
            pytorch_model = model
            if cfg.model_name is None:
                cfg.model_name = type(model).__name__

        # Prepare inputs for tracing
        input_tensors = {}
        input_specs = []

        for name, value in sample_input.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.tensor(value)

            input_tensors[name] = tensor
            input_specs.append(TensorSpec(
                name=name,
                shape=list(tensor.shape),
                dtype=_torch_dtype_to_str(tensor.dtype),
            ))

        # Set model to eval mode
        pytorch_model.eval()

        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_onnx_path = f.name

        try:
            # Prepare input for export
            if len(input_tensors) == 1:
                export_input = list(input_tensors.values())[0]
                input_names = list(input_tensors.keys())
            else:
                export_input = tuple(input_tensors.values())
                input_names = list(input_tensors.keys())

            # Run model once to get output names
            with torch.no_grad():
                if len(input_tensors) == 1:
                    output = pytorch_model(export_input)
                else:
                    output = pytorch_model(*export_input)

            # Determine output names
            if isinstance(output, torch.Tensor):
                output_names = ["output"]
                outputs = [output]
            elif isinstance(output, (tuple, list)):
                output_names = [f"output_{i}" for i in range(len(output))]
                outputs = list(output)
            elif isinstance(output, dict):
                output_names = list(output.keys())
                outputs = list(output.values())
            else:
                output_names = ["output"]
                outputs = [output]

            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                export_input,
                temp_onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=cfg.opset_version,
                do_constant_folding=True,
                dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
            )

            # Load the ONNX model
            onnx_model = onnx.load(temp_onnx_path)

            # Create output specs
            output_specs = []
            for name, tensor in zip(output_names, outputs):
                if isinstance(tensor, torch.Tensor):
                    output_specs.append(TensorSpec(
                        name=name,
                        shape=list(tensor.shape),
                        dtype=_torch_dtype_to_str(tensor.dtype),
                    ))

            # Continue with common processing
            return _process_onnx_model(
                onnx_model,
                output_path,
                cfg,
                input_specs,
                output_specs,
                source_framework="pytorch",
                source_version=torch.__version__,
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_onnx_path):
                os.unlink(temp_onnx_path)

    except Exception as e:
        return ExportResult(
            success=False,
            output_path=None,
            metadata=None,
            cut_points=[],
            allowed_shards=[],
            error=str(e),
        )


def enrich_onnx(
    onnx_path: Union[str, Path],
    output: Union[str, Path],
    sample_input: Optional[dict[str, Any]] = None,
    config: Optional[Union[dict, ExportConfig]] = None,
) -> ExportResult:
    """
    Enrich an existing ONNX model with OmnyNet metadata.

    Args:
        onnx_path: Path to existing ONNX file
        output: Output path for .omny file
        sample_input: Optional sample input for shape inference
        config: Export configuration

    Returns:
        ExportResult with export details
    """
    # Parse config
    if config is None:
        cfg = ExportConfig()
    elif isinstance(config, dict):
        cfg = ExportConfig(**config)
    else:
        cfg = config

    onnx_path = Path(onnx_path)
    output_path = Path(output)

    if not onnx_path.exists():
        return ExportResult(
            success=False,
            output_path=None,
            metadata=None,
            cut_points=[],
            allowed_shards=[],
            error=f"ONNX file not found: {onnx_path}",
        )

    try:
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))

        if cfg.model_name is None:
            cfg.model_name = onnx_path.stem

        # Extract input/output specs from ONNX
        input_specs = []
        for inp in onnx_model.graph.input:
            # Skip initializers
            if any(init.name == inp.name for init in onnx_model.graph.initializer):
                continue

            shape = []
            dtype = "float32"
            if inp.type.HasField("tensor_type"):
                tt = inp.type.tensor_type
                dtype = _onnx_dtype_to_str(tt.elem_type)
                if tt.HasField("shape"):
                    for dim in tt.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)

            input_specs.append(TensorSpec(name=inp.name, shape=shape, dtype=dtype))

        output_specs = []
        for out in onnx_model.graph.output:
            shape = []
            dtype = "float32"
            if out.type.HasField("tensor_type"):
                tt = out.type.tensor_type
                dtype = _onnx_dtype_to_str(tt.elem_type)
                if tt.HasField("shape"):
                    for dim in tt.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)

            output_specs.append(TensorSpec(name=out.name, shape=shape, dtype=dtype))

        # Get opset version
        opset = 17
        for oi in onnx_model.opset_import:
            if oi.domain == "" or oi.domain == "ai.onnx":
                opset = oi.version
                break

        return _process_onnx_model(
            onnx_model,
            output_path,
            cfg,
            input_specs,
            output_specs,
            source_framework="onnx",
            source_version=f"opset-{opset}",
            sample_inputs=sample_input,
        )

    except Exception as e:
        return ExportResult(
            success=False,
            output_path=None,
            metadata=None,
            cut_points=[],
            allowed_shards=[],
            error=str(e),
        )


def _process_onnx_model(
    onnx_model: ModelProto,
    output_path: Path,
    cfg: ExportConfig,
    input_specs: list[TensorSpec],
    output_specs: list[TensorSpec],
    source_framework: str,
    source_version: str,
    sample_inputs: Optional[dict[str, np.ndarray]] = None,
) -> ExportResult:
    """Common processing for ONNX models."""

    # Run shape inference
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception:
        pass  # Shape inference may fail, continue anyway

    # Analyze graph
    analyses = analyze_graph(onnx_model)

    # Calculate total memory
    total_bytes = analyses[-1].cumulative_bytes if analyses else 0
    total_mb = int(total_bytes / (1024 * 1024))

    # Estimate inference memory (weights + activations + overhead)
    inference_memory_mb = int(total_mb * 1.5)  # 50% overhead estimate

    # Calculate total parameters
    total_params = 0
    for init in onnx_model.graph.initializer:
        num_elements = 1
        for dim in init.dims:
            num_elements *= dim
        total_params += num_elements

    # Find cut points
    cut_points = find_cut_points(
        onnx_model,
        max_shard_size_mb=cfg.max_shard_size_mb,
        min_shards=cfg.min_shards,
        max_shards=cfg.max_shards,
    )

    # Validate cut points if requested
    validated = False
    validation_errors = []

    if cfg.validate and sample_inputs and cut_points:
        from .validator import find_valid_cut_points

        print("Validating cut points (this may take a moment)...")
        valid_cuts = find_valid_cut_points(
            onnx_model,
            cut_points,
            sample_inputs,
            tolerance=cfg.validation_tolerance,
            verbose=True,
        )

        # Track which cuts failed
        valid_ids = {cp.id for cp in valid_cuts}
        for cp in cut_points:
            if cp.id not in valid_ids:
                validation_errors.append(f"Cut point {cp.id} ({cp.after_node}) failed validation")

        # Use only valid cut points
        cut_points = valid_cuts
        validated = True

        if not cut_points:
            print("Warning: No valid cut points found. Model cannot be sharded.")

    # Calculate shard configurations
    shard_configs_raw = calculate_shard_configs(cut_points, inference_memory_mb)
    shard_configs = [
        ShardConfig(
            num_shards=num,
            memory_per_shard_mb=mems,
            cut_point_ids=ids,
        )
        for num, mems, ids in shard_configs_raw
    ]

    allowed_shards = sorted(set(sc.num_shards for sc in shard_configs))

    # Get opset version
    opset = cfg.opset_version
    for oi in onnx_model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            opset = oi.version
            break

    # Build metadata
    metadata = OmnyMetadata(
        version="1.0",
        model=ModelInfo(
            name=cfg.model_name or "unknown",
            architecture=cfg.model_architecture,
            total_params=total_params,
            total_size_mb=total_mb,
            inference_memory_mb=inference_memory_mb,
        ),
        inputs=input_specs,
        outputs=output_specs,
        cut_points=cut_points,
        sharding=ShardingConstraints(
            min_vram_mb=cfg.min_vram_mb,
            max_shard_size_mb=cfg.max_shard_size_mb,
            min_shards=cfg.min_shards,
            max_shards=cfg.max_shards,
            allowed_shards=allowed_shards,
            configurations=shard_configs,
        ),
        export_info=ExportInfo(
            exporter_version="0.1.0",
            source_framework=source_framework,
            source_version=source_version,
            onnx_opset=opset,
            validated=validated,
        ),
    )

    # Embed metadata and save
    onnx_model = embed_metadata(onnx_model, metadata)
    save_omny(onnx_model, output_path)

    return ExportResult(
        success=True,
        output_path=output_path,
        metadata=metadata,
        cut_points=cut_points,
        allowed_shards=allowed_shards,
        validated=validated,
        validation_errors=validation_errors if validation_errors else None,
    )


def _torch_dtype_to_str(dtype) -> str:
    """Convert PyTorch dtype to string."""
    import torch

    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
        torch.bfloat16: "bfloat16",
    }
    return dtype_map.get(dtype, "float32")


def _onnx_dtype_to_str(dtype: int) -> str:
    """Convert ONNX dtype to string."""
    from onnx import TensorProto

    dtype_map = {
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
    return dtype_map.get(dtype, "float32")
