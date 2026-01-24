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
from .format import (
    embed_metadata, save_omny,
    save_omny_v2, OmnyV2Manifest, ProcessorConfig,
    PreprocessConfig, PostprocessConfig,
)
from .compatibility import (
    CompatibilityReport,
    check_compatibility,
    fix_dynamic_shapes,
    format_compatibility_report,
)


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
    inference_memory_mb: Optional[int] = None  # Manual override for VRAM requirement (auto-calculated if None)

    # wonnx compatibility options
    check_compatibility: bool = True  # Check wonnx/ort compatibility
    fix_shapes: bool = False  # Fix dynamic shapes for wonnx
    input_shapes: Optional[dict[str, list[int]]] = None  # Concrete shapes for fixing


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
    compatibility_report: Optional["CompatibilityReport"] = None  # Backend compatibility

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

            # Convert input tensors to numpy for validation
            sample_inputs_np = {}
            for name, tensor in input_tensors.items():
                sample_inputs_np[name] = tensor.cpu().numpy()

            # Continue with common processing
            return _process_onnx_model(
                onnx_model,
                output_path,
                cfg,
                input_specs,
                output_specs,
                source_framework="pytorch",
                source_version=torch.__version__,
                sample_inputs=sample_inputs_np,
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

    compatibility_report = None

    # Fix dynamic shapes if requested
    if cfg.fix_shapes and cfg.input_shapes:
        print("Fixing dynamic shapes for wonnx compatibility...")
        onnx_model, changes = fix_dynamic_shapes(onnx_model, cfg.input_shapes)
        for change in changes:
            print(f"  {change}")
        print()

    # Run shape inference
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception:
        pass  # Shape inference may fail, continue anyway

    # Check backend compatibility
    if cfg.check_compatibility:
        compatibility_report = check_compatibility(onnx_model)

        # If shapes were fixed, update report
        if cfg.fix_shapes and cfg.input_shapes:
            compatibility_report.shapes_fixed = True

        # Print compatibility report
        print(format_compatibility_report(compatibility_report))

    # Analyze graph
    analyses = analyze_graph(onnx_model)

    # Calculate total memory
    total_bytes = analyses[-1].cumulative_bytes if analyses else 0
    total_mb = int(total_bytes / (1024 * 1024))

    # Inference memory: use manual override if provided, otherwise estimate
    if cfg.inference_memory_mb is not None:
        inference_memory_mb = cfg.inference_memory_mb
    else:
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
        compatibility_report=compatibility_report,
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


# ============================================================================
# v2 Format Export (with Native Processor Support)
# ============================================================================

@dataclass
class ExportV2Config:
    """Configuration for v2 format export with native processor support."""

    # Model info
    model_name: Optional[str] = None
    model_architecture: str = "unknown"
    task: str = ""  # "ocr", "detection", "classification", "embedding"

    # Input/output types
    input_type: str = "image"  # "image", "text", "tensor"
    input_format: str = "base64"  # "base64", "bytes", "path"
    output_type: str = "raw"  # "detection", "ocr", "embedding", "raw"
    output_format: str = "json"  # "json", "bytes"

    # Processor (native executables, not bundled in .omny file)
    processor_type: str = "none"  # "native", "none"
    processor_id: Optional[str] = None  # For native: "clip", "ocr", "sam2", "dino"

    # Preprocessing (metadata for documentation/reference)
    preprocess_resize: Optional[tuple[int, int]] = None  # [width, height]
    preprocess_resize_mode: str = "exact"  # "exact", "preserve_aspect", "pad", "shortest_center_crop"
    preprocess_normalize_mean: Optional[tuple[float, float, float]] = None
    preprocess_normalize_std: Optional[tuple[float, float, float]] = None
    preprocess_scale: float = 255.0
    preprocess_channel_order: str = "RGB"

    # Postprocessing
    postprocess_type: str = "raw"  # "detection", "classification", "ocr"
    postprocess_confidence_threshold: float = 0.5
    postprocess_nms_threshold: Optional[float] = None
    postprocess_labels: Optional[list[str]] = None

    # Sharding (same as v1)
    min_vram_mb: int = 1500
    max_shard_size_mb: int = 1200
    min_shards: int = 2
    max_shards: int = 8
    inference_memory_mb: Optional[int] = None


@dataclass
class ExportV2Result:
    """Result of v2 export operation."""
    success: bool
    output_path: Optional[Path]
    manifest: Optional[OmnyV2Manifest]
    error: Optional[str] = None
    
    def summary(self) -> str:
        if not self.success:
            return f"Export failed: {self.error}"
        return f"Exported .omny v2: {self.output_path}"


def export_v2(
    onnx_path: Union[str, Path],
    output: Union[str, Path],
    config: Optional[Union[dict, ExportV2Config]] = None,
) -> ExportV2Result:
    """
    Export an ONNX model to .omny v2 format.

    Note: Native processors are separate executables installed on the system,
    not bundled in the .omny file.

    Args:
        onnx_path: Path to ONNX file
        output: Output path for .omny file
        config: v2 export configuration

    Returns:
        ExportV2Result with details
    """
    # Parse config
    if config is None:
        cfg = ExportV2Config()
    elif isinstance(config, dict):
        cfg = ExportV2Config(**config)
    else:
        cfg = config

    onnx_path = Path(onnx_path)
    output_path = Path(output)

    if not onnx_path.exists():
        return ExportV2Result(
            success=False,
            output_path=None,
            manifest=None,
            error=f"ONNX file not found: {onnx_path}",
        )

    try:
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))

        if cfg.model_name is None:
            cfg.model_name = onnx_path.stem

        # Extract input/output tensor names
        input_names = []
        for inp in onnx_model.graph.input:
            if not any(init.name == inp.name for init in onnx_model.graph.initializer):
                input_names.append(inp.name)

        output_names = [out.name for out in onnx_model.graph.output]

        # Calculate model size
        total_bytes = sum(
            len(init.raw_data) if init.raw_data else 0
            for init in onnx_model.graph.initializer
        )
        total_mb = int(total_bytes / (1024 * 1024))

        # Calculate parameters
        total_params = 0
        for init in onnx_model.graph.initializer:
            num_elements = 1
            for dim in init.dims:
                num_elements *= dim
            total_params += num_elements

        # Inference memory estimate
        inference_memory_mb = cfg.inference_memory_mb or int(total_mb * 1.5)

        # Build preprocess config
        preprocess = None
        if cfg.preprocess_resize or cfg.preprocess_normalize_mean:
            preprocess = PreprocessConfig(
                resize=cfg.preprocess_resize,
                resize_mode=cfg.preprocess_resize_mode,
                normalize_mean=cfg.preprocess_normalize_mean,
                normalize_std=cfg.preprocess_normalize_std,
                scale=cfg.preprocess_scale,
                channel_order=cfg.preprocess_channel_order,
            )

        # Build postprocess config
        postprocess = None
        if cfg.postprocess_type != "raw":
            postprocess = PostprocessConfig(
                postprocess_type=cfg.postprocess_type,
                confidence_threshold=cfg.postprocess_confidence_threshold,
                nms_threshold=cfg.postprocess_nms_threshold,
                labels=cfg.postprocess_labels,
            )

        # Build manifest
        manifest = OmnyV2Manifest(
            model_name=cfg.model_name,
            model_architecture=cfg.model_architecture,
            task=cfg.task,
            total_params=total_params,
            total_size_mb=total_mb,
            inference_memory_mb=inference_memory_mb,
            input_type=cfg.input_type,
            input_format=cfg.input_format,
            input_tensor_names=input_names,
            output_type=cfg.output_type,
            output_format=cfg.output_format,
            output_tensor_names=output_names,
            processor=ProcessorConfig(
                processor_type=cfg.processor_type,
                processor_id=cfg.processor_id,
            ),
            preprocess=preprocess,
            postprocess=postprocess,
            sharding={
                "min_vram_mb": cfg.min_vram_mb,
                "max_shard_size_mb": cfg.max_shard_size_mb,
                "min_shards": cfg.min_shards,
                "max_shards": cfg.max_shards,
            },
        )

        # Save v2 format
        save_omny_v2(onnx_model, output_path, manifest)

        return ExportV2Result(
            success=True,
            output_path=output_path,
            manifest=manifest,
        )

    except Exception as e:
        return ExportV2Result(
            success=False,
            output_path=None,
            manifest=None,
            error=str(e),
        )


def upgrade_to_v2(
    omny_v1_path: Union[str, Path],
    output: Union[str, Path],
    task: str = "",
    processor_type: str = "none",
    processor_id: Optional[str] = None,
) -> ExportV2Result:
    """
    Upgrade an existing .omny v1 file to v2 format.

    Args:
        omny_v1_path: Path to existing .omny v1 file
        output: Output path for v2 file
        task: Task type ("ocr", "detection", "embedding", etc.)
        processor_type: "native" or "none"
        processor_id: For native processors (e.g., "clip", "ocr", "sam2")

    Returns:
        ExportV2Result
    """
    from .format import load_omny

    omny_v1_path = Path(omny_v1_path)
    output_path = Path(output)

    try:
        # Load v1 file
        model, metadata = load_omny(omny_v1_path)

        if metadata is None:
            return ExportV2Result(
                success=False,
                output_path=None,
                manifest=None,
                error="No OmnyNet metadata found in v1 file",
            )

        # Create v2 manifest from v1 metadata
        manifest = OmnyV2Manifest.from_v1_metadata(metadata)
        manifest.task = task
        manifest.processor = ProcessorConfig(
            processor_type=processor_type,
            processor_id=processor_id,
        )

        # Save as v2
        save_omny_v2(model, output_path, manifest)

        return ExportV2Result(
            success=True,
            output_path=output_path,
            manifest=manifest,
        )

    except Exception as e:
        return ExportV2Result(
            success=False,
            output_path=None,
            manifest=None,
            error=str(e),
        )
