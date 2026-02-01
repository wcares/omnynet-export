"""
.omny file format I/O - read/write ONNX with embedded OmnyNet metadata.

Supports two format versions:
- v1: Standard ONNX with metadata in metadata_props (backward compatible)
- v2: Custom binary format with JSON manifest (processor is external native executable)

v2 Format Structure:
    [0-3]    Magic: "OMNY"
    [4-7]    Version: u32 (2)
    [8-15]   Flags: u64
    [16-23]  Model offset: u64
    [24-31]  Model size: u64
    [32-39]  Reserved offset: u64 (unused, set to 0)
    [40-47]  Reserved size: u64 (unused, set to 0)
    [48-55]  Manifest offset: u64
    [56-63]  Manifest size: u64
    [64-255] Reserved (padding)
    [256..]  Data sections (model, manifest)

Native processors are separate executables installed at:
    ~/.local/share/omnynet/processors/<processor_id>-processor
"""

import json
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union

import onnx
from onnx import ModelProto

from .metadata import OmnyMetadata

# Metadata keys in ONNX metadata_props
OMNYNET_VERSION_KEY = "omnynet_version"
OMNYNET_METADATA_KEY = "omnynet_metadata"

# v2 format constants
OMNY_MAGIC = b"OMNY"
OMNY_VERSION_2 = 2
HEADER_SIZE = 256


@dataclass
class OmnyInfo:
    """Information extracted from an .omny file."""

    path: Path
    is_valid: bool
    metadata: Optional[OmnyMetadata]
    onnx_model: Optional[ModelProto]
    error: Optional[str] = None

    def summary(self) -> str:
        if not self.is_valid:
            return f"Invalid .omny file: {self.error}"
        if self.metadata:
            return self.metadata.summary()
        return "No metadata found"


def embed_metadata(model: ModelProto, metadata: OmnyMetadata) -> ModelProto:
    """Embed OmnyNet metadata into an ONNX model."""
    # Remove existing omnynet metadata if present
    keys_to_remove = {OMNYNET_VERSION_KEY, OMNYNET_METADATA_KEY}
    props_to_keep = [
        (prop.key, prop.value)
        for prop in model.metadata_props
        if prop.key not in keys_to_remove
    ]
    del model.metadata_props[:]
    for key, value in props_to_keep:
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    # Add version
    version_prop = model.metadata_props.add()
    version_prop.key = OMNYNET_VERSION_KEY
    version_prop.value = metadata.version

    # Add metadata JSON
    metadata_prop = model.metadata_props.add()
    metadata_prop.key = OMNYNET_METADATA_KEY
    metadata_prop.value = metadata.to_json()

    # Update producer info
    model.producer_name = "omnynet-export"
    model.producer_version = metadata.export_info.exporter_version

    return model


def extract_metadata(model: ModelProto) -> Optional[OmnyMetadata]:
    """Extract OmnyNet metadata from an ONNX model."""
    metadata_json = None

    for prop in model.metadata_props:
        if prop.key == OMNYNET_METADATA_KEY:
            metadata_json = prop.value
            break

    if metadata_json:
        return OmnyMetadata.from_json(metadata_json)

    return None


def save_omny(model: ModelProto, path: Union[str, Path]) -> None:
    """Save an ONNX model with OmnyNet metadata as .omny file."""
    path = Path(path)
    onnx.save(model, str(path))


def load_omny(path: Union[str, Path]) -> tuple[ModelProto, Optional[OmnyMetadata]]:
    """Load an .omny file and extract metadata."""
    path = Path(path)
    model = onnx.load(str(path))
    metadata = extract_metadata(model)
    return model, metadata


def inspect_omny(path: Union[str, Path]) -> OmnyInfo:
    """Inspect an .omny file and return information about it."""
    path = Path(path)

    if not path.exists():
        return OmnyInfo(
            path=path,
            is_valid=False,
            metadata=None,
            onnx_model=None,
            error=f"File not found: {path}",
        )

    try:
        model, metadata = load_omny(path)

        if metadata is None:
            return OmnyInfo(
                path=path,
                is_valid=False,
                metadata=None,
                onnx_model=model,
                error="No OmnyNet metadata found in file",
            )

        return OmnyInfo(
            path=path,
            is_valid=True,
            metadata=metadata,
            onnx_model=model,
        )

    except Exception as e:
        return OmnyInfo(
            path=path,
            is_valid=False,
            metadata=None,
            onnx_model=None,
            error=str(e),
        )


def validate_omny(path: Union[str, Path]) -> tuple[bool, list[str]]:
    """
    Validate an .omny file.

    Returns:
        (is_valid, list of error messages)
    """
    errors = []
    path = Path(path)

    # Check file exists
    if not path.exists():
        return False, [f"File not found: {path}"]

    # Load and check ONNX validity
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as e:
        return False, [f"Invalid ONNX model: {e}"]

    # Check for OmnyNet metadata
    metadata = extract_metadata(model)
    if metadata is None:
        errors.append("No OmnyNet metadata found")
        return False, errors

    # Validate metadata
    if not metadata.cut_points:
        errors.append("No cut points defined")

    if not metadata.sharding.allowed_shards:
        errors.append("No allowed shard configurations")

    if metadata.sharding.min_vram_mb <= 0:
        errors.append("Invalid min_vram_mb")

    if metadata.sharding.max_shard_size_mb <= 0:
        errors.append("Invalid max_shard_size_mb")

    # Validate cut points reference existing nodes
    graph = model.graph
    node_names = {node.name for node in graph.node}
    output_names = set()
    for node in graph.node:
        output_names.update(node.output)

    for cp in metadata.cut_points:
        if cp.after_node and cp.after_node not in node_names:
            errors.append(f"Cut point '{cp.id}' references non-existent node: {cp.after_node}")
        if cp.tensor_name and cp.tensor_name not in output_names:
            errors.append(
                f"Cut point '{cp.id}' references non-existent tensor: {cp.tensor_name}"
            )

    return len(errors) == 0, errors


def has_omny_metadata(path: Union[str, Path]) -> bool:
    """Check if a file has OmnyNet metadata."""
    try:
        model = onnx.load(str(path))
        for prop in model.metadata_props:
            if prop.key == OMNYNET_METADATA_KEY:
                return True
        return False
    except Exception:
        return False


# ============================================================================
# v2 Format Support (Custom Binary with Bundled Processor)
# ============================================================================

@dataclass
class PreprocessConfig:
    """Preprocessing configuration for v2 format."""
    resize: Optional[tuple[int, int]] = None  # [width, height]
    resize_mode: str = "exact"  # "exact", "preserve_aspect", "pad", "shortest_center_crop"
    interpolation: str = "bicubic"  # "bicubic", "lanczos", "bilinear", "nearest"
    normalize_mean: Optional[tuple[float, float, float]] = None  # [R, G, B]
    normalize_std: Optional[tuple[float, float, float]] = None  # [R, G, B]
    scale: float = 255.0
    channel_order: str = "RGB"  # "RGB", "BGR"
    dtype: str = "float32"


@dataclass
class PostprocessConfig:
    """Postprocessing configuration for v2 format."""
    postprocess_type: str = "raw"  # "detection", "classification", "ocr", etc.
    confidence_threshold: float = 0.5
    nms_threshold: Optional[float] = None
    labels: Optional[list[str]] = None
    extra: Optional[dict] = None


@dataclass
class ProcessorConfig:
    """Processor configuration for v2 format."""
    processor_type: str = "none"  # "native", "none"
    processor_id: Optional[str] = None  # For native processors (e.g., "clip", "ocr", "sam2")
    processor_version: Optional[str] = None


@dataclass
class OmnyV2Manifest:
    """Manifest for .omny v2 format."""
    version: str = "2.0"
    model_name: str = "unknown"
    model_architecture: str = ""
    task: str = ""  # "ocr", "detection", "classification", "embedding", etc.
    total_params: int = 0
    total_size_mb: int = 0
    inference_memory_mb: int = 0
    
    # Input/Output specs
    input_type: str = "image"  # "image", "text", "tensor"
    input_format: str = "base64"  # "base64", "bytes", "path"
    input_tensor_names: list[str] = field(default_factory=list)
    output_type: str = "raw"  # "detection", "ocr", "embedding", etc.
    output_format: str = "json"
    output_tensor_names: list[str] = field(default_factory=list)
    
    # Processor info
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    preprocess: Optional[PreprocessConfig] = None
    postprocess: Optional[PostprocessConfig] = None
    
    # Sharding (from v1)
    sharding: Optional[dict] = None
    export_info: Optional[dict] = None
    
    def to_json(self) -> str:
        """Serialize manifest to JSON."""
        data = {
            "version": self.version,
            "model": {
                "name": self.model_name,
                "architecture": self.model_architecture,
                "task": self.task,
                "total_params": self.total_params,
                "total_size_mb": self.total_size_mb,
                "inference_memory_mb": self.inference_memory_mb,
            },
            "processor": {
                "processor_type": self.processor.processor_type,
                "processor_id": self.processor.processor_id,
                "processor_version": self.processor.processor_version,
            },
            "input": {
                "input_type": self.input_type,
                "input_format": self.input_format,
                "tensor_names": self.input_tensor_names,
            },
            "output": {
                "output_type": self.output_type,
                "output_format": self.output_format,
                "tensor_names": self.output_tensor_names,
            },
        }
        
        if self.preprocess:
            data["preprocess"] = {
                "resize": list(self.preprocess.resize) if self.preprocess.resize else None,
                "resize_mode": self.preprocess.resize_mode,
                "interpolation": self.preprocess.interpolation,
                "normalize_mean": list(self.preprocess.normalize_mean) if self.preprocess.normalize_mean else None,
                "normalize_std": list(self.preprocess.normalize_std) if self.preprocess.normalize_std else None,
                "scale": self.preprocess.scale,
                "channel_order": self.preprocess.channel_order,
                "dtype": self.preprocess.dtype,
            }
        
        if self.postprocess:
            data["postprocess"] = {
                "postprocess_type": self.postprocess.postprocess_type,
                "confidence_threshold": self.postprocess.confidence_threshold,
                "nms_threshold": self.postprocess.nms_threshold,
                "labels": self.postprocess.labels,
                "extra": self.postprocess.extra,
            }
        
        if self.sharding:
            data["sharding"] = self.sharding
        
        if self.export_info:
            data["export_info"] = self.export_info
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_v1_metadata(cls, metadata: OmnyMetadata) -> "OmnyV2Manifest":
        """Create v2 manifest from v1 metadata."""
        return cls(
            model_name=metadata.model.name,
            model_architecture=metadata.model.architecture,
            total_params=metadata.model.total_params,
            total_size_mb=metadata.model.total_size_mb,
            inference_memory_mb=metadata.model.inference_memory_mb,
            input_tensor_names=[inp.name for inp in metadata.inputs],
            output_tensor_names=[out.name for out in metadata.outputs],
            sharding={
                "min_vram_mb": metadata.sharding.min_vram_mb,
                "max_shard_size_mb": metadata.sharding.max_shard_size_mb,
                "min_shards": metadata.sharding.min_shards,
                "max_shards": metadata.sharding.max_shards,
                "allowed_shards": metadata.sharding.allowed_shards,
            },
            export_info={
                "exporter_version": metadata.export_info.exporter_version,
                "source_framework": metadata.export_info.source_framework,
                "onnx_opset": metadata.export_info.onnx_opset,
            },
        )


def save_omny_v2(
    model: ModelProto,
    path: Union[str, Path],
    manifest: OmnyV2Manifest,
) -> None:
    """
    Save an ONNX model as .omny v2 file.

    Note: Processors are separate native executables, not bundled in the .omny file.

    Args:
        model: ONNX model
        path: Output path
        manifest: v2 manifest with config
    """
    path = Path(path)

    # Serialize model to bytes
    model_bytes = model.SerializeToString()

    # Serialize manifest
    manifest_bytes = manifest.to_json().encode("utf-8")

    # Calculate offsets (reserved space for backward compatibility)
    model_offset = HEADER_SIZE
    reserved_offset = model_offset + len(model_bytes)  # No processor bytes
    manifest_offset = reserved_offset  # Manifest immediately after model

    # Build header
    header = bytearray(HEADER_SIZE)

    # [0-3] Magic
    header[0:4] = OMNY_MAGIC

    # [4-7] Version
    header[4:8] = struct.pack("<I", OMNY_VERSION_2)

    # [8-15] Flags (reserved)
    header[8:16] = struct.pack("<Q", 0)

    # [16-23] Model offset
    header[16:24] = struct.pack("<Q", model_offset)

    # [24-31] Model size
    header[24:32] = struct.pack("<Q", len(model_bytes))

    # [32-39] Reserved offset (was processor offset)
    header[32:40] = struct.pack("<Q", 0)

    # [40-47] Reserved size (was processor size)
    header[40:48] = struct.pack("<Q", 0)

    # [48-55] Manifest offset
    header[48:56] = struct.pack("<Q", manifest_offset)

    # [56-63] Manifest size
    header[56:64] = struct.pack("<Q", len(manifest_bytes))

    # Write file
    with open(path, "wb") as f:
        f.write(header)
        f.write(model_bytes)
        f.write(manifest_bytes)

    total_kb = (len(header) + len(model_bytes) + len(manifest_bytes)) // 1024
    print(f"Saved .omny v2: {path} ({total_kb} KB)")
    print(f"  Model: {len(model_bytes) // 1024} KB")
    print(f"  Manifest: {len(manifest_bytes)} bytes")
    if manifest.processor.processor_type == "native":
        print(f"  Native processor: {manifest.processor.processor_id}-processor (external)")


def load_omny_v2(path: Union[str, Path]) -> tuple[bytes, OmnyV2Manifest]:
    """
    Load a .omny v2 file.

    Returns:
        (model_bytes, manifest)
    """
    path = Path(path)

    with open(path, "rb") as f:
        data = f.read()

    # Check magic
    if data[0:4] != OMNY_MAGIC:
        raise ValueError(f"Invalid .omny file: bad magic bytes")

    # Check version
    version = struct.unpack("<I", data[4:8])[0]
    if version != OMNY_VERSION_2:
        raise ValueError(f"Unsupported .omny version: {version}")

    # Read offsets
    model_offset = struct.unpack("<Q", data[16:24])[0]
    model_size = struct.unpack("<Q", data[24:32])[0]
    # Skip reserved fields [32-48] (was processor offset/size)
    manifest_offset = struct.unpack("<Q", data[48:56])[0]
    manifest_size = struct.unpack("<Q", data[56:64])[0]

    # Extract sections
    model_bytes = data[model_offset:model_offset + model_size]
    manifest_json = data[manifest_offset:manifest_offset + manifest_size].decode("utf-8")

    # Parse manifest
    manifest_data = json.loads(manifest_json)
    proc_data = manifest_data.get("processor", {})
    manifest = OmnyV2Manifest(
        version=manifest_data.get("version", "2.0"),
        model_name=manifest_data.get("model", {}).get("name", "unknown"),
        model_architecture=manifest_data.get("model", {}).get("architecture", ""),
        task=manifest_data.get("model", {}).get("task", ""),
        total_params=manifest_data.get("model", {}).get("total_params", 0),
        total_size_mb=manifest_data.get("model", {}).get("total_size_mb", 0),
        inference_memory_mb=manifest_data.get("model", {}).get("inference_memory_mb", 0),
        processor=ProcessorConfig(
            processor_type=proc_data.get("processor_type", "none"),
            processor_id=proc_data.get("processor_id"),
        ),
    )

    return model_bytes, manifest


def is_omny_v2(path: Union[str, Path]) -> bool:
    """Check if a file is .omny v2 format."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            return magic == OMNY_MAGIC
    except Exception:
        return False


def detect_omny_version(path: Union[str, Path]) -> Optional[int]:
    """
    Detect the version of an .omny file.
    
    Returns:
        1 for v1 (ONNX with metadata), 2 for v2 (custom binary), None if invalid
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic == OMNY_MAGIC:
                return 2
            # Try loading as ONNX
            f.seek(0)
            # ONNX files start with protobuf format
            # First byte is typically 0x08 (varint field 1) or similar
            if magic[0] == 0x08:
                return 1
        return None
    except Exception:
        return None
