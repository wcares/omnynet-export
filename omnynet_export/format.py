"""
.omny file format I/O - read/write ONNX with embedded OmnyNet metadata.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import onnx
from onnx import ModelProto

from .metadata import OmnyMetadata

# Metadata keys in ONNX metadata_props
OMNYNET_VERSION_KEY = "omnynet_version"
OMNYNET_METADATA_KEY = "omnynet_metadata"


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
