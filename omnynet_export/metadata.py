"""
OmnyNet metadata schema for .omny files.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime


@dataclass
class CutPoint:
    """A safe location to cut the model for distributed inference."""

    id: str
    after_node: str
    tensor_name: str
    shape: list[int]
    dtype: str = "float32"
    cumulative_memory_mb: int = 0
    shard_memory_mb: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "after_node": self.after_node,
            "tensor_name": self.tensor_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "cumulative_memory_mb": self.cumulative_memory_mb,
            "shard_memory_mb": self.shard_memory_mb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CutPoint":
        return cls(**data)


@dataclass
class ShardConfig:
    """A specific shard configuration."""

    num_shards: int
    memory_per_shard_mb: list[int]
    cut_point_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "num_shards": self.num_shards,
            "memory_per_shard_mb": self.memory_per_shard_mb,
            "cut_point_ids": self.cut_point_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShardConfig":
        return cls(**data)


@dataclass
class ModelInfo:
    """Model metadata."""

    name: str
    architecture: str = "unknown"
    total_params: int = 0
    total_size_mb: int = 0
    inference_memory_mb: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "total_params": self.total_params,
            "total_size_mb": self.total_size_mb,
            "inference_memory_mb": self.inference_memory_mb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return cls(**data)


@dataclass
class TensorSpec:
    """Input/output tensor specification."""

    name: str
    shape: list[int]
    dtype: str = "float32"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorSpec":
        return cls(**data)


@dataclass
class ShardingConstraints:
    """Constraints for sharding the model."""

    min_vram_mb: int = 1500
    max_shard_size_mb: int = 1200
    min_shards: int = 2
    max_shards: int = 8
    allowed_shards: list[int] = field(default_factory=lambda: [2, 4])
    configurations: list[ShardConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "min_vram_mb": self.min_vram_mb,
            "max_shard_size_mb": self.max_shard_size_mb,
            "min_shards": self.min_shards,
            "max_shards": self.max_shards,
            "allowed_shards": self.allowed_shards,
            "configurations": [c.to_dict() for c in self.configurations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShardingConstraints":
        configs = [ShardConfig.from_dict(c) for c in data.get("configurations", [])]
        return cls(
            min_vram_mb=data.get("min_vram_mb", 1500),
            max_shard_size_mb=data.get("max_shard_size_mb", 1200),
            min_shards=data.get("min_shards", 2),
            max_shards=data.get("max_shards", 8),
            allowed_shards=data.get("allowed_shards", [2, 4]),
            configurations=configs,
        )


@dataclass
class ExportInfo:
    """Information about the export process."""

    exported_at: str = ""
    exporter_version: str = "0.1.0"
    source_framework: str = "pytorch"
    source_version: str = ""
    onnx_opset: int = 17
    validated: bool = False  # Whether cut points were validated at runtime

    def __post_init__(self):
        if not self.exported_at:
            self.exported_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "exported_at": self.exported_at,
            "exporter_version": self.exporter_version,
            "source_framework": self.source_framework,
            "source_version": self.source_version,
            "onnx_opset": self.onnx_opset,
            "validated": self.validated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExportInfo":
        return cls(**data)


@dataclass
class OmnyMetadata:
    """Complete OmnyNet metadata for a .omny file."""

    version: str = "1.0"
    model: ModelInfo = field(default_factory=lambda: ModelInfo(name="unknown"))
    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    cut_points: list[CutPoint] = field(default_factory=list)
    sharding: ShardingConstraints = field(default_factory=ShardingConstraints)
    export_info: ExportInfo = field(default_factory=ExportInfo)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model": self.model.to_dict(),
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "cut_points": [c.to_dict() for c in self.cut_points],
            "sharding": self.sharding.to_dict(),
            "export_info": self.export_info.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "OmnyMetadata":
        return cls(
            version=data.get("version", "1.0"),
            model=ModelInfo.from_dict(data.get("model", {})),
            inputs=[TensorSpec.from_dict(i) for i in data.get("inputs", [])],
            outputs=[TensorSpec.from_dict(o) for o in data.get("outputs", [])],
            cut_points=[CutPoint.from_dict(c) for c in data.get("cut_points", [])],
            sharding=ShardingConstraints.from_dict(data.get("sharding", {})),
            export_info=ExportInfo.from_dict(data.get("export_info", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OmnyMetadata":
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Model: {self.model.name}",
            f"Architecture: {self.model.architecture}",
            f"Parameters: {self.model.total_params:,}",
            f"Size: {self.model.total_size_mb} MB",
            f"Inference Memory: {self.model.inference_memory_mb} MB",
            "",
            f"Inputs: {len(self.inputs)}",
        ]
        for inp in self.inputs:
            lines.append(f"  - {inp.name}: {inp.shape} ({inp.dtype})")

        lines.append(f"Outputs: {len(self.outputs)}")
        for out in self.outputs:
            lines.append(f"  - {out.name}: {out.shape} ({out.dtype})")

        lines.append("")
        lines.append(f"Cut Points: {len(self.cut_points)}")
        for cp in self.cut_points:
            lines.append(f"  - {cp.id}: {cp.tensor_name} {cp.shape} ({cp.shard_memory_mb} MB)")

        lines.append("")
        lines.append(f"Allowed Shards: {self.sharding.allowed_shards}")
        lines.append(f"Min VRAM: {self.sharding.min_vram_mb} MB")
        lines.append(f"Max Shard Size: {self.sharding.max_shard_size_mb} MB")

        return "\n".join(lines)
