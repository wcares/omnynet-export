"""
OmnyNet Export - Convert PyTorch models to .omny format for distributed inference.

Usage:
    from omnynet_export import export_model, inspect_omny

    # Export a PyTorch model
    result = export_model(
        model="path/to/model.pt",
        output="model.omny",
        sample_input={"image": torch.randn(1, 3, 224, 224)},
    )

    # Inspect an .omny file
    info = inspect_omny("model.omny")
    print(info.summary())
"""

__version__ = "0.1.0"

from .exporter import export_model, enrich_onnx, ExportResult, ExportConfig
from .format import inspect_omny, validate_omny, OmnyInfo
from .metadata import OmnyMetadata, CutPoint, ShardConfig
from .validator import validate_cut_point, validate_sharding, find_valid_cut_points

__all__ = [
    "export_model",
    "enrich_onnx",
    "ExportResult",
    "ExportConfig",
    "inspect_omny",
    "validate_omny",
    "OmnyInfo",
    "OmnyMetadata",
    "CutPoint",
    "ShardConfig",
    "validate_cut_point",
    "validate_sharding",
    "find_valid_cut_points",
    "__version__",
]
