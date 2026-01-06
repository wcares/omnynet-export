# omnynet-export

Export PyTorch models to `.omny` format for distributed inference on [OmnyNet](https://omnynet.io).

## What is .omny?

`.omny` is an ONNX-based format with embedded metadata for reliable distributed inference:

- **Cut points**: Pre-defined safe locations to shard the model
- **Shapes**: Exact tensor shapes at each cut point
- **Memory estimates**: Runtime memory requirements per shard
- **Constraints**: Min/max shards, VRAM requirements

## Installation

```bash
pip install omnynet-export
```

## Quick Start

```bash
# Export a PyTorch model
omnynet-export export model.pt --sample-input input.npy --output model.omny

# Inspect an .omny file
omnynet-export inspect model.omny
```

## Python API

```python
from omnynet_export import export_model

export_model(
    model_path="sam2.pt",
    output_path="sam2.omny",
    sample_input={"image": torch.randn(1, 3, 1024, 1024)},
    target_shard_size_mb=1200,  # Target ~1.2GB per shard
)
```

## Supported Models

Currently tested and optimized for:

| Model | Status | Notes |
|-------|--------|-------|
| CLIP | Supported | Vision + Text encoder |
| SAM2 | Supported | Segment Anything 2 |
| Grounded SAM2 | Supported | Grounded + SAM2 |
| OCR | Supported | Text recognition |

## .omny Format

See [FORMAT_SPEC.md](FORMAT_SPEC.md) for the full specification.

### Quick Overview

An `.omny` file is a standard ONNX file with additional metadata in the `metadata_props` field:

```yaml
omnynet_version: "1.0"
model_name: "sam2-large"
total_memory_mb: 2400

cut_points:
  - after_node: "encoder.layer.12"
    shape: [1, 64, 64, 768]
    cumulative_memory_mb: 600

  - after_node: "encoder.layer.24"
    shape: [1, 64, 64, 768]
    cumulative_memory_mb: 1200

sharding:
  min_vram_mb: 1500
  max_shard_size_mb: 1200
  allowed_shards: [2, 3, 4, 6]
```

## Hardware Requirements

Designed for distributed inference across heterogeneous hardware:

- **Minimum VRAM**: 1.5GB per node
- **Maximum shard size**: ~1.2GB (with 300MB headroom)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.
