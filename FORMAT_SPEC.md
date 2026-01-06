# .omny Format Specification

**Version**: 1.0
**Status**: Draft

## Overview

`.omny` is an extension of the ONNX format designed for reliable distributed inference. It embeds metadata required for sharding models across multiple devices with heterogeneous hardware.

## File Structure

An `.omny` file is a valid ONNX protobuf file with additional metadata stored in the `metadata_props` field of the `ModelProto`.

```
model.omny
├── ONNX ModelProto
│   ├── ir_version
│   ├── opset_import
│   ├── producer_name: "omnynet-export"
│   ├── producer_version: "1.0.0"
│   ├── graph (standard ONNX graph)
│   └── metadata_props ← OmnyNet metadata here
│       ├── omnynet_version
│       ├── omnynet_metadata (JSON blob)
│       └── ...
```

## Metadata Schema

The `omnynet_metadata` field contains a JSON-encoded object:

```json
{
  "version": "1.0",
  "model": {
    "name": "sam2-large",
    "architecture": "vision_transformer",
    "total_params": 312000000,
    "total_size_mb": 1200,
    "inference_memory_mb": 2400
  },
  "inputs": [
    {
      "name": "image",
      "shape": [1, 3, 1024, 1024],
      "dtype": "float32"
    }
  ],
  "outputs": [
    {
      "name": "masks",
      "shape": [1, 4, 256, 256],
      "dtype": "float32"
    }
  ],
  "cut_points": [
    {
      "id": "cut_1",
      "after_node": "encoder.layer.12",
      "tensor_name": "hidden_states_12",
      "shape": [1, 64, 64, 768],
      "dtype": "float32",
      "cumulative_memory_mb": 600,
      "shard_memory_mb": 600
    },
    {
      "id": "cut_2",
      "after_node": "encoder.layer.24",
      "tensor_name": "hidden_states_24",
      "shape": [1, 64, 64, 768],
      "dtype": "float32",
      "cumulative_memory_mb": 1200,
      "shard_memory_mb": 600
    }
  ],
  "sharding": {
    "min_vram_mb": 1500,
    "max_shard_size_mb": 1200,
    "min_shards": 2,
    "max_shards": 6,
    "allowed_shards": [2, 3, 4, 6],
    "configurations": [
      {
        "num_shards": 2,
        "memory_per_shard_mb": [1200, 1200],
        "cut_point_ids": ["cut_2"]
      },
      {
        "num_shards": 4,
        "memory_per_shard_mb": [600, 600, 600, 600],
        "cut_point_ids": ["cut_1", "cut_2", "cut_3"]
      }
    ]
  },
  "export_info": {
    "exported_at": "2026-01-06T12:00:00Z",
    "exporter_version": "0.1.0",
    "source_framework": "pytorch",
    "source_version": "2.1.0",
    "onnx_opset": 17
  }
}
```

## Field Definitions

### model

| Field | Type | Description |
|-------|------|-------------|
| name | string | Human-readable model name |
| architecture | string | Model architecture type (e.g., "vision_transformer", "cnn") |
| total_params | int | Total number of parameters |
| total_size_mb | int | Model file size in MB |
| inference_memory_mb | int | Total VRAM needed for inference (weights + activations) |

### inputs / outputs

| Field | Type | Description |
|-------|------|-------------|
| name | string | Tensor name in ONNX graph |
| shape | int[] | Tensor shape (use -1 for dynamic dimensions) |
| dtype | string | Data type ("float32", "float16", "int64", etc.) |

### cut_points

Cut points define safe locations where the model can be split for distributed inference.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier for this cut point |
| after_node | string | ONNX node name after which to cut |
| tensor_name | string | Output tensor name at this cut point |
| shape | int[] | Tensor shape at this point |
| dtype | string | Data type |
| cumulative_memory_mb | int | Total memory from start to this point |
| shard_memory_mb | int | Memory for just this shard (from previous cut) |

### sharding

| Field | Type | Description |
|-------|------|-------------|
| min_vram_mb | int | Minimum VRAM required per node |
| max_shard_size_mb | int | Maximum memory per shard |
| min_shards | int | Minimum number of shards supported |
| max_shards | int | Maximum number of shards supported |
| allowed_shards | int[] | List of tested/supported shard counts |
| configurations | object[] | Pre-computed shard configurations |

### configurations

Each configuration describes a specific way to shard the model:

| Field | Type | Description |
|-------|------|-------------|
| num_shards | int | Number of shards in this configuration |
| memory_per_shard_mb | int[] | Memory requirement for each shard |
| cut_point_ids | string[] | Cut points to use (N-1 cuts for N shards) |

## Constraints

### VRAM Baseline

- **Minimum VRAM per node**: 1.5GB
- **Maximum shard size**: 1.2GB (leaves 300MB headroom)
- **Target overhead**: 20-25% of VRAM for runtime buffers

### Cut Point Requirements

A valid cut point must:
1. Have a fully defined output shape (no dynamic dimensions except batch)
2. Have exactly one output tensor (no multi-output cuts)
3. Not be inside a control flow block (if/loop)
4. Have reasonable tensor size for network transfer

## File Extension

- **Extension**: `.omny`
- **MIME type**: `application/x-omnynet-model`
- **Magic bytes**: Same as ONNX (starts with protobuf)

## Validation

An `.omny` file is valid if:
1. It is a valid ONNX file
2. It contains `omnynet_version` in metadata_props
3. It contains `omnynet_metadata` with valid JSON
4. All cut points reference existing nodes in the graph
5. Memory estimates are positive and reasonable
6. At least one shard configuration is defined

## Backwards Compatibility

- `.omny` files are valid ONNX files and can be loaded by any ONNX runtime
- The metadata is ignored by standard ONNX tools
- OmnyNet runtime requires the metadata for distributed inference

## Example: Reading Metadata

```python
import onnx
import json

model = onnx.load("model.omny")

# Get OmnyNet metadata
for prop in model.metadata_props:
    if prop.key == "omnynet_metadata":
        metadata = json.loads(prop.value)
        print(f"Cut points: {len(metadata['cut_points'])}")
        print(f"Allowed shards: {metadata['sharding']['allowed_shards']}")
```

## Changelog

- **1.0** (2026-01-06): Initial specification
