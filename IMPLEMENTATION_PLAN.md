# OmnyNet Export - Implementation Plan

**Created**: 2026-01-06
**Purpose**: Convert PyTorch models to `.omny` format for distributed inference on OmnyNet

---

## Context & Vision

### The Problem
- ONNX models lack metadata needed for reliable distributed inference
- Graph cutting at arbitrary points fails due to unknown shapes
- No standardized way to shard models across heterogeneous hardware

### The Solution
Create `.omny` format - ONNX with embedded cut points, shapes, and memory estimates.

```
PyTorch Model (.pt)
        │
        ▼
┌─────────────────────┐
│  omnynet-export     │
│                     │
│  1. Load model      │
│  2. Trace           │
│  3. Convert to ONNX │
│  4. Find cut points │
│  5. Profile memory  │
│  6. Embed metadata  │
└─────────────────────┘
        │
        ▼
   model.omny (100% reliable for distributed inference)
```

---

## Hardware Constraints

Based on testing with actual hardware:

| Machine | GPU 1 | GPU 2 |
|---------|-------|-------|
| 3ts | AMD R7 M340 (2GB) | Intel HD 620 (2GB) |
| MacBook Pro | Radeon Pro 555X (4GB) | Intel UHD 630 (1.5GB) |

### Design Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| **Min VRAM baseline** | 1.5GB | Smallest GPU in fleet (Intel UHD 630) |
| **Max shard size** | ~1.2GB | Leave 300MB headroom for runtime |
| **Target overhead** | 20-25% | OS, buffers, activations |

---

## Supported Input Formats

| Format | Support | Priority |
|--------|---------|----------|
| **PyTorch (.pt/.pth)** | Full export | P0 (now) |
| **Existing ONNX (.onnx)** | Enrich with metadata | P0 (now) |
| TensorFlow | Via tf2onnx | P3 (later) |
| JAX | Via jax2onnx | P3 (later) |

**Current Focus**: PyTorch + existing ONNX for the 4 Genesis models.

---

## Target Models

For Genesis AI, we need 4 models:

| Model | Source | Size (approx) | Priority |
|-------|--------|---------------|----------|
| CLIP | OpenAI/HuggingFace | ~400MB | P0 (smallest, test first) |
| SAM2 | Meta | ~2.5GB | P1 |
| Grounded SAM2 | IDEA-Research | ~3GB | P2 |
| OCR | TBD | TBD | P2 |

---

## Project Structure

```
omnynet-export/
├── pyproject.toml           # Package config
├── README.md                 # Public documentation
├── FORMAT_SPEC.md           # .omny format specification
├── IMPLEMENTATION_PLAN.md   # This file
├── LICENSE                  # MIT License
├── omnynet_export/
│   ├── __init__.py          # Package exports
│   ├── cli.py               # Command-line interface
│   ├── exporter.py          # Main export logic
│   ├── tracer.py            # PyTorch model tracing
│   ├── converter.py         # ONNX conversion
│   ├── analyzer.py          # Cut point detection
│   ├── profiler.py          # Memory profiling
│   ├── metadata.py          # Metadata generation
│   └── format.py            # .omny file I/O
├── tests/
│   ├── test_exporter.py
│   ├── test_analyzer.py
│   └── test_models/
└── examples/
    ├── export_clip.py
    ├── export_sam2.py
    └── export_grounded_sam2.py
```

---

## Implementation Phases

### Phase 1: Project Setup
- [x] Create project directory
- [x] Create pyproject.toml
- [x] Create README.md
- [x] Create FORMAT_SPEC.md
- [ ] Create GitHub repo (public)
- [ ] Create package structure
- [ ] Create __init__.py with exports

### Phase 2: Core Export Pipeline
- [ ] **exporter.py** - Main export function
  - Load PyTorch model
  - Trace with sample input
  - Convert to ONNX
  - Analyze for cut points
  - Profile memory
  - Generate metadata
  - Save as .omny

- [ ] **tracer.py** - Model tracing
  - Support torch.jit.trace
  - Support torch.export (PyTorch 2.0+)
  - Handle dynamic shapes
  - Extract layer info

- [ ] **converter.py** - ONNX conversion
  - torch.onnx.export wrapper
  - Opset version handling
  - Input/output naming
  - Shape inference

### Phase 3: Analysis & Profiling
- [ ] **analyzer.py** - Cut point detection
  - Find layer boundaries
  - Identify safe cut points (no dynamic shapes)
  - Calculate tensor sizes at each point
  - Rank cut points by quality

- [ ] **profiler.py** - Memory profiling
  - Estimate weight memory per layer
  - Estimate activation memory
  - Calculate cumulative memory at each cut point
  - Generate shard configurations

### Phase 4: Metadata & Format
- [ ] **metadata.py** - Metadata generation
  - Create OmnyNet metadata schema
  - Populate cut points
  - Generate shard configurations
  - Validate constraints

- [ ] **format.py** - .omny file I/O
  - Embed metadata in ONNX
  - Read metadata from .omny
  - Validate .omny files

### Phase 5: CLI
- [ ] **cli.py** - Command-line interface
  - `omnynet-export export` command
  - `omnynet-export inspect` command
  - `omnynet-export validate` command
  - Rich output formatting

### Phase 6: Model-Specific Testing
- [ ] **CLIP**
  - Download/load model
  - Export to .omny
  - Validate cut points
  - Test on 3ts + MacBook

- [ ] **SAM2**
  - Download/load model
  - Export to .omny
  - Validate cut points
  - Test on 3ts + MacBook

- [ ] **Grounded SAM2**
  - Download/load model
  - Export to .omny
  - Validate cut points
  - Test on 3ts + MacBook

- [ ] **OCR**
  - Identify which OCR model
  - Download/load model
  - Export to .omny
  - Validate cut points
  - Test on 3ts + MacBook

---

## Key Decisions Made

### Format: ONNX + Metadata (not new container)
- `.omny` is valid ONNX with metadata in `metadata_props`
- Can be read by standard ONNX tools
- OmnyNet runtime reads metadata for distributed inference

### Export Path: From Source (not enriching existing ONNX)
- Re-export from PyTorch for 100% reliability
- Full control over conversion
- Exact shapes from tracing
- Can fall back to enriching existing ONNX later

### Fixed Shard Configurations
- Not arbitrary sharding - pre-defined, tested configurations
- Each .omny file specifies allowed shard counts
- Runtime picks from allowed list based on available nodes

### Open Source Strategy
- omnynet-export: **Public** (MIT license)
- omny-compute: **Private** (distributed runtime)
- Format spec: **Public** (enables ecosystem)

---

## API Design

### Python API
```python
from omnynet_export import export_model, enrich_onnx, inspect_omny

# Option 1: Export from PyTorch
result = export_model(
    model="path/to/model.pt",
    # OR model=pytorch_model_instance,
    output="model.omny",
    sample_input={"image": torch.randn(1, 3, 1024, 1024)},
    config={
        "min_vram_mb": 1500,
        "max_shard_size_mb": 1200,
    }
)

# Option 2: Enrich existing ONNX
result = enrich_onnx(
    onnx_path="model.onnx",
    output="model.omny",
    sample_input={"image": np.random.randn(1, 3, 1024, 1024).astype(np.float32)},
)

print(f"Exported with {len(result.cut_points)} cut points")
print(f"Allowed shards: {result.allowed_shards}")

# Inspect an .omny file
info = inspect_omny("model.omny")
print(info.summary())
```

### CLI
```bash
# Export
omnynet-export export model.pt \
    --sample-input input.npy \
    --output model.omny \
    --min-vram 1500 \
    --max-shard-size 1200

# Inspect
omnynet-export inspect model.omny

# Validate
omnynet-export validate model.omny
```

---

## Testing Plan

### Unit Tests
- [ ] Test ONNX conversion works
- [ ] Test cut point detection finds valid points
- [ ] Test memory profiling produces reasonable estimates
- [ ] Test metadata serialization/deserialization

### Integration Tests
- [ ] Export simple model end-to-end
- [ ] Export CLIP and validate
- [ ] Export SAM2 and validate
- [ ] Load .omny in omny-compute and run inference

### Hardware Tests
Run on actual target hardware:

| Test | 3ts | MacBook Pro |
|------|-----|-------------|
| Load .omny | | |
| Shard model | | |
| Distribute shards | | |
| Run inference | | |
| Verify output | | |

---

## Integration with omny-compute

Once .omny files are created, update omny-compute to:

1. **Read .omny metadata**
   ```rust
   // In src/graph/omny_format.rs
   pub fn load_omny_metadata(path: &Path) -> Result<OmnyMetadata>;
   ```

2. **Use embedded cut points**
   ```rust
   // Don't use heuristics, use metadata
   let cut_points = metadata.cut_points;
   let shard_config = metadata.best_config_for_nodes(available_nodes);
   ```

3. **Extract at known-safe boundaries**
   ```rust
   // Guaranteed to work because cut points are pre-validated
   extractor.extract_at_cut_points(&cut_points)?;
   ```

---

## Quick Start After Context Compaction

If you're Claude and context was compacted:

1. Read this file: `IMPLEMENTATION_PLAN.md`
2. Check the todo list in this file
3. Read `FORMAT_SPEC.md` for .omny format details
4. Continue from the last unchecked item

**Current State**: Creating project structure and core modules.

**Goal**: Export 4 models (CLIP, SAM2, Grounded SAM2, OCR) to .omny format that work reliably on OmnyNet distributed inference.

---

## Changelog

- **2026-01-06**: Initial plan created
  - Defined .omny format
  - Set hardware constraints (1.5GB baseline)
  - Identified 4 target models
  - Created project structure
