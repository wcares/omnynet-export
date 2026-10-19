# Native Processors for OmnyNet

This directory contains native processor executables for preprocessing and postprocessing model inputs/outputs. Each model type has its own processor that handles the specific requirements for that model architecture.

## Architecture

Native processors are separate executables that communicate via stdin/stdout:

```
Input Image → Processor preprocess → ONNX Runtime → Processor postprocess → Output JSON
```

The .omny v2 format specifies which processor to use in its manifest, but processors are installed separately:

```
~/.local/share/omnynet/processors/
├── clip-processor
├── ocr-processor
├── sam2-processor
└── dino-processor
```

## Available Processors

| Processor | Model Types | Preprocessing | Output Type |
|-----------|-------------|---------------|-------------|
| `clip` | CLIP ViT | shortest_center_crop to 224x224, bicubic, CLIP normalization | embedding |
| `ocr` | PaddleOCR | resize to target height, pad width to multiple of 32, ImageNet normalization | ocr/detection |
| `sam2` | SAM2 | resize longest side to 1024, pad to square, ImageNet normalization | segmentation |
| `dino` | Grounding DINO | resize to fit 800x1333, ImageNet normalization | detection |
| `doc-layout` | docling-layout-heron (RT-DETRv2, Apache-2.0) | resize to 640x640 (bilinear), rescale 1/255, NO mean/std | detection (17 page-layout classes: text, table, picture, section_header, ...) |
| `table-structure` | TableFormerV2 (Apache-2.0, PREVIEW; 3 ONNX graphs: `encoder.onnx`, `decoder.onnx`, `bbox.onnx`) | runs the model itself with `ort` (a greedy token loop): resize to 448x448 (bilinear), rescale 1/255, ImageNet mean/std | table_structure (OTSL tokens, grid with spans, one box per cell) |

## Building Processors

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build All Processors

```bash
cd processors
cargo build --release
```

Output binaries will be in `target/release/`:
- `clip-processor`
- `ocr-processor`
- `sam2-processor`
- `dino-processor`

### Install Processors

```bash
mkdir -p ~/.local/share/omnynet/processors
cp target/release/*-processor ~/.local/share/omnynet/processors/
```

## Processor Interface

All processors accept `preprocess` or `postprocess` as the first argument:

### preprocess

```bash
cat image.png | clip-processor preprocess > tensors.json
```

- **Input:** Raw image bytes (PNG/JPG) on stdin
- **Output:** JSON tensor map on stdout

Output format:
```json
{
  "pixel_values": [0.1, 0.2, 0.3, ...]
}
```

### postprocess

```bash
cat model_output.json | clip-processor postprocess > result.json
```

- **Input:** JSON tensor map from ONNX model on stdin
- **Output:** Structured JSON result on stdout

Output format (varies by model type):
```json
{
  "type": "embedding",
  "embedding": [0.1, 0.2, ...]
}
```

## Using Processors with omnynet-export

When exporting a model, specify the native processor:

```bash
# Export CLIP with native processor
omnynet-export export-v2 clip.onnx -o clip.omny \
    --task embedding \
    --processor native \
    --processor-id clip
```

Or programmatically in Python:

```python
from omnynet_export import export_v2, ExportV2Config

config = ExportV2Config(
    model_name="clip-vision",
    task="embedding",
    processor_type="native",
    processor_id="clip",
)

result = export_v2("clip.onnx", "clip.omny", config)
```

## Creating a New Processor

1. Create new crate in workspace:
```bash
mkdir mymodel-native && cd mymodel-native
```

2. Add `Cargo.toml`:
```toml
[package]
name = "mymodel-processor"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "mymodel-processor"
path = "src/main.rs"

[dependencies]
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

3. Add to workspace `Cargo.toml`:
```toml
members = [
    # ...existing members...
    "mymodel-native",
]
```

4. Implement `src/main.rs`:
```rust
use std::io::{self, Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("preprocess") => preprocess(),
        Some("postprocess") => postprocess(),
        _ => {
            eprintln!("Usage: mymodel-processor <preprocess|postprocess>");
            std::process::exit(1);
        }
    }
}

fn preprocess() {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).unwrap();

    // Process image and create tensor
    let tensor = process_image(&input);

    let output = serde_json::json!({
        "pixel_values": tensor
    });
    io::stdout().write_all(serde_json::to_vec(&output).unwrap().as_slice()).unwrap();
}

fn postprocess() {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).unwrap();

    let tensors: serde_json::Value = serde_json::from_slice(&input).unwrap();

    let result = serde_json::json!({
        "type": "mymodel",
        "output": tensors
    });
    io::stdout().write_all(serde_json::to_vec(&result).unwrap().as_slice()).unwrap();
}
```

5. Build and install:
```bash
cargo build --release
cp target/release/mymodel-processor ~/.local/share/omnynet/processors/
```

## Normalization Constants

### CLIP
```rust
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
```

### ImageNet (SAM2, DINO, OCR)
```rust
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
```

## Processor Search Paths

omny-compute searches for processors in this order:

1. `~/.local/share/omnynet/processors/<id>-processor`
2. `/usr/local/share/omnynet/processors/<id>-processor`
3. `/usr/share/omnynet/processors/<id>-processor`
4. System PATH (as `<id>-processor`)

## Cross-Platform Support

Processors are compiled natively for each platform:

```bash
# Linux
cargo build --release

# macOS (from macOS or cross-compile)
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Windows (cross-compile)
cargo build --release --target x86_64-pc-windows-gnu
```

## table-structure-processor

`table-structure-processor <model-dir>` reads a table crop (PNG/JPEG, or `{"image": "<base64>"}`) on stdin and prints the table's structure as JSON. `ORT_DYLIB_PATH` names `libonnxruntime` (ort load-dynamic, same as `ocr`). It is a native processor that drives its model itself because the work is a loop (encoder once, decoder per token, box head once), so it does not fit the preprocess / run / postprocess split above. `<model-dir>` (or `OMNYNET_TABLE_MODEL_DIR`) holds the three graphs. The graphs are exported from `docling-project/TableFormerV2` with `table-structure-native/export/*.py` (needs torch, onnxscript and `docling-ibm-models` at EXPORT time only). Output boxes are xyxy, normalised 0..1 to the crop.

Verified 2026-10-19 on three table crops cut from the W-9 by `doc-layout`: the token sequences equal an onnxruntime (Python) greedy run of the same graphs token for token, boxes within 9e-5 (the two resize implementations differ slightly); 1.0 to 1.4 s per crop on 4 threads including the three model loads, 420 to 475 MB RSS.
