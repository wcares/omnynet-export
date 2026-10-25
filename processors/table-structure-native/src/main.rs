//! Table structure processor (intake reader R2; gateway doc 4.6 F: docling-project/TableFormerV2, Apache-2.0, 52 M parameters, PREVIEW).
//!
//! A table crop image in, the table's structure out: OTSL tokens, the grid they describe (rows, columns, spans) and one box per cell.
//! Unlike the other native processors this one drives its model itself, because the work is a loop: three ONNX graphs run by `ort`
//! (load-dynamic: `ORT_DYLIB_PATH` names `libonnxruntime`), a greedy token loop between them.
//!
//!   cat crop.png | table-structure-processor <model-dir>      (also accepts {"image":"base64..."})
//!
//! `<model-dir>` (or `OMNYNET_TABLE_MODEL_DIR`) holds `encoder.onnx` (448x448 image -> memory [1,196,512]), `decoder.onnx` (token prefix + memory ->
//! logits + hidden states, no KV cache: the whole prefix is re-run each step) and `bbox.onnx` (hidden states of the cell tokens + memory -> boxes).
//! Output: {"type":"table_structure","tokens":[ids],"otsl":[names],"rows":R,"cols":C,"complete":bool,
//!          "cells":[{"row","col","rowspan","colspan","kind":"<fcel>","box":[x1,y1,x2,y2]}]} with boxes normalised 0..1 to the crop (xyxy).
//! Preprocessing as Docling's `table_structure_model_v2`: resize to exactly 448x448 (bilinear), rescale 1/255, ImageNet mean/std.

mod otsl;

use base64::Engine;
use image::imageops::FilterType;
use ort::session::Session;
use serde_json::{json, Value};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const SIZE: u32 = 448;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Docling's `max_length` for this model.
const MAX_TOKENS: usize = 512;
const THREADS: usize = 4;
type AnyErr = Box<dyn std::error::Error>;

fn run_main() {
    let dir = std::env::args().nth(1).or_else(|| std::env::var("OMNYNET_TABLE_MODEL_DIR").ok());
    let Some(dir) = dir else {
        eprintln!("Usage: table-structure-processor <model-dir>   (image on stdin)");
        std::process::exit(1);
    };
    if let Err(e) = run(Path::new(&dir)) {
        eprintln!("table-structure-processor error: {e}");
        std::process::exit(1);
    }
}

fn run(dir: &Path) -> Result<(), AnyErr> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let bytes = image_bytes(&input)?;
    let pixels = image_to_tensor(&bytes)?;
    let out = Model::load(dir)?.read_table(pixels)?;
    io::stdout().write_all(&serde_json::to_vec(&out)?)?;
    Ok(())
}

/// Raw PNG/JPEG bytes (the omny-compute native-processor path) or `{"image": "<base64>"}`.
fn image_bytes(input: &[u8]) -> Result<Vec<u8>, AnyErr> {
    if input.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{') {
        let json: Value = serde_json::from_slice(input)?;
        let b64 = json.get("image").and_then(Value::as_str).ok_or("missing 'image' (base64 PNG/JPEG)")?;
        return Ok(base64::engine::general_purpose::STANDARD.decode(b64)?);
    }
    Ok(input.to_vec())
}

/// CHW float32, resized to 448x448 (triangle filter = PIL bilinear, which widens its support when shrinking), rescaled, mean/std normalised.
fn image_to_tensor(bytes: &[u8]) -> Result<Vec<f32>, AnyErr> {
    let rgb = image::load_from_memory(bytes)?.resize_exact(SIZE, SIZE, FilterType::Triangle).to_rgb8();
    let mut t = Vec::with_capacity((3 * SIZE * SIZE) as usize);
    for c in 0..3 {
        for y in 0..SIZE {
            for x in 0..SIZE {
                t.push((rgb.get_pixel(x, y)[c] as f32 / 255.0 - MEAN[c]) / STD[c]);
            }
        }
    }
    Ok(t)
}

/// Index of the largest value (first on a tie, like `torch.argmax`); NaN never wins.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if x.is_finite() && (!v[best].is_finite() || *x > v[best]) {
            best = i;
        }
    }
    best
}

/// Rows `positions` of a [L, D] row-major matrix, as one [N, D] buffer.
fn gather_rows(hidden: &[f32], dim: usize, positions: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions.len() * dim);
    for &p in positions {
        out.extend_from_slice(&hidden[p * dim..(p + 1) * dim]);
    }
    out
}

struct Model {
    encoder: Session,
    decoder: Session,
    bbox: Session,
}

fn session(path: PathBuf) -> Result<Session, AnyErr> {
    if !path.exists() {
        return Err(format!("model file not found: {}", path.display()).into());
    }
    Ok(Session::builder()?.with_intra_threads(THREADS)?.commit_from_file(path)?)
}

impl Model {
    fn load(dir: &Path) -> Result<Self, AnyErr> {
        Ok(Self { encoder: session(dir.join("encoder.onnx"))?, decoder: session(dir.join("decoder.onnx"))?, bbox: session(dir.join("bbox.onnx"))? })
    }

    fn read_table(&mut self, pixels: Vec<f32>) -> Result<Value, AnyErr> {
        let images = ort::value::Tensor::<f32>::from_array(([1usize, 3, SIZE as usize, SIZE as usize], pixels.into_boxed_slice()))?;
        let enc = self.encoder.run(ort::inputs!["images" => images])?;
        let (mshape, mdata) = enc[0].try_extract_tensor::<f32>()?;
        let mshape: Vec<usize> = mshape.iter().map(|&d| d as usize).collect();
        let memory: Vec<f32> = mdata.to_vec();
        drop(enc);
        let memory_tensor = || ort::value::Tensor::<f32>::from_array((mshape.clone(), memory.clone().into_boxed_slice()));

        let mut ids: Vec<i64> = vec![otsl::BOS];
        let mut complete = false;
        let mut hidden: Vec<f32> = Vec::new();
        let mut dim = 0usize;
        for _ in 0..MAX_TOKENS {
            let input_ids = ort::value::Tensor::<i64>::from_array(([1usize, ids.len()], ids.clone().into_boxed_slice()))?;
            let out = self.decoder.run(ort::inputs!["input_ids" => input_ids, "memory" => memory_tensor()?])?;
            let (lshape, logits) = out[0].try_extract_tensor::<f32>()?;
            let vocab = *lshape.last().ok_or("decoder logits have no shape")? as usize;
            let next = argmax(&logits[logits.len() - vocab..]) as i64;
            let (hshape, h) = out[1].try_extract_tensor::<f32>()?;
            dim = *hshape.last().ok_or("decoder hidden has no shape")? as usize;
            hidden = h.to_vec();
            ids.push(next);
            if next == otsl::EOS {
                complete = true;
                break;
            }
        }
        // hidden states of the sequence WITHOUT the token just appended; the box head reads the cell positions of the sequence so far
        // (all of them: the appended token is EOS when complete, and a cell token's hidden state belongs to the prefix that ended before it).
        let positions: Vec<usize> = ids.iter().enumerate().take(hidden.len() / dim.max(1)).filter(|(_, t)| otsl::is_data_cell(**t)).map(|(i, _)| i).collect();
        let boxes = if positions.is_empty() { Vec::new() } else { self.cell_boxes(&hidden, dim, &positions, memory_tensor()?)? };
        let grid = otsl::parse(&ids);
        let cells: Vec<Value> = grid
            .cells
            .iter()
            .map(|c| {
                let b = boxes.get(c.box_index).copied().unwrap_or([f32::NAN; 4]);
                json!({ "row": c.row, "col": c.col, "rowspan": c.rowspan, "colspan": c.colspan, "kind": otsl::name(c.token),
                        "box": if b.iter().all(|v| v.is_finite()) { json!(b) } else { Value::Null } })
            })
            .collect();
        Ok(json!({ "type": "table_structure", "tokens": ids, "otsl": ids.iter().map(|t| otsl::name(*t)).collect::<Vec<_>>(),
                   "rows": grid.rows, "cols": grid.cols, "complete": complete, "cells": cells }))
    }

    fn cell_boxes(&mut self, hidden: &[f32], dim: usize, positions: &[usize], memory: ort::value::Tensor<f32>) -> Result<Vec<[f32; 4]>, AnyErr> {
        let embs = gather_rows(hidden, dim, positions);
        let cell_embs = ort::value::Tensor::<f32>::from_array(([positions.len(), dim], embs.into_boxed_slice()))?;
        let out = self.bbox.run(ort::inputs!["cell_embs" => cell_embs, "memory" => memory])?;
        let (_, b) = out[0].try_extract_tensor::<f32>()?;
        Ok(b.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tensor_is_chw_resized_and_imagenet_normalised() {
        let mut img = image::RgbImage::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                img.put_pixel(x, y, if x < 2 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) });
            }
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(&mut io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let t = image_to_tensor(&png).unwrap();
        assert_eq!(t.len(), 3 * 448 * 448);
        let plane = 448 * 448;
        let (r, g, b) = (t[10], t[plane + 10], t[2 * plane + 10]); // left edge: pure red
        assert!((r - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-4, "red channel normalised: {r}");
        assert!((g - (0.0 - MEAN[1]) / STD[1]).abs() < 1e-4 && (b - (0.0 - MEAN[2]) / STD[2]).abs() < 1e-4);
    }

    #[test]
    fn argmax_takes_the_first_largest_and_ignores_nan() {
        assert_eq!(argmax(&[0.1, 0.9, 0.9, 0.2]), 1);
        assert_eq!(argmax(&[f32::NAN, 0.5, 0.4]), 1);
        assert_eq!(argmax(&[3.0]), 0);
    }

    #[test]
    fn gather_rows_picks_the_named_rows_in_order() {
        let h: Vec<f32> = (0..12).map(|v| v as f32).collect(); // 4 rows of 3
        assert_eq!(gather_rows(&h, 3, &[2, 0]), vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0]);
        assert!(gather_rows(&h, 3, &[]).is_empty());
    }

    #[test]
    fn raw_bytes_and_the_json_form_give_the_same_image() {
        let raw = b"\x89PNG-bytes".to_vec();
        assert_eq!(image_bytes(&raw).unwrap(), raw);
        let json = format!("{{\"image\":\"{}\"}}", base64::engine::general_purpose::STANDARD.encode(&raw));
        assert_eq!(image_bytes(json.as_bytes()).unwrap(), raw);
        assert!(image_bytes(b" {\"nope\":1}").is_err());
    }

    #[test]
    fn a_non_image_is_an_error_not_a_panic() {
        assert!(image_to_tensor(b"not an image").is_err());
    }
}

/// macOS: ONNX Runtime's global environment crashes in exit() ("libc++abi: terminating due to uncaught exception of type std::system_error: mutex lock failed"),
/// which turned every finished job into a failed one (MEASURED on an M5, ONNX Runtime 1.22.2). This is a one-shot and its output is written by now, so it leaves
/// through `_exit` and skips the static destructors.
fn main() {
    run_main();
    #[cfg(target_os = "macos")]
    {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: `_exit` only takes a status and never returns; stdout and stderr were flushed above.
        unsafe { libc::_exit(0) }
    }
}
