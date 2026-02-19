//! OCR Processor - Full Two-Stage Pipeline
//!
//! Detection (CRAFT/EasyOCR) + Recognition in one processor.
//! Uses ort crate to run the recognizer internally.
//!
//! Usage:
//!   cat image.png | ocr-processor preprocess
//!   cat tensors.json | ocr-processor postprocess

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

// Note: ndarray is used only by build deps; ort uses raw tensors

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Qualcomm EasyOCR detector expects 608x800 input
const DET_HEIGHT: u32 = 608;
const DET_WIDTH: u32 = 800;

/// Recognizer input size (asmud EasyOCR: grayscale, height=32, width=100)
const REC_HEIGHT: u32 = 32;
const REC_WIDTH: u32 = 100;

/// Detection thresholds
const TEXT_THRESHOLD: f32 = 0.3;
const LOW_TEXT: f32 = 0.2;
const MIN_AREA: u32 = 50;

/// Temp file for passing image between preprocess and postprocess
const TEMP_IMAGE_PATH: &str = "/tmp/ocr-processor-input.png";

/// Character set for English recognition (EasyOCR model - 95 chars)
static CHARS: OnceLock<Vec<char>> = OnceLock::new();

fn get_chars() -> &'static Vec<char> {
    CHARS.get_or_init(|| {
        // EasyOCR English character set (95 characters)
        // Index 0 = blank/CTC token
        // Order: blank, 0-9, special chars, uppercase A-Z, lowercase a-z
        let charset = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \u{20AC}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        charset.chars().collect()
    })
}

/// Path to recognizer ONNX model
fn recognizer_path() -> PathBuf {
    // Look in standard locations (asmud EasyOCR ONNX from HuggingFace)
    let paths = [
        PathBuf::from("/home/immortalb/.cache/huggingface/hub/models--asmud--EasyOCR-onnx/snapshots/4cb20758ed63725b7b57deb48b8e64b3217053b0/english_g2_jpqd.onnx"),
        dirs::data_local_dir()
            .map(|d| d.join("omnynet").join("models").join("ocr").join("recognizer.onnx"))
            .unwrap_or_default(),
    ];

    for p in &paths {
        if p.exists() {
            return p.clone();
        }
    }
    paths[0].clone() // Default
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: ocr-processor <preprocess|postprocess>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "preprocess" => {
            if let Err(e) = preprocess() {
                eprintln!("Preprocess error: {}", e);
                std::process::exit(1);
            }
        }
        "postprocess" => {
            if let Err(e) = postprocess() {
                eprintln!("Postprocess error: {}", e);
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown command: {}. Use 'preprocess' or 'postprocess'", args[1]);
            std::process::exit(1);
        }
    }
}

fn preprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read image from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Decode image
    let img = image::load_from_memory(&input)?;
    let (orig_w, orig_h) = img.dimensions();

    // Save original image for postprocess to use for cropping
    img.save(TEMP_IMAGE_PATH)?;

    // Resize to detector input size (608x800) preserving aspect ratio with padding
    let scale = f32::min(DET_HEIGHT as f32 / orig_h as f32, DET_WIDTH as f32 / orig_w as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    // Create padded image (black padding)
    let mut padded = image::RgbImage::new(DET_WIDTH, DET_HEIGHT);
    image::imageops::replace(&mut padded, &resized.to_rgb8(), 0, 0);
    let padded = DynamicImage::ImageRgb8(padded);

    // Convert to normalized tensor
    let tensor = image_to_normalized_tensor(&padded);

    // Output as JSON - only tensor data
    let mut output: HashMap<String, Vec<f32>> = HashMap::new();
    output.insert("image".to_string(), tensor);

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    Ok(())
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read tensor JSON from stdin (detector output)
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    let tensors: serde_json::Value = serde_json::from_slice(&input)?;

    // Get detector output
    let results = tensors.get("results")
        .or_else(|| tensors.get("output"))
        .ok_or("Missing 'results' tensor")?;

    // Parse heatmap
    let results_arr: Vec<f32> = if let Some(arr) = results.as_array() {
        flatten_to_f32(arr)
    } else {
        return Err("Invalid results format".into());
    };

    // Load original image for cropping
    let orig_img = image::open(TEMP_IMAGE_PATH)?;
    let (orig_w, orig_h) = orig_img.dimensions();

    // Detector output is [1, 304, 400, 2]
    let map_h = 304usize;
    let map_w = 400usize;

    // Extract text score map
    let mut text_map = vec![0.0f32; map_h * map_w];
    for y in 0..map_h {
        for x in 0..map_w {
            let idx = y * map_w * 2 + x * 2;
            if idx < results_arr.len() {
                text_map[y * map_w + x] = results_arr[idx];
            }
        }
    }

    // Find text boxes
    let boxes = extract_boxes(&text_map, map_w, map_h, (orig_h, orig_w));

    // Try to run recognition on each box
    let text_boxes = match run_recognition(&orig_img, &boxes) {
        Ok(recognized) => recognized,
        Err(e) => {
            eprintln!("Recognition failed (detection only): {}", e);
            // Fall back to detection-only results
            boxes.iter().map(|b| TextBoxResult {
                text: String::new(),
                confidence: b.score,
                polygon: vec![
                    [b.x1 as f32, b.y1 as f32],
                    [b.x2 as f32, b.y1 as f32],
                    [b.x2 as f32, b.y2 as f32],
                    [b.x1 as f32, b.y2 as f32],
                ],
            }).collect()
        }
    };

    // Build result
    let all_text: String = text_boxes.iter()
        .filter(|b| !b.text.is_empty())
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let boxes_json: Vec<serde_json::Value> = text_boxes.iter().map(|b| {
        serde_json::json!({
            "text": b.text,
            "confidence": b.confidence,
            "polygon": b.polygon
        })
    }).collect();

    let output = serde_json::json!({
        "type": "ocr",
        "text": if all_text.is_empty() {
            format!("{} text regions detected", text_boxes.len())
        } else {
            all_text
        },
        "boxes": boxes_json,
        "confidence": if text_boxes.is_empty() { 0.0 } else {
            text_boxes.iter().map(|b| b.confidence).sum::<f32>() / text_boxes.len() as f32
        }
    });

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    // Cleanup temp file
    let _ = std::fs::remove_file(TEMP_IMAGE_PATH);

    Ok(())
}

#[derive(Debug)]
struct TextBox {
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    score: f32,
}

#[derive(Debug)]
struct TextBoxResult {
    text: String,
    confidence: f32,
    polygon: Vec<[f32; 2]>,
}

/// Run recognition on detected text boxes
fn run_recognition(img: &DynamicImage, boxes: &[TextBox]) -> Result<Vec<TextBoxResult>, Box<dyn std::error::Error>> {
    use ort::session::{Session, builder::GraphOptimizationLevel};

    let rec_path = recognizer_path();
    if !rec_path.exists() {
        return Err(format!("Recognizer model not found: {:?}", rec_path).into());
    }

    // Load recognizer model
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&rec_path)?;

    let mut results = Vec::new();
    let chars = get_chars();

    for b in boxes {
        // Crop region from image
        let x1 = b.x1.min(img.width() - 1);
        let y1 = b.y1.min(img.height() - 1);
        let w = (b.x2 - b.x1).min(img.width() - x1);
        let h = (b.y2 - b.y1).min(img.height() - y1);

        if w < 5 || h < 5 {
            continue;
        }

        let crop = img.crop_imm(x1, y1, w, h);

        // Convert to grayscale and resize for recognizer
        let gray = crop.to_luma8();
        let resized = image::imageops::resize(&gray, REC_WIDTH, REC_HEIGHT, FilterType::Triangle);

        // Normalize to tensor [1, 1, 64, 200]
        let mut tensor_data = Vec::with_capacity((REC_HEIGHT * REC_WIDTH) as usize);
        for y in 0..REC_HEIGHT {
            for x in 0..REC_WIDTH {
                let pixel = resized.get_pixel(x, y);
                let value = pixel[0] as f32 / 255.0;
                // Simple normalization
                let normalized = (value - 0.5) / 0.5;
                tensor_data.push(normalized);
            }
        }

        // Create input tensor using ort 2.x API - shape tuple + data
        let shape = [1_i64, 1, REC_HEIGHT as i64, REC_WIDTH as i64];
        let input_tensor = ort::value::Tensor::<f32>::from_array((shape, tensor_data.into_boxed_slice()))?;

        // Run inference (input name is "input" for asmud EasyOCR model)
        let outputs = session.run(ort::inputs!["input" => input_tensor])?;

        // Get output: [1, seq_len, num_chars]
        // Use index-based access since the output name may vary
        let output = outputs.iter().next()
            .ok_or("No output from recognizer")?;
        let (output_name, output_value) = output;
        eprintln!("Recognizer output name: {}", output_name);

        // Extract tensor - ort 2.x returns (Shape, &[f32])
        let (output_shape, output_data) = output_value.try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = output_shape.iter().map(|&x| x as usize).collect();

        // CTC decode using raw data
        let text = ctc_decode_raw(output_data, &shape, chars);

        results.push(TextBoxResult {
            text,
            confidence: b.score,
            polygon: vec![
                [b.x1 as f32, b.y1 as f32],
                [b.x2 as f32, b.y1 as f32],
                [b.x2 as f32, b.y2 as f32],
                [b.x1 as f32, b.y2 as f32],
            ],
        });
    }

    Ok(results)
}

/// Simple CTC greedy decode using raw tensor data
fn ctc_decode_raw(data: &[f32], shape: &[usize], chars: &[char]) -> String {
    if shape.len() < 3 {
        return String::new();
    }

    let seq_len = shape[1];
    let num_chars = shape[2];

    let mut result = String::new();
    let mut prev_idx: Option<usize> = None;

    for t in 0..seq_len {
        // Find argmax for this timestep
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for c in 0..num_chars {
            // Index into [1, seq_len, num_chars] tensor
            let idx = t * num_chars + c;
            if idx < data.len() {
                let val = data[idx];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }
        }

        // CTC: skip blanks (idx 0) and repeated characters
        if max_idx != 0 && Some(max_idx) != prev_idx {
            if max_idx < chars.len() {
                result.push(chars[max_idx]);
            }
        }
        prev_idx = Some(max_idx);
    }

    result
}

/// Extract bounding boxes from text heatmap using connected components
fn extract_boxes(text_map: &[f32], width: usize, height: usize, orig_size: (u32, u32)) -> Vec<TextBox> {
    let mut boxes = Vec::new();
    let mut mask = vec![false; width * height];

    for i in 0..text_map.len() {
        mask[i] = text_map[i] > LOW_TEXT;
    }

    // Compute the same aspect-ratio-preserving scale used in preprocess.
    // The image was resized to fit within DET_HEIGHT x DET_WIDTH with padding,
    // so the heatmap (half the detector input) only covers the resized region,
    // not the full map dimensions.
    let (orig_h, orig_w) = orig_size;
    let scale = f32::min(DET_HEIGHT as f32 / orig_h as f32, DET_WIDTH as f32 / orig_w as f32);

    let mut visited = vec![false; width * height];

    for start_y in 0..height {
        for start_x in 0..width {
            let start_idx = start_y * width + start_x;
            if mask[start_idx] && !visited[start_idx] {
                let mut min_x = start_x;
                let mut max_x = start_x;
                let mut min_y = start_y;
                let mut max_y = start_y;
                let mut sum_score = 0.0f32;
                let mut count = 0u32;

                let mut stack = vec![(start_x, start_y)];
                while let Some((x, y)) = stack.pop() {
                    let idx = y * width + x;
                    if visited[idx] || !mask[idx] {
                        continue;
                    }
                    visited[idx] = true;

                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    sum_score += text_map[idx];
                    count += 1;

                    if x > 0 { stack.push((x - 1, y)); }
                    if x + 1 < width { stack.push((x + 1, y)); }
                    if y > 0 { stack.push((x, y - 1)); }
                    if y + 1 < height { stack.push((x, y + 1)); }
                }

                let area = (max_x - min_x + 1) * (max_y - min_y + 1);
                let avg_score = sum_score / count as f32;

                if area >= MIN_AREA as usize && avg_score > TEXT_THRESHOLD {
                    // Map heatmap coords back to original image.
                    // Heatmap is half the detector input, and the image was scaled by `scale`
                    // before padding, so: orig_coord = heatmap_coord * 2 / scale
                    let inv_scale = 2.0 / scale;

                    let x1 = (min_x as f32 * inv_scale) as u32;
                    let y1 = (min_y as f32 * inv_scale) as u32;
                    let x2 = ((max_x as f32 + 1.0) * inv_scale) as u32;
                    let y2 = ((max_y as f32 + 1.0) * inv_scale) as u32;

                    boxes.push(TextBox {
                        x1: x1.min(orig_w - 1),
                        y1: y1.min(orig_h - 1),
                        x2: x2.min(orig_w),
                        y2: y2.min(orig_h),
                        score: avg_score,
                    });
                }
            }
        }
    }

    boxes
}

/// Flatten nested JSON arrays to Vec<f32>
fn flatten_to_f32(arr: &[serde_json::Value]) -> Vec<f32> {
    let mut result = Vec::new();
    for item in arr {
        match item {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    result.push(f as f32);
                }
            }
            serde_json::Value::Array(nested) => {
                result.extend(flatten_to_f32(nested));
            }
            _ => {}
        }
    }
    result
}

/// Convert image to normalized CHW tensor
fn image_to_normalized_tensor(img: &DynamicImage) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut tensor = Vec::with_capacity((3 * width * height) as usize);

    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                let value = pixel[c] as f32 / 255.0;
                let normalized = (value - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                tensor.push(normalized);
            }
        }
    }

    tensor
}
