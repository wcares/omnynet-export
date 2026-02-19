//! PaddleOCR Processor - PP-OCRv5 Pipeline
//!
//! Two-stage OCR: Detection + Recognition
//! Runs recognition internally during postprocess.
//!
//! Usage:
//!   cat image.png | paddleocr-processor preprocess
//!   cat det_output.json | paddleocr-processor postprocess

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

/// PaddleOCR normalization: (x/255 - 0.5) / 0.5
const NORM_MEAN: f32 = 0.5;
const NORM_STD: f32 = 0.5;

/// Detection parameters
const DET_LIMIT_SIDE: u32 = 960;
const DET_THRESH: f32 = 0.3;
const BOX_THRESH: f32 = 0.5;

/// Recognition parameters
const REC_HEIGHT: u32 = 48;
const REC_WIDTH: u32 = 320;

/// Temp file for passing image between preprocess and postprocess
const TEMP_IMAGE_PATH: &str = "/tmp/paddleocr-processor-input.png";

/// Character dictionary (loaded lazily)
static CHARS: OnceLock<Vec<String>> = OnceLock::new();

fn get_chars() -> &'static Vec<String> {
    CHARS.get_or_init(|| {
        // Try to load from standard locations
        let paths = [
            PathBuf::from("/tmp/ppocr_keys.txt"),
            dirs::data_local_dir()
                .map(|d| d.join("omnynet").join("paddleocr").join("ppocr_keys.txt"))
                .unwrap_or_default(),
        ];

        for path in &paths {
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(path) {
                    let mut chars: Vec<String> = vec!["".to_string()]; // blank at index 0
                    for line in content.lines() {
                        chars.push(line.to_string());
                    }
                    eprintln!("[paddleocr] Loaded {} characters from {:?}", chars.len(), path);
                    return chars;
                }
            }
        }

        eprintln!("[paddleocr] Warning: character dictionary not found, using ASCII fallback");
        // Fallback to basic ASCII
        let mut chars = vec!["".to_string()];
        for c in ' '..='~' {
            chars.push(c.to_string());
        }
        chars
    })
}

/// Path to recognition ONNX model
fn recognizer_path() -> PathBuf {
    let paths = [
        PathBuf::from("/tmp/ppocr_rec.onnx"),
        dirs::data_local_dir()
            .map(|d| d.join("omnynet").join("paddleocr").join("models").join("rec.onnx"))
            .unwrap_or_default(),
    ];

    for p in &paths {
        if p.exists() {
            return p.clone();
        }
    }
    paths[0].clone()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: paddleocr-processor <preprocess|postprocess>");
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

    eprintln!("[paddleocr] Preprocess received {} bytes", input.len());

    // Decode image
    let img = image::load_from_memory(&input)?;
    let (orig_w, orig_h) = img.dimensions();

    eprintln!("[paddleocr] Original image: {}x{}", orig_w, orig_h);

    // Save for postprocess
    img.save(TEMP_IMAGE_PATH)?;

    // Calculate resize dimensions
    let ratio = if orig_w.max(orig_h) > DET_LIMIT_SIDE {
        DET_LIMIT_SIDE as f32 / orig_w.max(orig_h) as f32
    } else {
        1.0
    };

    let mut new_h = (orig_h as f32 * ratio) as u32;
    let mut new_w = (orig_w as f32 * ratio) as u32;

    // Round to multiple of 32
    new_h = (new_h / 32).max(1) * 32;
    new_w = (new_w / 32).max(1) * 32;

    eprintln!("[paddleocr] Resized to: {}x{}", new_w, new_h);

    // Resize
    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    // Convert to tensor with PaddleOCR normalization
    let tensor = image_to_paddle_tensor(&resized);

    eprintln!("[paddleocr] Tensor size: {} (expected {})", tensor.len(), 3 * new_w * new_h);

    // Debug: check tensor stats
    let min_val = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("[paddleocr] Tensor range: [{:.4}, {:.4}]", min_val, max_val);

    // Output as JSON with shape information
    // Format: {"x": [data], "x_shape": [1, 3, H, W]}
    let output = serde_json::json!({
        "x": tensor,
        "x_shape": [1, 3, new_h, new_w]
    });

    let json = serde_json::to_vec(&output)?;
    eprintln!("[paddleocr] Output JSON size: {} bytes", json.len());
    io::stdout().write_all(&json)?;

    Ok(())
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read detection output from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Debug: save raw input to file
    std::fs::write("/tmp/postprocess_input.json", &input)?;
    eprintln!("[paddleocr] Postprocess input size: {} bytes", input.len());

    let tensors: serde_json::Value = serde_json::from_slice(&input)?;
    eprintln!("[paddleocr] Parsed JSON keys: {:?}", tensors.as_object().map(|o| o.keys().collect::<Vec<_>>()));

    // Get detection output - omny-compute sends flat arrays as {"tensor_name": [values]}
    // Look for any key that might contain the detection output
    let det_arr: Vec<f32> = if let Some(obj) = tensors.as_object() {
        // Get the first (and usually only) tensor output
        obj.values()
            .next()
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .ok_or("No tensor data found")?
    } else {
        return Err("Invalid output format - expected JSON object".into());
    };

    eprintln!("[paddleocr] Detection array size: {}", det_arr.len());
    if !det_arr.is_empty() {
        let min_val = det_arr.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = det_arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let above_thresh = det_arr.iter().filter(|&&v| v > DET_THRESH).count();
        eprintln!("[paddleocr] Detection values: min={:.4}, max={:.4}, above_thresh={}", min_val, max_val, above_thresh);
    }

    // Load original image to get dimensions for shape inference
    let orig_img = image::open(TEMP_IMAGE_PATH)?;
    let (orig_w, orig_h) = orig_img.dimensions();

    // Calculate detection map dimensions based on preprocessing
    // Detection input was resized to max 960 and rounded to multiple of 32
    let ratio = if orig_w.max(orig_h) > DET_LIMIT_SIDE {
        DET_LIMIT_SIDE as f32 / orig_w.max(orig_h) as f32
    } else {
        1.0
    };
    let det_h = ((orig_h as f32 * ratio) as u32 / 32).max(1) * 32;
    let det_w = ((orig_w as f32 * ratio) as u32 / 32).max(1) * 32;

    // Verify the array size matches expected shape [1, 1, det_h, det_w]
    let expected_size = (det_h * det_w) as usize;
    if det_arr.len() != expected_size {
        return Err(format!(
            "Detection output size mismatch: got {}, expected {} ({}x{})",
            det_arr.len(), expected_size, det_h, det_w
        ).into());
    }

    // Extract probability map (the detection array IS the probability map)
    let prob_map: Vec<f32> = det_arr;

    // Find text boxes
    let boxes = extract_boxes(&prob_map, det_w as usize, det_h as usize, orig_w, orig_h);

    // Run recognition on each box
    let text_boxes = match run_recognition(&orig_img, &boxes) {
        Ok(results) => results,
        Err(e) => {
            eprintln!("[paddleocr] Recognition failed: {}", e);
            // Return detection-only results
            boxes.iter().map(|b| TextBoxResult {
                text: String::new(),
                confidence: b.score,
                polygon: box_to_polygon(b),
            }).collect()
        }
    };

    // Build output
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

    let avg_conf = if text_boxes.is_empty() {
        0.0
    } else {
        text_boxes.iter().map(|b| b.confidence).sum::<f32>() / text_boxes.len() as f32
    };

    let output = serde_json::json!({
        "type": "ocr",
        "text": all_text,
        "boxes": boxes_json,
        "confidence": avg_conf
    });

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    // Cleanup
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

fn box_to_polygon(b: &TextBox) -> Vec<[f32; 2]> {
    vec![
        [b.x1 as f32, b.y1 as f32],
        [b.x2 as f32, b.y1 as f32],
        [b.x2 as f32, b.y2 as f32],
        [b.x1 as f32, b.y2 as f32],
    ]
}

/// Run recognition on detected boxes
fn run_recognition(img: &DynamicImage, boxes: &[TextBox]) -> Result<Vec<TextBoxResult>, Box<dyn std::error::Error>> {
    use ort::session::{Session, builder::GraphOptimizationLevel};

    let rec_path = recognizer_path();
    if !rec_path.exists() {
        return Err(format!("Recognition model not found: {:?}", rec_path).into());
    }

    eprintln!("[paddleocr] Loading recognizer from {:?}", rec_path);

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&rec_path)?;

    let chars = get_chars();
    let mut results = Vec::new();

    for b in boxes {
        // Crop region
        let x1 = b.x1.min(img.width().saturating_sub(1));
        let y1 = b.y1.min(img.height().saturating_sub(1));
        let w = (b.x2.saturating_sub(b.x1)).min(img.width().saturating_sub(x1));
        let h = (b.y2.saturating_sub(b.y1)).min(img.height().saturating_sub(y1));

        if w < 3 || h < 3 {
            continue;
        }

        let crop = img.crop_imm(x1, y1, w, h);

        // Preprocess for recognition (PaddleOCR Chinese style)
        let tensor_data = preprocess_rec_crop(&crop);

        // Create input tensor
        let target_w = calculate_rec_width(w, h);
        let shape = [1_i64, 3, REC_HEIGHT as i64, target_w as i64];
        let input_tensor = ort::value::Tensor::<f32>::from_array((shape, tensor_data.into_boxed_slice()))?;

        // Run inference
        let outputs = session.run(ort::inputs!["x" => input_tensor])?;

        // Get output
        let output = outputs.iter().next()
            .ok_or("No output from recognizer")?;
        let (_, output_value) = output;

        let (output_shape, output_data) = output_value.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<usize> = output_shape.iter().map(|&x| x as usize).collect();

        // CTC decode
        let text = ctc_decode(output_data, &shape_vec, chars);

        results.push(TextBoxResult {
            text,
            confidence: b.score,
            polygon: box_to_polygon(b),
        });
    }

    Ok(results)
}

/// Calculate target width for recognition based on aspect ratio
fn calculate_rec_width(w: u32, h: u32) -> u32 {
    let ratio = w as f32 / h as f32;
    let max_ratio = REC_WIDTH as f32 / REC_HEIGHT as f32;
    let target_ratio = ratio.max(max_ratio);
    let target_w = (REC_HEIGHT as f32 * target_ratio).ceil() as u32;
    target_w.max(REC_WIDTH)
}

/// Preprocess crop for recognition
fn preprocess_rec_crop(img: &DynamicImage) -> Vec<f32> {
    let (w, h) = img.dimensions();

    // Calculate target dimensions
    let ratio = w as f32 / h as f32;
    let max_ratio = REC_WIDTH as f32 / REC_HEIGHT as f32;
    let target_ratio = ratio.max(max_ratio);
    let target_w = (REC_HEIGHT as f32 * target_ratio).ceil() as u32;

    let resized_w = if (REC_HEIGHT as f32 * ratio).ceil() as u32 > target_w {
        target_w
    } else {
        (REC_HEIGHT as f32 * ratio).ceil() as u32
    };

    // Resize
    let resized = img.resize_exact(resized_w, REC_HEIGHT, FilterType::Triangle);

    // Convert to tensor with padding
    let rgb = resized.to_rgb8();
    let mut tensor = vec![0.0f32; (3 * REC_HEIGHT * target_w) as usize];

    // Fill with normalized values (BGR order for PaddleOCR)
    for c in 0..3 {
        let channel_idx = 2 - c; // BGR -> RGB index mapping
        for y in 0..REC_HEIGHT {
            for x in 0..resized_w {
                let pixel = rgb.get_pixel(x, y);
                let value = pixel[channel_idx] as f32 / 255.0;
                let normalized = (value - NORM_MEAN) / NORM_STD;
                let idx = c * (REC_HEIGHT * target_w) as usize
                    + y as usize * target_w as usize
                    + x as usize;
                tensor[idx] = normalized;
            }
        }
    }

    tensor
}

/// CTC greedy decode
fn ctc_decode(data: &[f32], shape: &[usize], chars: &[String]) -> String {
    if shape.len() < 3 {
        return String::new();
    }

    let seq_len = shape[1];
    let num_chars = shape[2];

    let mut result = String::new();
    let mut prev_idx: Option<usize> = None;

    for t in 0..seq_len {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for c in 0..num_chars {
            let idx = t * num_chars + c;
            if idx < data.len() {
                let val = data[idx];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }
        }

        // Skip blank (0) and repeats
        if max_idx != 0 && Some(max_idx) != prev_idx {
            if max_idx < chars.len() {
                result.push_str(&chars[max_idx]);
            }
        }
        prev_idx = Some(max_idx);
    }

    result
}

/// Extract bounding boxes from probability map
fn extract_boxes(prob_map: &[f32], width: usize, height: usize, orig_w: u32, orig_h: u32) -> Vec<TextBox> {
    let mut boxes = Vec::new();
    let mut visited = vec![false; width * height];

    let scale_x = orig_w as f32 / width as f32;
    let scale_y = orig_h as f32 / height as f32;

    for start_y in 0..height {
        for start_x in 0..width {
            let start_idx = start_y * width + start_x;

            if prob_map[start_idx] <= DET_THRESH || visited[start_idx] {
                continue;
            }

            // Flood fill to find connected component
            let mut min_x = start_x;
            let mut max_x = start_x;
            let mut min_y = start_y;
            let mut max_y = start_y;
            let mut sum_score = 0.0f32;
            let mut count = 0u32;

            let mut stack = vec![(start_x, start_y)];

            while let Some((x, y)) = stack.pop() {
                let idx = y * width + x;

                if visited[idx] || prob_map[idx] <= DET_THRESH {
                    continue;
                }

                visited[idx] = true;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                sum_score += prob_map[idx];
                count += 1;

                // Check neighbors
                if x > 0 { stack.push((x - 1, y)); }
                if x + 1 < width { stack.push((x + 1, y)); }
                if y > 0 { stack.push((x, y - 1)); }
                if y + 1 < height { stack.push((x, y + 1)); }
            }

            let avg_score = sum_score / count.max(1) as f32;
            let box_w = max_x - min_x + 1;
            let box_h = max_y - min_y + 1;

            if box_w < 3 || box_h < 2 || avg_score < BOX_THRESH {
                continue;
            }

            // Scale to original image coordinates
            let x1 = (min_x as f32 * scale_x) as u32;
            let y1 = (min_y as f32 * scale_y) as u32;
            let x2 = ((max_x + 1) as f32 * scale_x).min(orig_w as f32) as u32;
            let y2 = ((max_y + 1) as f32 * scale_y).min(orig_h as f32) as u32;

            boxes.push(TextBox {
                x1,
                y1,
                x2,
                y2,
                score: avg_score,
            });
        }
    }

    // Sort by y then x
    boxes.sort_by(|a, b| {
        let y_cmp = (a.y1 / 20).cmp(&(b.y1 / 20));
        if y_cmp == std::cmp::Ordering::Equal {
            a.x1.cmp(&b.x1)
        } else {
            y_cmp
        }
    });

    boxes
}

/// Convert image to PaddleOCR tensor format (CHW, BGR, normalized)
fn image_to_paddle_tensor(img: &DynamicImage) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut tensor = Vec::with_capacity((3 * width * height) as usize);

    // BGR order for PaddleOCR
    for c in 0..3 {
        let channel_idx = 2 - c; // BGR mapping
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                let value = pixel[channel_idx] as f32 / 255.0;
                let normalized = (value - NORM_MEAN) / NORM_STD;
                tensor.push(normalized);
            }
        }
    }

    tensor
}

/// Flatten nested JSON arrays to Vec<f32>
fn flatten_to_f32(value: &serde_json::Value) -> Vec<f32> {
    let mut result = Vec::new();

    match value {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                result.push(f as f32);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                result.extend(flatten_to_f32(item));
            }
        }
        _ => {}
    }

    result
}

/// Infer shape from nested JSON arrays
fn infer_shape(value: &serde_json::Value) -> Vec<usize> {
    let mut shape = Vec::new();
    let mut current = value;

    while let serde_json::Value::Array(arr) = current {
        shape.push(arr.len());
        if arr.is_empty() {
            break;
        }
        current = &arr[0];
    }

    shape
}
