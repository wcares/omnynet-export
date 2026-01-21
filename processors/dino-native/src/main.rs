//! DINO Processor - Native Executable with BERT Tokenization
//!
//! Reads image + text prompt from stdin, outputs preprocessed tensors for Grounding DINO.
//! Handles both image preprocessing and BERT text tokenization.
//!
//! Usage:
//!   echo '{"image":"base64...","text":"a cat. a dog."}' | dino-processor preprocess
//!   cat tensors.json | dino-processor postprocess

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// DINO image size - fixed 800x800 as expected by the model
const DINO_IMAGE_SIZE: u32 = 800;

/// BERT special token IDs
const CLS_TOKEN_ID: i64 = 101;
const SEP_TOKEN_ID: i64 = 102;
const PAD_TOKEN_ID: i64 = 0;
const UNK_TOKEN_ID: i64 = 100;
const MAX_TEXT_LEN: usize = 256;

/// Embedded BERT vocabulary (loaded at compile time)
const BERT_VOCAB: &str = include_str!("bert_vocab.txt");

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: dino-processor <preprocess|postprocess>");
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

/// Build vocabulary lookup from embedded vocab
fn build_vocab() -> HashMap<String, i64> {
    BERT_VOCAB
        .lines()
        .enumerate()
        .map(|(i, word)| (word.to_string(), i as i64))
        .collect()
}

/// Simple BERT tokenizer (WordPiece)
fn tokenize_text(text: &str, vocab: &HashMap<String, i64>) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let text = text.to_lowercase();

    // Basic tokenization: split on whitespace and punctuation
    let mut tokens = Vec::new();
    let mut current_word = String::new();

    for c in text.chars() {
        if c.is_whitespace() || c.is_ascii_punctuation() {
            if !current_word.is_empty() {
                tokens.push(current_word.clone());
                current_word.clear();
            }
            if c.is_ascii_punctuation() {
                tokens.push(c.to_string());
            }
        } else {
            current_word.push(c);
        }
    }
    if !current_word.is_empty() {
        tokens.push(current_word);
    }

    // WordPiece tokenization
    let mut input_ids = vec![CLS_TOKEN_ID];

    for token in tokens {
        let mut start = 0;
        let chars: Vec<char> = token.chars().collect();

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let subword: String = if start > 0 {
                    format!("##{}", chars[start..end].iter().collect::<String>())
                } else {
                    chars[start..end].iter().collect()
                };

                if let Some(&id) = vocab.get(&subword) {
                    input_ids.push(id);
                    found = true;
                    start = end;
                    break;
                }
                end -= 1;
            }

            if !found {
                input_ids.push(UNK_TOKEN_ID);
                start += 1;
            }
        }
    }

    input_ids.push(SEP_TOKEN_ID);

    // Truncate if exceeds max length (but don't pad - model accepts variable length)
    let len = input_ids.len().min(MAX_TEXT_LEN);
    input_ids.truncate(len);

    // Create attention mask and token type ids (same length as input_ids, no padding)
    let attention_mask: Vec<i64> = vec![1; len];
    let token_type_ids: Vec<i64> = vec![0; len];

    (input_ids, attention_mask, token_type_ids)
}

fn preprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Build vocab
    let vocab = build_vocab();

    // Read input from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Parse JSON input
    let json: serde_json::Value = serde_json::from_slice(&input)?;

    // Extract image
    let image_b64 = json.get("image").and_then(|v| v.as_str()).unwrap_or("");
    let image_bytes = if !image_b64.is_empty() {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(image_b64)?
    } else {
        return Err("No image provided".into());
    };

    // Extract text prompt
    let text_prompt = json.get("prompt")
        .or_else(|| json.get("text"))
        .and_then(|v| v.as_str())
        .unwrap_or("object");

    // Decode and preprocess image
    let img = image::load_from_memory(&image_bytes)?;

    // Resize to exactly 800x800 as required by the model
    let resized = img.resize_exact(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE, FilterType::Triangle);

    // Convert to normalized tensor
    let pixel_values = image_to_normalized_tensor(&resized);

    // Create pixel mask (all ones for valid pixels)
    let pixel_mask: Vec<i64> = vec![1; (DINO_IMAGE_SIZE * DINO_IMAGE_SIZE) as usize];

    // Tokenize text
    let (input_ids, attention_mask, token_type_ids) = tokenize_text(text_prompt, &vocab);

    // Output as JSON (flat arrays only - no metadata objects)
    let output = serde_json::json!({
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_mask": pixel_mask
    });

    let json_bytes = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json_bytes)?;

    Ok(())
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read tensor JSON from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Parse tensors - format is HashMap<String, Vec<Value>> (flat arrays, may have nulls for NaN)
    let tensors: HashMap<String, Vec<serde_json::Value>> = serde_json::from_slice(&input)?;

    // DINO outputs:
    // - pred_boxes: [1, 900, 4] = 3600 values (cx, cy, w, h normalized)
    // - logits: [1, 900, 256] = 230400 values
    let boxes_flat = tensors.get("pred_boxes");
    let logits_flat = tensors.get("logits");

    let mut detection_boxes: Vec<serde_json::Value> = Vec::new();

    // Confidence threshold
    let confidence_threshold = 0.25;

    // Helper to extract f64 from Value (handles nulls from NaN)
    let to_f64 = |v: &serde_json::Value| -> f64 {
        v.as_f64().unwrap_or(f64::NEG_INFINITY)
    };

    if let (Some(boxes), Some(logits)) = (boxes_flat, logits_flat) {
        // boxes shape: [1, 900, 4] -> 900 boxes, 4 values each
        // logits shape: [1, 900, 256] -> 900 queries, 256 classes each
        let num_boxes = 900;
        let num_classes = 256;

        for i in 0..num_boxes {
            // Get logits for this box [i * 256 .. (i+1) * 256]
            let logit_start = i * num_classes;
            let logit_end = logit_start + num_classes;

            if logit_end > logits.len() {
                break;
            }

            // Find max logit
            let max_logit = logits[logit_start..logit_end]
                .iter()
                .map(to_f64)
                .fold(f64::NEG_INFINITY, f64::max);

            // Sigmoid
            let confidence = 1.0 / (1.0 + (-max_logit).exp());

            if confidence > confidence_threshold {
                // Get box coordinates [i * 4 .. (i+1) * 4]
                let box_start = i * 4;
                if box_start + 4 <= boxes.len() {
                    let cx = to_f64(&boxes[box_start]) as f32;
                    let cy = to_f64(&boxes[box_start + 1]) as f32;
                    let w = to_f64(&boxes[box_start + 2]) as f32;
                    let h = to_f64(&boxes[box_start + 3]) as f32;

                    // Skip invalid boxes
                    if cx.is_nan() || cy.is_nan() || w.is_nan() || h.is_nan() {
                        continue;
                    }

                    // Convert cxcywh to xyxy (normalized 0-1)
                    let x1 = cx - w / 2.0;
                    let y1 = cy - h / 2.0;
                    let x2 = cx + w / 2.0;
                    let y2 = cy + h / 2.0;

                    detection_boxes.push(serde_json::json!({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence as f32
                    }));
                }
            }
        }
    }

    // Convert to ProcessorOutput Detection format expected by omny-compute
    // Format: {"type": "detection", "boxes": [{"x1":..., "y1":..., "x2":..., "y2":..., "confidence":..., "class":"..."}]}
    let output = serde_json::json!({
        "type": "detection",
        "boxes": detection_boxes
    });

    let json_bytes = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json_bytes)?;

    Ok(())
}

/// Convert image to normalized CHW tensor
fn image_to_normalized_tensor(img: &DynamicImage) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut tensor = Vec::with_capacity((3 * width * height) as usize);

    // CHW format with ImageNet normalization
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
