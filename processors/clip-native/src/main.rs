//! CLIP Processor - Native Executable
//!
//! Reads image bytes from stdin, writes tensor JSON to stdout.
//!
//! Usage:
//!   cat image.png | clip-processor preprocess
//!   cat tensors.json | clip-processor postprocess

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// CLIP normalization constants
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const TARGET_SIZE: u32 = 224;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: clip-processor <preprocess|postprocess>");
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

    // Resize shortest side to 224, center crop to 224x224
    let img = resize_shortest_center_crop(&img, TARGET_SIZE);

    // Convert to normalized tensor
    let tensor = image_to_normalized_tensor(&img);

    // Output as JSON
    let mut output: HashMap<String, Vec<f32>> = HashMap::new();
    output.insert("pixel_values".to_string(), tensor);

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    Ok(())
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read tensor JSON from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Parse tensors
    let tensors: HashMap<String, Vec<f32>> = serde_json::from_slice(&input)?;

    // Get embedding
    let embedding = tensors.get("image_embeds")
        .or_else(|| tensors.get("image_features"))
        .or_else(|| tensors.get("pooled_output"))
        .or_else(|| tensors.values().next())
        .cloned()
        .unwrap_or_default();

    // Output as embedding result (matching EmbeddingResult struct)
    let dimension = embedding.len();
    let output = serde_json::json!({
        "type": "embedding",
        "vector": embedding,
        "dimension": dimension
    });

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    Ok(())
}

/// Resize image so shortest side = target_size, then center crop
fn resize_shortest_center_crop(img: &DynamicImage, target_size: u32) -> DynamicImage {
    let (orig_w, orig_h) = img.dimensions();

    // Calculate scale to make shortest side = target_size
    let scale = if orig_w < orig_h {
        target_size as f32 / orig_w as f32
    } else {
        target_size as f32 / orig_h as f32
    };

    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    // Resize with bicubic (CatmullRom)
    let resized = img.resize_exact(new_w, new_h, FilterType::CatmullRom);

    // Center crop
    let x_offset = (new_w.saturating_sub(target_size)) / 2;
    let y_offset = (new_h.saturating_sub(target_size)) / 2;

    resized.crop_imm(x_offset, y_offset, target_size, target_size)
}

/// Convert image to normalized CHW tensor
fn image_to_normalized_tensor(img: &DynamicImage) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut tensor = Vec::with_capacity((3 * width * height) as usize);

    // CHW format with normalization
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                let value = pixel[c] as f32 / 255.0;
                let normalized = (value - CLIP_MEAN[c]) / CLIP_STD[c];
                tensor.push(normalized);
            }
        }
    }

    tensor
}
