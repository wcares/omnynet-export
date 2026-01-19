//! SAM2 Processor - Native Executable
//!
//! Reads image bytes from stdin, writes tensor JSON to stdout.
//! Used for Segment Anything Model 2.
//!
//! Usage:
//!   cat image.png | sam2-processor preprocess
//!   cat tensors.json | sam2-processor postprocess

use image::{DynamicImage, GenericImageView, imageops::FilterType, Rgba, RgbaImage};
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// SAM2 normalization constants (ImageNet)
const SAM_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const SAM_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SAM2 target size
const TARGET_SIZE: u32 = 1024;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: sam2-processor <preprocess|postprocess>");
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

    // Resize longest side to TARGET_SIZE, pad to square
    let scale = TARGET_SIZE as f32 / orig_w.max(orig_h) as f32;
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    // Resize with bilinear
    let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

    // Create padded square image (black padding)
    let mut padded = RgbaImage::from_pixel(TARGET_SIZE, TARGET_SIZE, Rgba([0, 0, 0, 255]));
    image::imageops::replace(&mut padded, &resized.to_rgba8(), 0, 0);
    let padded = DynamicImage::ImageRgba8(padded);

    // Convert to normalized tensor
    let tensor = image_to_normalized_tensor(&padded);

    // Output as JSON with scale info for postprocessing
    let mut output: HashMap<String, serde_json::Value> = HashMap::new();
    output.insert("image".to_string(), serde_json::json!(tensor));
    output.insert("_shape".to_string(), serde_json::json!([1, 3, TARGET_SIZE, TARGET_SIZE]));
    output.insert("_orig_size".to_string(), serde_json::json!([orig_h, orig_w]));
    output.insert("_scale".to_string(), serde_json::json!([scale]));

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    Ok(())
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read tensor JSON from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Parse tensors
    let tensors: serde_json::Value = serde_json::from_slice(&input)?;

    // Extract output based on model type
    let output = if let Some(masks) = tensors.get("masks").or_else(|| tensors.get("low_res_masks")) {
        // Full decoder output with masks
        serde_json::json!({
            "type": "segmentation",
            "masks": masks
        })
    } else if tensors.get("image_embed").is_some() || tensors.get("image_embeddings").is_some() {
        // SAM2 encoder output - use "json" type for complex multi-tensor output
        serde_json::json!({
            "type": "json",
            "output_type": "sam2_encoder",
            "image_embed": tensors.get("image_embed"),
            "high_res_feats_0": tensors.get("high_res_feats_0"),
            "high_res_feats_1": tensors.get("high_res_feats_1")
        })
    } else {
        // Return raw tensors if structure is unknown
        serde_json::json!({
            "type": "raw",
            "tensors": tensors
        })
    };

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    Ok(())
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
                let normalized = (value - SAM_MEAN[c]) / SAM_STD[c];
                tensor.push(normalized);
            }
        }
    }

    tensor
}
