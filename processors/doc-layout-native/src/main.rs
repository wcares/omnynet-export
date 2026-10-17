//! Document layout processor (intake reader, gateway doc: docling-layout-heron, RT-DETRv2, Apache-2.0).
//!
//! Pre/post-processing around the `doc_layout` `.omny` model: a page image in, labelled boxes out.
//!
//! Usage:
//!   cat page.png                  | doc-layout-processor preprocess      -> (also accepts {"image":"base64..."}) {"pixel_values": [...], "pixel_values_shape": [1,3,640,640]}
//!   cat tensors.json              | doc-layout-processor postprocess     -> {"type":"detection","boxes":[{"x1","y1","x2","y2","confidence","class"}]}
//!
//! Matches the model's own preprocessor config: resize to exactly 640x640 (bilinear), rescale by 1/255, NO mean/std normalisation
//! (`do_normalize: false` in docling-layout-heron's preprocessor_config.json). Box coordinates are normalised (0..1) to the page image,
//! so the caller (which knows the page size) scales them.

use base64::Engine;
use image::imageops::FilterType;
use serde_json::{json, Value};
use std::io::{self, Read, Write};

/// The model takes a fixed 640x640 input.
const SIZE: u32 = 640;
/// Queries the model returns, and classes it scores (docling-layout-heron: `id2label` has 17 entries).
const NUM_QUERIES: usize = 300;
const NUM_CLASSES: usize = 17;
/// Minimum sigmoid score kept (Docling's layout predictor default).
const DEFAULT_THRESHOLD: f32 = 0.3;

/// `id2label` of docling-layout-heron, in id order.
const LABELS: [&str; NUM_CLASSES] = [
    "caption", "footnote", "formula", "list_item", "page_footer", "page_header", "picture", "section_header", "table", "text", "title",
    "document_index", "code", "checkbox_selected", "checkbox_unselected", "form", "key_value_region",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("preprocess") => preprocess(),
        Some("postprocess") => postprocess(),
        _ => {
            eprintln!("Usage: doc-layout-processor <preprocess|postprocess>");
            std::process::exit(1);
        }
    };
    if let Err(e) = result {
        eprintln!("doc-layout-processor error: {e}");
        std::process::exit(1);
    }
}

fn preprocess() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let bytes = image_bytes(&input)?;
    let (pixels, shape) = image_to_tensor(&bytes)?;
    let out = json!({ "pixel_values": pixels, "pixel_values_shape": shape });
    io::stdout().write_all(&serde_json::to_vec(&out)?)?;
    Ok(())
}

/// The page image from stdin: omny-compute's native-processor path writes the RAW image bytes (PNG/JPEG), like clip and ocr; a JSON object
/// `{"image": "<base64>"}` is accepted too (the dino form) so the command line from the README works.
fn image_bytes(input: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if input.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{') {
        let json: Value = serde_json::from_slice(input)?;
        let b64 = json.get("image").and_then(Value::as_str).ok_or("missing 'image' (base64 PNG/JPEG)")?;
        return Ok(base64::engine::general_purpose::STANDARD.decode(b64)?);
    }
    Ok(input.to_vec())
}

/// Decode, resize to 640x640 (bilinear / triangle filter, which scales its support when shrinking like PIL's), CHW float32 in 0..1.
fn image_to_tensor(bytes: &[u8]) -> Result<(Vec<f32>, [u32; 4]), Box<dyn std::error::Error>> {
    let img = image::load_from_memory(bytes)?;
    let rgb = img.resize_exact(SIZE, SIZE, FilterType::Triangle).to_rgb8();
    let mut t = Vec::with_capacity((3 * SIZE * SIZE) as usize);
    for c in 0..3 {
        for y in 0..SIZE {
            for x in 0..SIZE {
                t.push(rgb.get_pixel(x, y)[c] as f32 / 255.0);
            }
        }
    }
    Ok((t, [1, 3, SIZE, SIZE]))
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    let tensors: Value = serde_json::from_slice(&input)?;
    let floats = |name: &str| -> Option<Vec<f32>> {
        // JSON has null for NaN: read as a non-finite value the decoder skips.
        tensors.get(name)?.as_array().map(|a| a.iter().map(|v| v.as_f64().map(|x| x as f32).unwrap_or(f32::NAN)).collect())
    };
    let (Some(logits), Some(boxes)) = (floats("logits"), floats("pred_boxes")) else {
        return Err("missing 'logits' or 'pred_boxes'".into());
    };
    let out = json!({ "type": "detection", "boxes": decode(&logits, &boxes, DEFAULT_THRESHOLD) });
    io::stdout().write_all(&serde_json::to_vec(&out)?)?;
    Ok(())
}

/// RT-DETR post-processing as Hugging Face does it (`post_process_object_detection`): sigmoid over every (query, class) score, keep those at or
/// above `threshold`, one box per kept pair (a query can come out under two classes), best first. Boxes are cx,cy,w,h in 0..1; output is x1,y1,x2,y2 in 0..1
/// clamped to the page. Non-finite scores or boxes are skipped.
fn decode(logits: &[f32], boxes: &[f32], threshold: f32) -> Vec<Value> {
    let n = (logits.len() / NUM_CLASSES).min(boxes.len() / 4).min(NUM_QUERIES);
    let mut hits: Vec<(f32, usize, usize)> = Vec::new();
    for q in 0..n {
        for c in 0..NUM_CLASSES {
            let l = logits[q * NUM_CLASSES + c];
            if !l.is_finite() {
                continue;
            }
            let s = 1.0 / (1.0 + (-l).exp());
            if s >= threshold {
                hits.push((s, q, c));
            }
        }
    }
    hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    hits.into_iter()
        .filter_map(|(s, q, c)| {
            let (cx, cy, w, h) = (boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]);
            if ![cx, cy, w, h].iter().all(|v| v.is_finite()) {
                return None;
            }
            let clamp = |v: f32| v.clamp(0.0, 1.0);
            Some(json!({
                "x1": clamp(cx - w / 2.0), "y1": clamp(cy - h / 2.0), "x2": clamp(cx + w / 2.0), "y2": clamp(cy + h / 2.0),
                "confidence": s, "class": LABELS[c],
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A logits/boxes pair of NUM_QUERIES rows where every score is very low except the ones set by the test.
    fn empty() -> (Vec<f32>, Vec<f32>) {
        (vec![-20.0; NUM_QUERIES * NUM_CLASSES], vec![0.5; NUM_QUERIES * 4])
    }

    #[test]
    fn labels_are_the_models_seventeen_classes_in_id_order() {
        assert_eq!(LABELS.len(), NUM_CLASSES);
        assert_eq!((LABELS[0], LABELS[6], LABELS[8], LABELS[9], LABELS[16]), ("caption", "picture", "table", "text", "key_value_region"));
    }

    #[test]
    fn a_confident_query_becomes_one_labelled_xyxy_box() {
        let (mut l, mut b) = empty();
        l[5 * NUM_CLASSES + 8] = 4.0; // query 5, class 8 = table, sigmoid(4) = 0.982
        b[5 * 4..5 * 4 + 4].copy_from_slice(&[0.5, 0.4, 0.2, 0.1]);
        let d = decode(&l, &b, 0.3);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0]["class"], "table");
        let close = |k: &str, want: f64| assert!((d[0][k].as_f64().unwrap() - want).abs() < 1e-6, "{k}: {}", d[0][k]);
        close("x1", 0.4);
        close("y1", 0.35);
        close("x2", 0.6);
        close("y2", 0.45);
        assert!((d[0]["confidence"].as_f64().unwrap() - 0.98201).abs() < 1e-4);
    }

    #[test]
    fn below_the_threshold_is_dropped_and_the_rest_is_best_first() {
        let (mut l, b) = empty();
        l[0 * NUM_CLASSES + 9] = 0.0; // sigmoid 0.5
        l[1 * NUM_CLASSES + 9] = 2.0; // 0.88
        l[2 * NUM_CLASSES + 9] = -1.0; // 0.27 < 0.3
        let d = decode(&l, &b, 0.3);
        assert_eq!(d.len(), 2);
        assert!(d[0]["confidence"].as_f64().unwrap() > d[1]["confidence"].as_f64().unwrap());
    }

    #[test]
    fn a_query_can_come_out_under_two_classes_like_hugging_face_does() {
        let (mut l, b) = empty();
        l[3 * NUM_CLASSES + 9] = 3.0;
        l[3 * NUM_CLASSES + 3] = 2.0;
        let d = decode(&l, &b, 0.3);
        assert_eq!(d.iter().map(|x| x["class"].as_str().unwrap()).collect::<Vec<_>>(), vec!["text", "list_item"]);
    }

    #[test]
    fn boxes_are_clamped_to_the_page_and_nan_is_skipped() {
        let (mut l, mut b) = empty();
        l[0 * NUM_CLASSES + 9] = 5.0;
        b[0..4].copy_from_slice(&[0.95, 0.5, 0.3, 0.2]); // x2 would be 1.1
        l[1 * NUM_CLASSES + 9] = 5.0;
        b[4] = f32::NAN;
        l[2 * NUM_CLASSES + 9] = f32::NAN;
        let d = decode(&l, &b, 0.3);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0]["x2"].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn the_tensor_is_chw_in_zero_to_one_and_not_normalised() {
        let mut img = image::RgbImage::new(4, 2); // R in the left half, B in the right
        for y in 0..2 {
            for x in 0..4 {
                img.put_pixel(x, y, if x < 2 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) });
            }
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(&mut io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let (t, shape) = image_to_tensor(&png).unwrap();
        assert_eq!(shape, [1, 3, 640, 640]);
        assert_eq!(t.len(), 3 * 640 * 640);
        assert!(t.iter().all(|v| (0.0..=1.0).contains(v)), "no mean/std normalisation: values stay in 0..1");
        let (r, g, b) = (&t[..640 * 640], &t[640 * 640..2 * 640 * 640], &t[2 * 640 * 640..]);
        assert!(r[10] > 0.99 && b[10] < 0.01 && g[10] < 0.01, "left edge is red (channel 0 first)");
        assert!(b[640 - 10] > 0.99 && r[640 - 10] < 0.01, "right edge is blue");
    }

    #[test]
    fn raw_bytes_and_the_json_form_give_the_same_image() {
        let raw = b"\x89PNG-bytes".to_vec();
        assert_eq!(image_bytes(&raw).unwrap(), raw, "raw bytes pass through (the omny-compute path)");
        let json = format!("{{\"image\":\"{}\"}}", base64::engine::general_purpose::STANDARD.encode(&raw));
        assert_eq!(image_bytes(json.as_bytes()).unwrap(), raw);
        assert!(image_bytes(b"  {\"nope\":1}").is_err(), "JSON without an image is an error");
    }

    #[test]
    fn a_non_image_is_an_error_not_a_panic() {
        assert!(image_to_tensor(b"not an image").is_err());
    }
}
