//! Semantic Text Embedding Processor - Native Executable
//!
//! Tokenizes text input for MiniLM-L6-v2 sentence transformer.
//!
//! Usage:
//!   echo '{"text": "a photo of a cat"}' | semantic-processor preprocess
//!   cat tensors.json | semantic-processor postprocess

use serde::Deserialize;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use tokenizers::Tokenizer;

/// Max sequence length for MiniLM-L6-v2
const MAX_LENGTH: usize = 256;

/// Embedded tokenizer JSON (from sentence-transformers/all-MiniLM-L6-v2)
const TOKENIZER_JSON: &str = include_str!("../tokenizer.json");

#[derive(Deserialize)]
struct TextInput {
    text: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: semantic-processor <preprocess|postprocess>");
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
            eprintln!(
                "Unknown command: {}. Use 'preprocess' or 'postprocess'",
                args[1]
            );
            std::process::exit(1);
        }
    }
}

fn preprocess() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Parse as JSON with text field, or use raw string
    let text = if let Ok(json_input) = serde_json::from_slice::<TextInput>(&input) {
        json_input.text
    } else {
        String::from_utf8_lossy(&input).trim().to_string()
    };

    // Load tokenizer
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes())
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Tokenize
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| format!("Tokenization failed: {}", e))?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention: Vec<u32> = encoding.get_attention_mask().to_vec();
    let type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

    // Truncate to max length
    let len = token_ids.len().min(MAX_LENGTH);

    // Pad to MAX_LENGTH
    let mut input_ids: Vec<i64> = vec![0; MAX_LENGTH];
    let mut attention_mask: Vec<i64> = vec![0; MAX_LENGTH];
    let mut token_type_ids: Vec<i64> = vec![0; MAX_LENGTH];

    for i in 0..len {
        input_ids[i] = token_ids[i] as i64;
        attention_mask[i] = attention[i] as i64;
        token_type_ids[i] = type_ids[i] as i64;
    }

    // Output as JSON with shape hints for [1, MAX_LENGTH] (batch=1, seq=MAX_LENGTH)
    let shape = vec![1u64, MAX_LENGTH as u64];
    let output = serde_json::json!({
        "input_ids": input_ids,
        "input_ids_shape": shape,
        "attention_mask": attention_mask,
        "attention_mask_shape": shape,
        "token_type_ids": token_type_ids,
        "token_type_ids_shape": shape,
    });

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    Ok(())
}

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    let tensors: HashMap<String, Vec<f32>> = serde_json::from_slice(&input)?;

    // Extract sentence embedding
    let embedding = tensors
        .get("sentence_embedding")
        .or_else(|| tensors.get("pooler_output"))
        .or_else(|| tensors.values().next())
        .cloned()
        .unwrap_or_default();

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
