//! CLIP Text Processor - Native Executable
//!
//! Tokenizes text input for CLIP text encoder.
//!
//! Usage:
//!   echo '{"text": "a photo of a cat"}' | clip-text-processor preprocess
//!   cat tensors.json | clip-text-processor postprocess

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use tokenizers::Tokenizer;

/// CLIP context length
const CONTEXT_LENGTH: usize = 77;

/// Start of text token
const SOT_TOKEN: u32 = 49406;
/// End of text token
const EOT_TOKEN: u32 = 49407;

/// Embedded tokenizer JSON (from openai/clip-vit-base-patch32)
const TOKENIZER_JSON: &str = include_str!("../tokenizer.json");

#[derive(Deserialize)]
struct TextInput {
    text: String,
}

#[derive(Serialize)]
struct PreprocessOutput {
    input_ids: Vec<i64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: clip-text-processor <preprocess|postprocess>");
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
    // Read input from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    // Try to parse as JSON with text field, or use raw string
    let text = if let Ok(json_input) = serde_json::from_slice::<TextInput>(&input) {
        json_input.text
    } else {
        // Treat as raw text
        String::from_utf8_lossy(&input).trim().to_string()
    };

    // Load tokenizer
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes())
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Tokenize
    let encoding = tokenizer.encode(text.as_str(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Build input_ids with SOT, tokens, EOT, padding
    let mut input_ids: Vec<i64> = vec![0; CONTEXT_LENGTH];
    input_ids[0] = SOT_TOKEN as i64;

    // Copy tokens (truncate if needed, leave room for EOT)
    let max_tokens = CONTEXT_LENGTH - 2; // -2 for SOT and EOT
    let num_tokens = token_ids.len().min(max_tokens);

    for (i, &token) in token_ids.iter().take(num_tokens).enumerate() {
        input_ids[i + 1] = token as i64;
    }

    // Add EOT after last token
    input_ids[num_tokens + 1] = EOT_TOKEN as i64;

    // Output as JSON
    let mut output: HashMap<String, Vec<i64>> = HashMap::new();
    output.insert("input_ids".to_string(), input_ids);

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
    let embedding = tensors.get("text_embeds")
        .or_else(|| tensors.get("pooler_output"))
        .or_else(|| tensors.values().next())
        .cloned()
        .unwrap_or_default();

    // Output as embedding result
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
