//! Whisper Processor - Audio Transcription Pipeline
//!
//! Two-stage: Encoder (GPU via omny-compute) + Decoder (CPU internally)
//! Runs decoder autoregressive loop during postprocess.
//!
//! Usage:
//!   cat audio.wav | whisper-processor preprocess
//!   cat encoder_output.json | whisper-processor postprocess

use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Whisper audio parameters
const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const CHUNK_LENGTH: usize = 30; // seconds
const N_SAMPLES: usize = SAMPLE_RATE as usize * CHUNK_LENGTH; // 480000
const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000

/// Whisper special tokens (HuggingFace Optimum export IDs)
const SOT: i64 = 50258;       // <|startoftranscript|>
const EOT: i64 = 50257;       // <|endoftext|>
const LANG_EN: i64 = 50259;   // <|en|>
const TRANSCRIBE: i64 = 50359; // <|transcribe|>
const NO_TIMESTAMPS: i64 = 50363; // <|notimestamps|>
const MAX_TOKENS: usize = 224;

/// Temp file for passing audio between preprocess and postprocess
const TEMP_AUDIO_PATH: &str = "/tmp/whisper-processor-input.wav";

/// Mel filterbank (computed lazily)
static MEL_FILTERS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();

/// Vocabulary (loaded lazily)
static VOCAB: OnceLock<HashMap<i64, String>> = OnceLock::new();

fn get_mel_filters() -> &'static Vec<Vec<f32>> {
    MEL_FILTERS.get_or_init(|| compute_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE))
}

fn get_vocab() -> &'static HashMap<i64, String> {
    VOCAB.get_or_init(|| {
        let paths = [
            PathBuf::from("/tmp/whisper_vocab.json"),
            dirs::data_local_dir()
                .map(|d| d.join("omnynet").join("models").join("whisper").join("vocab.json"))
                .unwrap_or_default(),
        ];

        for path in &paths {
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(path) {
                    if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
                        let vocab: HashMap<i64, String> = map.into_iter()
                            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v)))
                            .collect();
                        eprintln!("[whisper] Loaded {} vocab entries from {:?}", vocab.len(), path);
                        return vocab;
                    }
                }
            }
        }

        eprintln!("[whisper] Warning: vocab not found, using empty vocab");
        HashMap::new()
    })
}

/// Path to decoder ONNX model
fn decoder_path() -> PathBuf {
    let paths = [
        PathBuf::from("/tmp/whisper_decoder.onnx"),
        dirs::data_local_dir()
            .map(|d| d.join("omnynet").join("models").join("whisper").join("decoder.onnx"))
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
        eprintln!("Usage: whisper-processor <preprocess|postprocess>");
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
        other => {
            eprintln!("Unknown command: {}. Use 'preprocess' or 'postprocess'", other);
            std::process::exit(1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// PREPROCESS: Audio WAV → Mel Spectrogram Tensor
// ═══════════════════════════════════════════════════════════════════

fn preprocess() -> Result<(), Box<dyn std::error::Error>> {
    // Read audio bytes from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;
    eprintln!("[whisper] Preprocess received {} bytes", input.len());

    // Save to temp for postprocess reference
    std::fs::write(TEMP_AUDIO_PATH, &input)?;

    // Parse WAV
    let samples = read_wav_to_f32_mono(&input)?;
    eprintln!("[whisper] Audio: {} samples ({:.1}s at {}Hz)",
        samples.len(), samples.len() as f32 / SAMPLE_RATE as f32, SAMPLE_RATE);

    // Pad or truncate to 30 seconds
    let mut padded = vec![0.0f32; N_SAMPLES];
    let copy_len = samples.len().min(N_SAMPLES);
    padded[..copy_len].copy_from_slice(&samples[..copy_len]);

    // Compute mel spectrogram
    let mel = compute_mel_spectrogram(&padded);
    eprintln!("[whisper] Mel spectrogram: {}x{}", N_MELS, N_FRAMES);

    // Flatten for JSON output (row-major: [n_mels, n_frames])
    let flat: Vec<f32> = mel.into_iter().flatten().collect();

    let output = serde_json::json!({
        "input_features": flat,
        "input_features_shape": [1, N_MELS, N_FRAMES],
    });

    let json = serde_json::to_vec(&output)?;
    eprintln!("[whisper] Output JSON size: {} bytes", json.len());
    io::stdout().write_all(&json)?;

    Ok(())
}

/// Read WAV file bytes into f32 mono samples at 16kHz
fn read_wav_to_f32_mono(data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let cursor = io::Cursor::new(data);
    let mut reader = hound::WavReader::new(cursor)?;
    let spec = reader.spec();

    eprintln!("[whisper] WAV: {}Hz, {} channels, {:?} {}bit",
        spec.sample_rate, spec.channels, spec.sample_format, spec.bits_per_sample);

    // Read samples as f32
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono = if spec.channels > 1 {
        raw_samples.chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    } else {
        raw_samples
    };

    // Resample to 16kHz if needed (simple linear interpolation)
    if spec.sample_rate != SAMPLE_RATE {
        eprintln!("[whisper] Resampling {}Hz → {}Hz", spec.sample_rate, SAMPLE_RATE);
        let ratio = SAMPLE_RATE as f64 / spec.sample_rate as f64;
        let new_len = (mono.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f64;
            let s0 = mono.get(src_idx).copied().unwrap_or(0.0);
            let s1 = mono.get(src_idx + 1).copied().unwrap_or(s0);
            resampled.push(s0 + (s1 - s0) * frac as f32);
        }
        Ok(resampled)
    } else {
        Ok(mono)
    }
}

/// Compute log-mel spectrogram matching Whisper's preprocessing
fn compute_mel_spectrogram(samples: &[f32]) -> Vec<Vec<f32>> {
    let filters = get_mel_filters();

    // STFT with Hann window
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // Hann window
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    let n_freq = N_FFT / 2 + 1; // 201 frequency bins

    // Compute STFT magnitude squared
    let mut magnitudes = vec![vec![0.0f32; N_FRAMES]; n_freq];

    for frame in 0..N_FRAMES {
        let start = frame * HOP_LENGTH;

        // Window the frame
        let mut fft_buf: Vec<Complex<f32>> = (0..N_FFT)
            .map(|i| {
                let sample = if start + i < samples.len() {
                    samples[start + i]
                } else {
                    0.0
                };
                Complex::new(sample * window[i], 0.0)
            })
            .collect();

        fft.process(&mut fft_buf);

        // Magnitude squared (power spectrum)
        for freq in 0..n_freq {
            magnitudes[freq][frame] = fft_buf[freq].norm_sqr();
        }
    }

    // Apply mel filterbank
    let mut mel = vec![vec![0.0f32; N_FRAMES]; N_MELS];
    for m in 0..N_MELS {
        for frame in 0..N_FRAMES {
            let mut sum = 0.0f32;
            for freq in 0..n_freq {
                sum += filters[m][freq] * magnitudes[freq][frame];
            }
            mel[m][frame] = sum;
        }
    }

    // Log scale and normalize (Whisper style)
    let mut max_val = f32::NEG_INFINITY;

    for m in 0..N_MELS {
        for frame in 0..N_FRAMES {
            let val = mel[m][frame].max(1e-10).log10();
            mel[m][frame] = val;
            if val > max_val {
                max_val = val;
            }
        }
    }

    // Whisper normalization: clamp to max - 8, then (x + 4) / 4
    let clamp_min = max_val - 8.0;
    for m in 0..N_MELS {
        for frame in 0..N_FRAMES {
            let val = mel[m][frame].max(clamp_min);
            mel[m][frame] = (val + 4.0) / 4.0;
        }
    }

    mel
}

/// Compute mel filterbank: N_MELS triangular filters over N_FFT/2+1 frequency bins
fn compute_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let n_freq = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;

    // Mel scale conversion
    let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(nyquist);

    // n_mels + 2 equally spaced mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f32> = hz_points.iter()
        .map(|&hz| hz * n_fft as f32 / sample_rate as f32)
        .collect();

    // Build triangular filters
    let mut filters = vec![vec![0.0f32; n_freq]; n_mels];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for freq in 0..n_freq {
            let f = freq as f32;
            if f >= left && f <= center && center > left {
                filters[m][freq] = (f - left) / (center - left);
            } else if f > center && f <= right && right > center {
                filters[m][freq] = (right - f) / (right - center);
            }
        }
    }

    // Slaney normalization
    for m in 0..n_mels {
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);
        for freq in 0..n_freq {
            filters[m][freq] *= enorm;
        }
    }

    filters
}

// ═══════════════════════════════════════════════════════════════════
// POSTPROCESS: Encoder Output → Autoregressive Decoder → Text
// ═══════════════════════════════════════════════════════════════════

fn postprocess() -> Result<(), Box<dyn std::error::Error>> {
    use ort::session::{Session, builder::GraphOptimizationLevel};

    // Read encoder output from stdin
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    let json_val: serde_json::Value = serde_json::from_slice(&input)?;

    // Extract encoder hidden states
    // omny-compute passes output tensors as JSON with tensor names as keys
    let encoder_output = extract_encoder_output(&json_val)?;
    eprintln!("[whisper] Encoder output: {} values", encoder_output.len());

    // Load decoder model
    let dec_path = decoder_path();
    if !dec_path.exists() {
        return Err(format!("Decoder model not found: {:?}", dec_path).into());
    }
    eprintln!("[whisper] Loading decoder from {:?}", dec_path);

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&dec_path)?;

    // Get decoder input/output names
    let input_names: Vec<String> = session.inputs.iter()
        .map(|i| i.name.clone())
        .collect();
    let output_names: Vec<String> = session.outputs.iter()
        .map(|o| o.name.clone())
        .collect();
    eprintln!("[whisper] Decoder inputs: {:?}", input_names);
    eprintln!("[whisper] Decoder outputs: {:?}", output_names);

    // Load vocab for decoding
    let vocab = get_vocab();

    // Initialize token sequence
    let mut tokens: Vec<i64> = vec![SOT, LANG_EN, TRANSCRIBE, NO_TIMESTAMPS];

    // Determine encoder hidden state shape
    // For whisper-base: [1, 1500, 512]
    let encoder_len = encoder_output.len();
    let hidden_size = if encoder_len % 1500 == 0 {
        encoder_len / 1500
    } else {
        // Try to infer from common model sizes
        match encoder_len {
            // tiny: 384, base: 512, small: 768, medium: 1024, large: 1280
            _ if encoder_len % 384 == 0 => 384,
            _ if encoder_len % 512 == 0 => 512,
            _ if encoder_len % 768 == 0 => 768,
            _ if encoder_len % 1024 == 0 => 1024,
            _ if encoder_len % 1280 == 0 => 1280,
            _ => return Err(format!("Cannot determine hidden size from encoder output len {}", encoder_len).into()),
        }
    };
    let seq_len = encoder_len / hidden_size;
    eprintln!("[whisper] Encoder: seq_len={}, hidden_size={}", seq_len, hidden_size);

    // Build encoder output tensor [1, seq_len, hidden_size]
    let encoder_shape = [1_i64, seq_len as i64, hidden_size as i64];
    let encoder_tensor = ort::value::Tensor::<f32>::from_array(
        (encoder_shape, encoder_output.into_boxed_slice())
    )?;

    // Autoregressive decoding
    for step in 0..MAX_TOKENS {
        let token_len = tokens.len();

        // Build decoder input tensor [1, token_len]
        let token_shape = [1_i64, token_len as i64];
        let token_tensor = ort::value::Tensor::<i64>::from_array(
            (token_shape, tokens.clone().into_boxed_slice())
        )?;

        // Run decoder
        let outputs = session.run(ort::inputs![
            "input_ids" => token_tensor,
            "encoder_hidden_states" => &encoder_tensor
        ])?;

        // Get logits from output
        let (_, output_val) = outputs.iter().next()
            .ok_or("No decoder output")?;

        let (shape, logits_data) = output_val.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();

        // shape: [1, token_len, vocab_size]
        let vocab_size = shape_vec.get(2).copied().unwrap_or(51865);

        // Get logits at last position
        let last_pos_start = (token_len - 1) * vocab_size;
        let last_logits = &logits_data[last_pos_start..last_pos_start + vocab_size];

        // Argmax
        let next_token = last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as i64)
            .unwrap_or(EOT);

        if next_token == EOT {
            eprintln!("[whisper] EOT at step {}", step);
            break;
        }

        tokens.push(next_token);

        if step % 10 == 0 {
            eprintln!("[whisper] Step {}: {} tokens generated", step, tokens.len() - 4);
        }
    }

    // Decode tokens to text (skip prompt tokens)
    let generated_tokens = &tokens[4..]; // Skip SOT, lang, task, notimestamps
    let text: String = generated_tokens.iter()
        .filter_map(|&id| {
            // Skip special tokens (>= 50257)
            if id >= 50257 {
                return None;
            }
            vocab.get(&id).map(|s| decode_token(s))
        })
        .collect::<Vec<_>>()
        .join("");

    let text = text.trim().to_string();
    eprintln!("[whisper] Transcription ({} tokens): {:?}", generated_tokens.len(), &text[..text.len().min(100)]);

    // Output JSON — use "type": "json" for omny-compute ProcessorOutput compatibility
    let output = serde_json::json!({
        "type": "json",
        "text": text,
        "language": "en",
        "tokens": generated_tokens.len(),
    });

    let json = serde_json::to_vec(&output)?;
    io::stdout().write_all(&json)?;

    // Cleanup
    let _ = std::fs::remove_file(TEMP_AUDIO_PATH);

    Ok(())
}

/// Extract encoder hidden states from omny-compute output JSON
fn extract_encoder_output(json: &serde_json::Value) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // omny-compute passes output tensors. The encoder output is typically named
    // "last_hidden_state" or the first output tensor.
    let possible_keys = ["last_hidden_state", "encoder_output", "output"];

    for key in &possible_keys {
        if let Some(arr) = json.get(key).and_then(|v| v.as_array()) {
            return Ok(flatten_to_f32(arr));
        }
    }

    // Try first key in object
    if let Some(obj) = json.as_object() {
        for (key, value) in obj {
            // Skip shape/metadata keys
            if key.contains("shape") || key.contains("_") {
                continue;
            }
            if let Some(arr) = value.as_array() {
                eprintln!("[whisper] Using encoder output from key: {}", key);
                return Ok(flatten_to_f32(arr));
            }
        }
    }

    // Try as flat array
    if let Some(arr) = json.as_array() {
        return Ok(flatten_to_f32(arr));
    }

    Err("Could not find encoder output in JSON".into())
}

/// Recursively flatten JSON array to Vec<f32>
fn flatten_to_f32(arr: &[serde_json::Value]) -> Vec<f32> {
    let mut result = Vec::new();
    for v in arr {
        match v {
            serde_json::Value::Number(n) => {
                result.push(n.as_f64().unwrap_or(0.0) as f32);
            }
            serde_json::Value::Array(inner) => {
                result.extend(flatten_to_f32(inner));
            }
            _ => {}
        }
    }
    result
}

/// Decode a token string (handle Whisper's byte-level BPE encoding)
fn decode_token(token: &str) -> String {
    // Whisper uses byte-level BPE with special unicode characters
    // Replace common byte tokens: Ġ = space, Ċ = newline, etc.
    token
        .replace('\u{0120}', " ")  // Ġ = leading space
        .replace('\u{010a}', "\n") // Ċ = newline
        .replace('\u{0109}', "\t") // ĉ = tab
}
