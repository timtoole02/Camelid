//! Local English dictation. Candle CPU + hound + the existing Rust tokenizer;
//! no whisper.cpp, Python process, native BLAS, or cloud transcription service.
use anyhow::{bail, ensure, Context, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as whisper, model::Whisper, Config};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Cursor, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

pub const MAX_SECONDS: usize = 30;
pub const MAX_UPLOAD_BYTES: usize = MAX_SECONDS * 96_000 * 2 + 4096;
pub const MODEL_BYTES: u64 = 153_467_752;
const REVISION: &str = "87c7102498dcde7456f24cfd30239ca606ed9063";
const FILES: [(&str, u64, &str); 3] = [
    (
        "config.json",
        1937,
        "b5366317df280f79e6cc366f259f379674f81a8721d458acf2e00680ccb10b1d",
    ),
    (
        "tokenizer.json",
        2405679,
        "5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0",
    ),
    (
        "model.safetensors",
        151060136,
        "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad",
    ),
];

pub fn model_dir(models: &Path) -> PathBuf {
    models.join("speech").join("whisper-tiny.en")
}
pub fn installed(dir: &Path) -> bool {
    FILES.iter().all(|(name, size, _)| {
        fs::metadata(dir.join(name)).is_ok_and(|m| m.is_file() && m.len() == *size)
    })
}
fn verified_bytes(dir: &Path, file: &(&str, u64, &str)) -> Result<Vec<u8>> {
    let (name, size, digest) = file;
    let path = dir.join(name);
    ensure!(
        fs::metadata(&path)?.len() == *size,
        "Speech model file {name} has the wrong size. Download it again."
    );
    let bytes = fs::read(&path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == *digest,
        "Speech model file {name} failed verification. Download it again."
    );
    Ok(bytes)
}

/// Only fixed, revision-pinned model files can be downloaded. Audio never enters
/// this HTTP client. A failed download never replaces a valid installed file.
pub fn install(dir: &Path, max_download_bytes: u64) -> Result<()> {
    ensure!(
        MODEL_BYTES <= max_download_bytes,
        "Speech model exceeds the configured download limit."
    );
    fs::create_dir_all(dir)?;
    let tls = ureq::tls::TlsConfig::builder()
        .unversioned_rustls_crypto_provider(std::sync::Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .build();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(tls)
        .timeout_global(Some(Duration::from_secs(600)))
        .build()
        .into();
    for file in &FILES {
        if verified_bytes(dir, file).is_ok() {
            continue;
        }
        let (name, size, digest) = file;
        let url =
            format!("https://huggingface.co/openai/whisper-tiny.en/resolve/{REVISION}/{name}");
        let temp = dir.join(format!(".{name}.{}.part", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut response = agent
                .get(&url)
                .call()
                .context("Could not download speech model. Check your connection and retry.")?;
            let mut output = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            let mut reader = response.body_mut().as_reader().take(size + 1);
            let mut hash = Sha256::new();
            let mut total = 0;
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                total += n as u64;
                ensure!(
                    total <= *size,
                    "Speech model download is larger than expected."
                );
                hash.update(&buffer[..n]);
                output.write_all(&buffer[..n])?;
            }
            ensure!(
                total == *size && format!("{:x}", hash.finalize()) == *digest,
                "Speech model download failed checksum verification. Retry the download."
            );
            output.sync_all()?;
            drop(output);
            let dest = dir.join(name);
            if dest.exists() {
                fs::remove_file(&dest)?;
            }
            fs::rename(&temp, dest)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result?;
    }
    Ok(())
}

/// Accept bounded mono PCM16 WAV, including the browser's native sample rate.
/// Resampling stays in Rust. Windowed sinc low-pass filtering avoids aliasing
/// when 44.1/48/96 kHz microphone audio is reduced to Whisper's 16 kHz input.
pub fn decode_audio(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() <= MAX_UPLOAD_BYTES,
        "Recording is too large (maximum 30 seconds)."
    );
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .context("Expected a mono PCM16 WAV recording.")?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.bits_per_sample == 16
            && spec.sample_format == hound::SampleFormat::Int,
        "Expected mono PCM16 WAV audio."
    );
    ensure!(
        (8_000..=96_000).contains(&spec.sample_rate),
        "Unsupported microphone sample rate."
    );
    let count = reader.len() as usize;
    ensure!(
        count >= spec.sample_rate as usize / 5,
        "Recording is too short. Speak for at least a moment."
    );
    ensure!(
        count <= spec.sample_rate as usize * MAX_SECONDS,
        "Recording exceeds 30 seconds."
    );
    let samples = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(samples.len() == count, "Recording is incomplete.");
    if spec.sample_rate == 16_000 {
        return Ok(samples);
    }
    Ok(resample(&samples, spec.sample_rate))
}
fn resample(input: &[f32], rate: u32) -> Vec<f32> {
    let ratio = rate as f64 / 16_000.0;
    let cutoff = (1.0 / ratio).min(1.0) * 0.95;
    let radius = (24.0 / cutoff).ceil() as isize;
    (0..input.len() * 16_000 / rate as usize)
        .map(|i| {
            let center = i as f64 * ratio;
            let mut sum = 0.0;
            let mut weights = 0.0;
            for j in center.floor() as isize - radius..=center.floor() as isize + radius {
                if j < 0 || j >= input.len() as isize {
                    continue;
                }
                let d = j as f64 - center;
                let x = std::f64::consts::PI * d * cutoff;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
                let weight = sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / radius as f64).cos());
                sum += input[j as usize] as f64 * weight;
                weights += weight;
            }
            (sum / weights) as f32
        })
        .collect()
}

/// Slaney-normalized triangular mel filters, matching Whisper's 80-bin bank.
fn mel_filters() -> Vec<f32> {
    let hz_to_mel = |hz: f64| {
        if hz < 1000.0 {
            hz * 3.0 / 200.0
        } else {
            15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15.0 {
            mel * 200.0 / 3.0
        } else {
            1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
        }
    };
    let edges: Vec<_> = (0..82)
        .map(|i| mel_to_hz(hz_to_mel(8000.0) * i as f64 / 81.0))
        .collect();
    (0..80)
        .flat_map(|band| {
            let edges = &edges;
            (0..201).map(move |bin| {
                let hz = bin as f64 * 40.0;
                let rise = (hz - edges[band]) / (edges[band + 1] - edges[band]);
                let fall = (edges[band + 2] - hz) / (edges[band + 2] - edges[band + 1]);
                (rise.min(fall).max(0.0) * 2.0 / (edges[band + 2] - edges[band])) as f32
            })
        })
        .collect()
}

pub fn transcribe(dir: &Path, samples: &[f32]) -> Result<String> {
    // Avoid Whisper hallucinations on digital silence or near-silent input.
    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    if rms < 0.0005 {
        return Ok(String::new());
    }
    let config: Config = serde_json::from_slice(&verified_bytes(dir, &FILES[0])?)?;
    let tokenizer =
        Tokenizer::from_bytes(verified_bytes(dir, &FILES[1])?).map_err(anyhow::Error::msg)?;
    let weights = verified_bytes(dir, &FILES[2])?;
    let device = Device::Cpu;
    let vb = VarBuilder::from_buffered_safetensors(weights, whisper::DTYPE, &device)?;
    let mut model = Whisper::load(&vb, config.clone())?;
    let id = |text: &str| {
        tokenizer
            .token_to_id(text)
            .with_context(|| format!("Missing speech token {text}"))
    };
    let sot = id(whisper::SOT_TOKEN)?;
    let eot = id(whisper::EOT_TOKEN)?;
    let no_timestamps = id(whisper::NO_TIMESTAMPS_TOKEN)?;
    let no_speech = whisper::NO_SPEECH_TOKENS
        .iter()
        .find_map(|s| tokenizer.token_to_id(s))
        .context("Missing no-speech token")?;
    let mut padded = samples.to_vec();
    padded.resize(whisper::N_SAMPLES, 0.0);
    let mel = whisper::audio::pcm_to_mel(&config, &padded, &mel_filters());
    let frames = mel.len() / config.num_mel_bins;
    let mel = Tensor::from_vec(mel, (1, config.num_mel_bins, frames), &device)?.narrow(
        2,
        0,
        whisper::N_FRAMES,
    )?;
    let started = Instant::now();
    let features = model.encoder.forward(&mel, true)?;
    // English-only Whisper uses SOT + no-timestamps, without a language/task token.
    let mut tokens = vec![sot, no_timestamps];
    let blank = tokenizer
        .encode(" ", false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let mut log_probability = 0.0;
    let mut no_speech_probability = 0.0;
    for step in 0..config.max_target_positions / 2 {
        ensure!(
            started.elapsed() < Duration::from_secs(120),
            "Transcription timed out. Try a shorter recording."
        );
        let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let hidden = model.decoder.forward(&input, &features, step == 0)?;
        if step == 0 {
            let first = model
                .decoder
                .final_linear(&hidden.i((.., ..1, ..))?)?
                .i((0, 0))?;
            no_speech_probability =
                candle_nn::ops::softmax(&first, 0)?.to_vec1::<f32>()?[no_speech as usize];
        }
        let logits = model
            .decoder
            .final_linear(&hidden.i((.., tokens.len() - 1.., ..))?)?
            .i((0, 0))?;
        let mut logits = logits.to_vec1::<f32>()?;
        for (i, value) in logits.iter_mut().enumerate() {
            if config.suppress_tokens.contains(&(i as u32))
                || (i as u32 > eot)
                || (step == 0 && (i as u32 == eot || blank.contains(&(i as u32))))
            {
                *value = f32::NEG_INFINITY;
            }
        }
        let (next, max) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("Empty speech logits")?;
        ensure!(max.is_finite(), "Speech decoder produced invalid logits.");
        let log_p = -logits
            .iter()
            .map(|v| (*v - *max).exp() as f64)
            .sum::<f64>()
            .ln();
        log_probability += log_p;
        if next as u32 == eot {
            if no_speech_probability > 0.6 && log_probability / ((step + 1) as f64) < -1.0 {
                return Ok(String::new());
            }
            return tokenizer
                .decode(&tokens[2..], true)
                .map(|s| s.trim().to_string())
                .map_err(anyhow::Error::msg);
        }
        tokens.push(next as u32);
    }
    bail!("Speech was too long or unclear to transcribe reliably. Try a shorter recording.")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wav(rate: u32, channels: u16, seconds: f32) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut cursor,
            hound::WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..(rate as f32 * seconds) as usize * channels as usize {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        cursor.into_inner()
    }
    #[test]
    fn recording_validation_and_resampling() {
        assert!(decode_audio(b"not wav").is_err());
        assert!(decode_audio(&wav(16000, 2, 1.0)).is_err());
        assert!(decode_audio(&wav(16000, 1, 0.1)).is_err());
        assert!(decode_audio(&wav(16000, 1, 30.1)).is_err());
        for rate in [16000, 44100, 48000, 96000] {
            let samples = decode_audio(&wav(rate, 1, 0.25)).unwrap();
            assert_eq!(samples.len(), 4000);
            assert!(samples.iter().all(|v| *v == 0.0));
        }
    }
    #[test]
    fn resampler_rejects_aliases_and_preserves_speech_band() {
        let sine = |hz: f32| {
            (0..4800)
                .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * hz / 48000.0).sin())
                .collect::<Vec<_>>()
        };
        let energy = |v: &[f32]| {
            v[100..v.len() - 100].iter().map(|x| x * x).sum::<f32>() / (v.len() - 200) as f32
        };
        assert!(energy(&resample(&sine(1000.0), 48000)) > 0.45);
        assert!(energy(&resample(&sine(12000.0), 48000)) < 0.001);
    }
    #[test]
    fn missing_or_corrupt_models_are_not_ready() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!installed(dir.path()));
        fs::write(dir.path().join("config.json"), vec![0; 1937]).unwrap();
        assert!(verified_bytes(dir.path(), &FILES[0]).is_err());
        assert_eq!(FILES.iter().map(|f| f.1).sum::<u64>(), MODEL_BYTES);
    }
    #[test]
    fn silence_does_not_load_model_or_hallucinate() {
        assert_eq!(
            transcribe(Path::new("/missing-speech-model"), &vec![0.0; 16000]).unwrap(),
            ""
        );
    }
    /// Opt-in, real checkpoint test; no network in ordinary cargo test runs.
    #[test]
    #[ignore = "requires CAMELID_SPEECH_TEST_MODEL_DIR and CAMELID_SPEECH_TEST_WAV"]
    fn real_recording() {
        let dir = std::env::var("CAMELID_SPEECH_TEST_MODEL_DIR").unwrap();
        let wav = fs::read(std::env::var("CAMELID_SPEECH_TEST_WAV").unwrap()).unwrap();
        let text = transcribe(Path::new(&dir), &decode_audio(&wav).unwrap()).unwrap();
        eprintln!("Transcript: {text}");
        let expected = std::env::var("CAMELID_SPEECH_TEST_EXPECT").unwrap();
        assert!(
            text.to_lowercase().contains(&expected.to_lowercase()),
            "{text}"
        );
    }
}
