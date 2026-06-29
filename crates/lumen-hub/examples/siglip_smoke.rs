//! Smoke test: load the SigLIP Burn models and run one forward pass each.
//!
//! Run with:
//!   cargo run --example siglip_smoke --no-default-features --features cpu,siglip -- <model_dir>

use lumen_hub::backend::default_device;
use lumen_hub::models::siglip::{SiglipTextModel, SiglipVisionModel};

fn main() {
    let model_dir = std::env::args().nth(1).unwrap_or_else(|| {
        "/Volumes/CodeBase/Projects/lumen-models/siglip2-base-patch16-224".into()
    });
    let device = default_device();

    // Optional 2nd arg: precision (fp32 | fp16 | fp16q8). Default fp32.
    let precision = std::env::args().nth(2).unwrap_or_else(|| "fp32".into());

    let model_name = std::path::Path::new(&model_dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("siglip2-base-patch16-224");
    // so400m is patch14/384; base is patch16/224.
    let res = if model_name.contains("so400m") || model_name.contains("384") {
        384
    } else {
        224
    };
    let vision_path = format!("{model_dir}/burn/vision.{precision}.bpk");
    let head_path = format!("{model_dir}/burn/aesthetic.{precision}.bpk");
    println!("loading vision model from {vision_path} (res={res})");
    let vision = SiglipVisionModel::load(
        model_name,
        &vision_path,
        &precision,
        device.clone(),
        std::path::Path::new(&head_path)
            .exists()
            .then_some(head_path.as_str()),
    )
    .expect("vision model loads");
    // Deterministic non-trivial input so the aesthetic head sees real features.
    let pixels: Vec<f32> = (0..3 * res * res)
        .map(|i| ((i % 255) as f32 / 255.0) - 0.5)
        .collect();
    let (emb, score) = vision.encode(pixels, 1, res, res);
    let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!(
        "vision embedding: dim={} first={:?} norm={norm}",
        emb.len(),
        &emb[..emb.len().min(5)]
    );
    println!("aesthetic score: {score:?}");

    let text_path = format!("{model_dir}/burn/text.{precision}.bpk");
    println!("loading text model from {text_path}");
    let text = SiglipTextModel::load(model_name, &text_path, &precision, device)
        .expect("text model loads");
    // 64 token ids padded with 0; a trivial sequence just to exercise forward.
    let mut ids = vec![0i64; 64];
    ids[0] = 2; // bos-ish
    let emb = text.encode(&ids);
    let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!(
        "text embedding: dim={} first={:?} norm={norm}",
        emb.len(),
        &emb[..emb.len().min(5)]
    );
}
