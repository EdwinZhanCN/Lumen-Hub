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

    let vision_path = format!("{model_dir}/burn/vision.fp32.bpk");
    println!("loading vision model from {vision_path}");
    let vision = SiglipVisionModel::load(
        "siglip2-base-patch16-224",
        &vision_path,
        "fp32",
        device.clone(),
    )
    .expect("vision model loads");
    let pixels = vec![0.0f32; 3 * 224 * 224];
    let emb = vision.encode(pixels, 1, 224, 224);
    let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!(
        "vision embedding: dim={} first={:?} norm={norm}",
        emb.len(),
        &emb[..emb.len().min(5)]
    );

    let text_path = format!("{model_dir}/burn/text.fp32.bpk");
    println!("loading text model from {text_path}");
    let text = SiglipTextModel::load("siglip2-base-patch16-224", &text_path, "fp32", device)
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
