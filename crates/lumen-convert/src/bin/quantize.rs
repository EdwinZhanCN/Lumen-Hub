//! Convert every model's fp32 `.bpk` to weight-only int8 and validate end-to-end.
//!
//! For each `<model>/burn/<component>.fp32.bpk`: load -> selective Q8 (per-block,
//! + size-gated embedding tables) -> save `<component>.int8.bpk`, then run forward
//! on both fp32 and the reloaded int8 and report the output cosine. Forward is
//! wrapped in `catch_unwind`, so a wrong input-shape guess reports SKIP rather than
//! aborting the run; the int8 artifact is already saved by then.
//!
//! Run:
//!   cargo run --release -p lumen-convert --bin quantize -- [models_dir]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use burn::module::Module;
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

use lumen_convert::{QuantConfig, SelectiveQuantizer, cosine, q8_scheme, server};
use lumen_hub::backend::{Backend, Device, default_device};
use lumen_hub::model_arch::{
    antelopev2, bioclip2, pp_ocrv5, siglip2_base_patch16_224 as siglip_base,
    siglip2_so400m_patch14_384 as siglip_so400m,
};

const DEFAULT_ROOT: &str = "lumen-models";

fn file_mb(path: &str) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / 1e6)
        .unwrap_or(0.0)
}

/// Deterministic image-like input `[1, 3, h, w]`.
fn img(h: usize, w: usize, device: &Device) -> Tensor<Backend, 4> {
    let pixels: Vec<f32> = (0..3 * h * w)
        .map(|i| {
            let c = (i / (h * w)) as f32;
            let y = ((i / w) % h) as f32 / h as f32;
            let x = (i % w) as f32 / w as f32;
            0.6 * ((x * 3.0 + c).sin() * (y * 2.0 + c).cos())
        })
        .collect();
    Tensor::from_data(TensorData::new(pixels, [1, 3, h, w]), device)
}

/// Deterministic token ids `[1, seq]`.
fn toks(seq: usize, device: &Device) -> Tensor<Backend, 2, Int> {
    let mut ids = vec![0i64; seq];
    ids[0] = 2;
    for (k, v) in ids.iter_mut().enumerate().take(seq.min(8)).skip(1) {
        *v = (k as i64) + 100;
    }
    Tensor::from_data(TensorData::new(ids, [1, seq]), device)
}

fn flat<const D: usize>(t: Tensor<Backend, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

/// Quantize one component and validate it end-to-end.
///
/// `embed` runs the model's forward and flattens the output to a vector; returning
/// an empty vector opts out of cosine validation (e.g. unused encoders with an
/// unknown input contract).
fn convert<M, L, N, E>(
    model: &str,
    comp: &str,
    root: &str,
    cfg: QuantConfig,
    load: L,
    new: N,
    embed: E,
) where
    M: Module<Backend> + ModuleSnapshot<Backend>,
    L: Fn(&str, &Device) -> M,
    N: Fn(&Device) -> M,
    E: Fn(&M, &Device) -> Vec<f32>,
{
    // Optional filter for iterating on a single component, e.g.
    // LUMEN_CONVERT_ONLY=siglip2-base-patch16-224/text
    if let Ok(only) = std::env::var("LUMEN_CONVERT_ONLY") {
        if !format!("{model}/{comp}").contains(&only) {
            return;
        }
    }
    let fp32_path = format!("{root}/{model}/burn/{comp}.fp32.bpk");
    let int8_path = format!("{root}/{model}/burn/{comp}.int8.bpk");
    if !Path::new(&fp32_path).exists() {
        println!("  {model}/{comp}: SKIP (no fp32 bpk)");
        return;
    }
    let device = default_device();
    let fp32 = load(&fp32_path, &device);
    // In-binary forward validation runs on the CPU/Flex backend, where int8
    // inference is slow; set LUMEN_CONVERT_NO_VALIDATE=1 to skip it (validate the
    // artifact separately on a GPU backend instead).
    let validate = std::env::var_os("LUMEN_CONVERT_NO_VALIDATE").is_none();
    let emb_fp32 = validate
        .then(|| catch_unwind(AssertUnwindSafe(|| embed(&fp32, &device))).ok())
        .flatten();

    let mut q = SelectiveQuantizer::new(q8_scheme::<Backend>(), cfg);
    let quantized = fp32.map(&mut q);
    let mut store = BurnpackStore::from_file(&int8_path).overwrite(true);
    quantized.save_into(&mut store).expect("save int8 burnpack");

    // Reload from disk and run forward → confirms QFloat serialization + quantized inference.
    let mut int8 = new(&device);
    let mut reload = BurnpackStore::from_file(&int8_path);
    int8.load_from(&mut reload).expect("reload int8 burnpack");
    let emb_int8 = validate
        .then(|| catch_unwind(AssertUnwindSafe(|| embed(&int8, &device))).ok())
        .flatten();

    let cos = match (&emb_fp32, &emb_int8) {
        (Some(a), Some(b)) if !a.is_empty() && a.len() == b.len() => Some(cosine(a, b)),
        _ => None,
    };
    let cos_str = cos.map_or_else(|| "  cos=  n/a  ".to_string(), |c| format!("  cos={c:.5}"));
    println!(
        "  {model}/{comp}: q={} skip={} | werr {:.4}/{:.4} |{cos_str}| {:.0}->{:.0} MB ({:.2}x)",
        q.quantized,
        q.skipped,
        q.mean_err(),
        q.err_max,
        file_mb(&fp32_path),
        file_mb(&int8_path),
        file_mb(&fp32_path) / file_mb(&int8_path).max(f64::MIN_POSITIVE),
    );
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ROOT.into());
    let cfg = QuantConfig::default();
    println!("models root: {root}");
    println!(
        "scheme: symmetric Q8, per-block cap {:?}, large-const>={:?}\n",
        cfg.block_cap, cfg.large_const_min_numel
    );
    println!(
        "(werr = weight reconstruction L1 error mean/max; cos = fp32-vs-int8 output cosine)\n"
    );

    // SigLIP base
    convert(
        "siglip2-base-patch16-224",
        "text",
        &root,
        cfg,
        |p, d| siglip_base::text::Model::<Backend>::from_file(p, d),
        siglip_base::text::Model::<Backend>::new,
        |m, d| flat(m.forward(toks(64, d))),
    );
    convert(
        "siglip2-base-patch16-224",
        "vision",
        &root,
        cfg,
        |p, d| siglip_base::vision::Model::<Backend>::from_file(p, d),
        siglip_base::vision::Model::<Backend>::new,
        |m, d| flat(m.forward(img(224, 224, d))),
    );

    // SigLIP so400m (forward returns (hidden, pooled) → validate pooled .1)
    convert(
        "siglip2-so400m-patch14-384",
        "text",
        &root,
        cfg,
        |p, d| siglip_so400m::text::Model::<Backend>::from_file(p, d),
        siglip_so400m::text::Model::<Backend>::new,
        |m, d| flat(m.forward(toks(64, d)).1),
    );
    convert(
        "siglip2-so400m-patch14-384",
        "vision",
        &root,
        cfg,
        |p, d| siglip_so400m::vision::Model::<Backend>::from_file(p, d),
        siglip_so400m::vision::Model::<Backend>::new,
        |m, d| flat(m.forward(img(384, 384, d)).1),
    );

    // BioCLIP (text encoder unused at runtime / unknown input contract → no cosine)
    convert(
        "bioclip-2",
        "text",
        &root,
        cfg,
        |p, d| bioclip2::text::Model::<Backend>::from_file(p, d),
        bioclip2::text::Model::<Backend>::new,
        |_, _| Vec::new(),
    );
    convert(
        "bioclip-2",
        "vision",
        &root,
        cfg,
        |p, d| bioclip2::vision::Model::<Backend>::from_file(p, d),
        bioclip2::vision::Model::<Backend>::new,
        |m, d| flat(m.forward(img(224, 224, d))),
    );

    // PP-OCRv5
    convert(
        "pp-ocrv5",
        "detection",
        &root,
        cfg,
        |p, d| pp_ocrv5::detection::Model::<Backend>::from_file(p, d),
        pp_ocrv5::detection::Model::<Backend>::new,
        |m, d| flat(m.forward(img(640, 640, d))),
    );
    convert(
        "pp-ocrv5",
        "recognition",
        &root,
        cfg,
        |p, d| pp_ocrv5::recognition::Model::<Backend>::from_file(p, d),
        pp_ocrv5::recognition::Model::<Backend>::new,
        |m, d| flat(m.forward(img(48, 320, d))),
    );

    // InsightFace antelopev2 (detection forward returns a 9-tuple of heads)
    convert(
        "antelopev2",
        "detection",
        &root,
        cfg,
        |p, d| antelopev2::detection::Model::<Backend>::from_file(p, d),
        antelopev2::detection::Model::<Backend>::new,
        |m, d| {
            let o = m.forward(img(640, 640, d));
            let mut v = Vec::new();
            for t in [o.0, o.1, o.2, o.3, o.4, o.5, o.6, o.7, o.8] {
                v.extend(flat(t));
            }
            v
        },
    );
    convert(
        "antelopev2",
        "recognition",
        &root,
        cfg,
        |p, d| antelopev2::recognition::Model::<Backend>::from_file(p, d),
        antelopev2::recognition::Model::<Backend>::new,
        |m, d| flat(m.forward(img(112, 112, d))),
    );

    // pp-ocrv5-server: generated from ONNX by build.rs. Stage the generated fp32
    // bpk into the new model repo, then convert + validate like any other model.
    stage_generated_fp32(
        "pp_ocrv5_server/detection/detection.bpk",
        &root,
        "detection",
    );
    convert(
        "pp-ocrv5-server",
        "detection",
        &root,
        cfg,
        |p, d| server::detection::Model::<Backend>::from_file(p, d),
        server::detection::Model::<Backend>::new,
        |m, d| flat(m.forward(img(640, 640, d))),
    );
    stage_generated_fp32(
        "pp_ocrv5_server/recognition/recognition.bpk",
        &root,
        "recognition",
    );
    convert(
        "pp-ocrv5-server",
        "recognition",
        &root,
        cfg,
        |p, d| server::recognition::Model::<Backend>::from_file(p, d),
        server::recognition::Model::<Backend>::new,
        |m, d| flat(m.forward(img(48, 320, d))),
    );
    stage_generated_fp32(
        "pp_ocrv5_server/classification/classification.bpk",
        &root,
        "classification",
    );
    convert(
        "pp-ocrv5-server",
        "classification",
        &root,
        cfg,
        |p, d| server::classification::Model::<Backend>::from_file(p, d),
        server::classification::Model::<Backend>::new,
        |m, d| flat(m.forward(img(80, 160, d))),
    );
}

/// Copy a build-time generated fp32 burnpack (`OUT_DIR/<gen_rel>`) into the model
/// repo as `<root>/pp-ocrv5-server/burn/<comp>.fp32.bpk`.
fn stage_generated_fp32(gen_rel: &str, root: &str, comp: &str) {
    if let Ok(only) = std::env::var("LUMEN_CONVERT_ONLY") {
        if !format!("pp-ocrv5-server/{comp}").contains(&only) {
            return;
        }
    }
    let src = format!("{}/{}", env!("OUT_DIR"), gen_rel);
    let dir = format!("{root}/pp-ocrv5-server/burn");
    std::fs::create_dir_all(&dir).expect("create repo burn dir");
    let dst = format!("{dir}/{comp}.fp32.bpk");
    std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("stage {src} -> {dst}: {e}"));
}
