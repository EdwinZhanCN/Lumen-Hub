//! De-risk check for selective weight-only Q8 quantization (route B core).
//!
//! Loads the SigLIP base vision encoder, runs it once in fp32 and once with a
//! *selective* int8 quantizer applied to weight-bearing modules only (Linear /
//! Conv / Embedding, rank >= 2), and reports the cosine similarity between the two
//! embeddings. This uses fake-quant (`quantize_dynamic().dequantize()`), whose
//! numerical error equals what weight-only int8 inference would produce — so it
//! answers "does selective Q8 preserve accuracy?" before we build the full
//! ONNX -> int8 toolchain.
//!
//! Run (CPU, has quantize ops):
//!   cargo run --release --example quantize_check --no-default-features \
//!     --features cpu,siglip -- <model_dir>

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::ops::QuantizedTensor;
use burn::tensor::quantization::{QTensorPrimitive, QuantLevel, QuantScheme, QuantValue};
use burn::tensor::{Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

use lumen_hub::backend::{Backend, Device, default_device};
use lumen_hub::model_arch::siglip2_base_patch16_224::vision;

const H: usize = 224;
const W: usize = 224;

/// Quantizes only float weights inside Linear/Conv/Embedding containers (rank >= 2),
/// leaving biases, norm params, and bare graph constants untouched.
struct SelectiveQuantizer {
    scheme: QuantScheme,
    /// `Some(cap)` = per-block along the last axis with block size = largest divisor
    /// of the last dim that is `<= cap`; `None` = per-tensor.
    block_cap: Option<usize>,
    /// When true, dequantize back to f32 after quantizing (fake-quant, measures
    /// accuracy only). When false, keep the weight as QFloat (real quantized
    /// inference + memory savings).
    dequantize: bool,
    /// Per-depth flag: is the current container a quantizable weight module?
    stack: Vec<bool>,
    quantized: usize,
    skipped: usize,
}

impl SelectiveQuantizer {
    fn new(scheme: QuantScheme, block_cap: Option<usize>, dequantize: bool) -> Self {
        Self {
            scheme,
            block_cap,
            dequantize,
            stack: Vec::new(),
            quantized: 0,
            skipped: 0,
        }
    }

    fn quantizable_container(container_type: &str) -> bool {
        matches!(
            container_type,
            "Struct:Linear"
                | "Struct:Conv1d"
                | "Struct:Conv2d"
                | "Struct:Conv3d"
                | "Struct:ConvTranspose1d"
                | "Struct:ConvTranspose2d"
                | "Struct:ConvTranspose3d"
                | "Struct:Embedding"
        )
    }
}

impl<B: BackendTrait> ModuleMapper<B> for SelectiveQuantizer {
    fn enter_module(&mut self, _name: &str, container_type: &str) {
        self.stack.push(Self::quantizable_container(container_type));
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let in_weight_module = *self.stack.last().unwrap_or(&false);
        // rank >= 2 keeps weight matrices/kernels and skips 1-D biases/norm params.
        if in_weight_module && D >= 2 {
            self.quantized += 1;
            let base = self.scheme;
            let block_cap = self.block_cap;
            let dequantize = self.dequantize;
            param.map(|t| {
                let last = t.dims()[D - 1];
                let scheme = match block_cap.and_then(|cap| block_size_for(last, cap)) {
                    Some(b) => base.with_level(QuantLevel::block([b])),
                    None => base.with_level(QuantLevel::Tensor),
                };
                let q = t.quantize_dynamic(&scheme);
                if dequantize { q.dequantize() } else { q }
            })
        } else {
            self.skipped += 1;
            param
        }
    }
}

/// Largest divisor of `last` in `[2, min(cap, 255)]` (block values are u8-capped),
/// or `None` when the last dim is prime/awkward — caller falls back to per-tensor.
fn block_size_for(last: usize, cap: usize) -> Option<u8> {
    (2..=last.min(cap).min(255))
        .rev()
        .find(|d| last % d == 0)
        .map(|d| d as u8)
}

fn embed(model: &vision::Model<Backend>, pixels: &[f32], device: &Device) -> Vec<f32> {
    let data = TensorData::new(pixels.to_vec(), [1, 3, H, W]);
    let input = Tensor::<Backend, 4>::from_data(data, device);
    model
        .forward(input)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("f32 embedding")
}

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .expect("usage: quantize_check <model_dir>");
    let device = default_device();
    let path = format!("{model_dir}/burn/vision.fp32.bpk");

    let pixels: Vec<f32> = (0..3 * H * W)
        .map(|i| {
            let c = (i / (H * W)) as f32;
            let y = ((i / W) % H) as f32 / H as f32;
            let x = (i % W) as f32 / W as f32;
            0.6 * ((x * 3.0 + c).sin() * (y * 2.0 + c).cos())
        })
        .collect();

    println!("loading fp32 vision model");
    let fp32 = vision::Model::<Backend>::from_file(&path, &device);
    let emb_fp32 = embed(&fp32, &pixels, &device);

    // dequantize=true measures accuracy (fake-quant, f32 forward).
    for (label, block_cap) in [
        ("fake-quant per-tensor", None),
        ("fake-quant block<=128", Some(128)),
        ("fake-quant block<=64", Some(64)),
        ("fake-quant block<=32", Some(32)),
    ] {
        let model = vision::Model::<Backend>::from_file(&path, &device);
        let mut q = SelectiveQuantizer::new(base_scheme(), block_cap, true);
        let q_model = model.map(&mut q);
        let emb_q = embed(&q_model, &pixels, &device);
        let cosine = cosine(&emb_fp32, &emb_q);
        println!(
            "{label:>24}: quantized {} skipped {} | cosine = {cosine:.6}",
            q.quantized, q.skipped
        );
    }

    // dequantize=false keeps QFloat weights: confirms real quantized inference works
    // (and that the runtime would keep int8 in memory) — must match the fake-quant cosine.
    println!("\n--- real quantized inference (weights kept as int8) ---");
    let model = vision::Model::<Backend>::from_file(&path, &device);
    let mut q = SelectiveQuantizer::new(base_scheme(), Some(32), false);
    let q_model = model.map(&mut q);
    let emb_q = embed(&q_model, &pixels, &device);
    println!(
        "  real Q8 block<=32: quantized {} skipped {} | cosine = {:.6}",
        q.quantized,
        q.skipped,
        cosine(&emb_fp32, &emb_q)
    );

    // Load the SAVED int8 .bpk (the artifact the service uses) and run the SAME
    // synthetic input — isolates whether the save→reload round-trip is the culprit
    // (vs in-memory quant above, which is identical to the .bpk weights).
    let int8_path = path.replace("fp32", "int8");
    if std::path::Path::new(&int8_path).exists() {
        let mut loaded = vision::Model::<Backend>::new(&device);
        loaded
            .load_from(&mut BurnpackStore::from_file(&int8_path))
            .expect("reload int8 bpk");
        let emb_loaded = embed(&loaded, &pixels, &device);
        println!(
            "  loaded int8.bpk (saved on cpu): cosine vs fp32 = {:.6}",
            cosine(&emb_fp32, &emb_loaded)
        );
    }

    // Same-backend round-trip: save the in-memory (metal) quantized model to a bpk
    // and reload it on metal. If this is ~0.999 while the cpu-saved .bpk above is
    // low, the bug is cross-backend quant serialization → fix = quantize on metal.
    let tmp = "/tmp/siglip_vision_samebackend.int8.bpk";
    let mut store = BurnpackStore::from_file(tmp).overwrite(true);
    q_model.save_into(&mut store).expect("save int8 on metal");
    let mut reloaded = vision::Model::<Backend>::new(&device);
    reloaded
        .load_from(&mut BurnpackStore::from_file(tmp))
        .expect("reload int8 on metal");
    let emb_re = embed(&reloaded, &pixels, &device);
    println!(
        "  metal-saved→metal-loaded:        cosine vs fp32 = {:.6}",
        cosine(&emb_fp32, &emb_re)
    );
}

fn base_scheme() -> QuantScheme {
    <QuantizedTensor<Backend> as QTensorPrimitive>::default_scheme().with_value(QuantValue::Q8S)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}
