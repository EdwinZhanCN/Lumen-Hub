//! Minimal reproducer: does burnpack save→reload preserve a per-block-quantized
//! weight? One Linear (2D) + one Conv2d (4D), quantized in-memory, saved, reloaded,
//! and compared element-wise — no inference, no model, no confounds.
//!
//! Run on CPU:   cargo run --example repro_quant_roundtrip --no-default-features --features cpu
//! Run on Metal: cargo run --example repro_quant_roundtrip --no-default-features --features metal

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::ops::QuantizedTensor;
use burn::tensor::quantization::{QTensorPrimitive, QuantLevel, QuantScheme, QuantValue};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};

use lumen_hub::backend::{Backend, default_device};

#[derive(Module, Debug)]
struct Tiny<B: BackendTrait> {
    lin: Linear<B>,
    conv: Conv2d<B>,
}

/// Quantizes Linear/Conv weights (rank >= 2) per-block, like the real toolchain.
struct Q {
    scheme: QuantScheme,
    stack: Vec<bool>,
}
impl<B: BackendTrait> ModuleMapper<B> for Q {
    fn enter_module(&mut self, _n: &str, ct: &str) {
        self.stack
            .push(matches!(ct, "Struct:Linear" | "Struct:Conv2d"));
    }
    fn exit_module(&mut self, _n: &str, _ct: &str) {
        self.stack.pop();
    }
    fn map_float<const D: usize>(&mut self, p: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D >= 2 && *self.stack.last().unwrap_or(&false) {
            let last = p.val().dims()[D - 1];
            let b = (2..=last.min(32))
                .rev()
                .find(|d| last % d == 0)
                .unwrap_or(1) as u8;
            let scheme = self.scheme.with_level(QuantLevel::block([b]));
            p.map(|t| t.quantize_dynamic(&scheme))
        } else {
            p
        }
    }
}

fn maxdiff<const D: usize>(a: &Tensor<Backend, D>, b: &Tensor<Backend, D>) -> f32 {
    (a.clone() - b.clone())
        .abs()
        .max()
        .into_scalar()
        .elem::<f32>()
}
fn f32vec<const D: usize>(t: Tensor<Backend, D>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

fn main() {
    let device = default_device();
    println!("backend = {}", lumen_hub::backend::BACKEND_NAME);

    // Deterministic weights.
    let tiny = Tiny::<Backend> {
        lin: LinearConfig::new(768, 768).with_bias(false).init(&device),
        conv: Conv2dConfig::new([32, 64], [3, 3]).init(&device),
    };
    let lin0 = tiny.lin.weight.val();
    let conv0 = tiny.conv.weight.val();

    let scheme = <QuantizedTensor<Backend> as QTensorPrimitive>::default_scheme()
        .with_value(QuantValue::Q8S);
    let tq = tiny.map(&mut Q {
        scheme,
        stack: vec![],
    });

    // In-memory quantized → dequantized (the "correct" target).
    let lin_mem = tq.lin.weight.val();
    let conv_mem = tq.conv.weight.val();
    println!(
        "in-memory dtypes: lin={:?} conv={:?}",
        lin_mem.dtype(),
        conv_mem.dtype()
    );
    let lin_mem_f = lin_mem.dequantize();
    let conv_mem_f = conv_mem.dequantize();

    // Save → reload.
    let path = "/tmp/tiny_quant.bpk";
    let mut store = BurnpackStore::from_file(path).overwrite(true);
    tq.save_into(&mut store).expect("save");
    let mut tr = Tiny::<Backend> {
        lin: LinearConfig::new(768, 768).with_bias(false).init(&device),
        conv: Conv2dConfig::new([32, 64], [3, 3]).init(&device),
    };
    tr.load_from(&mut BurnpackStore::from_file(path))
        .expect("reload");
    let lin_re = tr.lin.weight.val();
    let conv_re = tr.conv.weight.val();
    println!(
        "reloaded  dtypes: lin={:?} conv={:?}",
        lin_re.dtype(),
        conv_re.dtype()
    );
    let lin_re_f = lin_re.dequantize();
    let conv_re_f = conv_re.dequantize();

    println!("\n--- per-weight max|Δ| (lower = preserved) ---");
    println!(
        "LINEAR : orig-vs-inmem={:.5}  inmem-vs-reloaded={:.5}  orig-vs-reloaded={:.5}",
        maxdiff(&lin0, &lin_mem_f),
        maxdiff(&lin_mem_f, &lin_re_f),
        maxdiff(&lin0, &lin_re_f),
    );
    println!(
        "CONV   : orig-vs-inmem={:.5}  inmem-vs-reloaded={:.5}  orig-vs-reloaded={:.5}",
        maxdiff(&conv0, &conv_mem_f),
        maxdiff(&conv_mem_f, &conv_re_f),
        maxdiff(&conv0, &conv_re_f),
    );

    let cos = |a: &Tensor<Backend, 2>, b: &Tensor<Backend, 2>| {
        let (x, y) = (f32vec(a.clone()), f32vec(b.clone()));
        let d: f32 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
        let nx: f32 = x.iter().map(|p| p * p).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|p| p * p).sum::<f32>().sqrt();
        d / (nx * ny + 1e-12)
    };
    println!(
        "\nLINEAR weight cosine (inmem vs reloaded) = {:.6}",
        cos(&lin_mem_f, &lin_re_f)
    );
}
