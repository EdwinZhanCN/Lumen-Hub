//! Burn model architectures generated from the upstream ONNX graphs by
//! `burn-import`. Each module exposes a `Model<B: Backend>` with:
//!
//! - `Model::from_file(path, device)` — load weights from a `.bpk` burnpack file
//! - `Model::forward(...)` — run inference
//!
//! These files are machine-generated and intentionally left close to their
//! generated form; the hand-written model services in `crate::models` wrap them
//! with preprocessing and postprocessing.
#![allow(
    clippy::all,
    clippy::pedantic,
    dead_code,
    non_snake_case,
    unused_imports
)]

use burn::nn::conv::Conv2d;
use burn::prelude::*;
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use lumen_quant_core::{QuantConfig, RuntimeQ8Quantizer};

/// Convolution forward that works around a burn-wgpu correctness bug in the
/// 1×1 (pointwise) convolution kernel.
///
/// On the wgpu/Metal backends, `Conv2d::forward` produces incorrect results for
/// 1×1 pointwise convolutions (verified: ~1.8× too large), which collapses the
/// PP-LCNet-based PP-OCR models. A 1×1 convolution is mathematically a matmul
/// over the channel dimension, and matmul is correct on every backend, so we
/// route pointwise convs through matmul. All other convolutions use the normal
/// module forward.
pub fn conv_fwd<B: Backend>(conv: &Conv2d<B>, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, ci, h, w] = x.dims();
    let weight = conv.weight.val();
    let [co, ci_g, kh, kw] = weight.dims();
    let pointwise = kh == 1
        && kw == 1
        && ci_g == ci
        && conv.groups == 1
        && conv.stride == [1, 1]
        && conv.dilation == [1, 1];
    if pointwise && n == 1 {
        // [Co, Ci] x [Ci, H*W] -> [Co, H*W] (plain 2D matmul; n == 1 in serving).
        let wm = weight.reshape([co, ci]);
        let xm = x.reshape([ci, h * w]);
        let out = wm.matmul(xm).reshape([1, co, h, w]);
        return match &conv.bias {
            Some(bias) => out + bias.val().reshape([1, co, 1, 1]),
            None => out,
        };
    }
    conv.forward(x)
}

/// Returns `true` when a configured precision tag denotes fp16 weight storage.
pub fn is_fp16_precision(precision: &str) -> bool {
    precision.eq_ignore_ascii_case("fp16") || precision.eq_ignore_ascii_case("fp16q8")
}

/// Returns `true` when fp16 weights should be quantized to Q8 after loading.
pub fn is_runtime_q8_precision(precision: &str) -> bool {
    precision.eq_ignore_ascii_case("fp16q8")
}

/// Loads burnpack weights into a freshly-constructed module.
///
/// The generated `Model::from_file` reads each tensor at its on-disk dtype, which
/// is correct for fp32 artifacts. An fp16 artifact (produced by Burn-Convert's
/// `convert_weights` via [`HalfPrecisionAdapter`]) instead stores its
/// weight-bearing modules (Linear/Conv/Norm/Embedding/PRelu) as f16 while the
/// model graph's synthesized constants and the service's input tensors stay f32 —
/// mixing the two panics inside burn-ir with `DTypeMismatch`.
///
/// For fp16 storage we attach the same [`HalfPrecisionAdapter`] on load: it casts
/// the f16 weight modules back to f32 (leaving the already-f32 constants alone), so
/// the loaded module is uniformly f32. fp16 is therefore a storage/transfer saving
/// here, not f16 compute.
///
/// `fp16q8` uses the same on-disk fp16 burnpack format, then quantizes eligible
/// weights in memory to Q8. This avoids saving/reloading QFloat burnpacks, which
/// is currently not reliable on CubeCL/Metal.
pub fn load_burnpack<B: Backend, M: ModuleSnapshot<B> + Module<B>>(
    mut model: M,
    path: &str,
    precision: &str,
) -> Result<M, String> {
    let mut store = BurnpackStore::from_file(path);
    if is_fp16_precision(precision) {
        store = store.with_from_adapter(HalfPrecisionAdapter::new());
    }
    model
        .load_from(&mut store)
        .map_err(|err| format!("failed to load burnpack weights from `{path}`: {err}"))?;
    if is_runtime_q8_precision(precision) {
        model = quantize_runtime_q8(model);
    }
    Ok(model)
}

fn quantize_runtime_q8<B: Backend, M: Module<B>>(model: M) -> M {
    model.map(&mut RuntimeQ8Quantizer::<B>::new(&QuantConfig::default()))
}

// Each submodule directory holds the generated architecture for one model
// repository (named after its `model_info.json` `name`). Adding a new model
// variant — e.g. a `siglip2_so400m_patch14_384` or a `buffalo_l` face pack —
// means dropping in a new directory here and registering it in the matching
// `models::<family>::model` dispatcher. Model *precision* (fp32/fp16) reuses
// the same module; only the architecture warrants new generated code.

#[cfg(feature = "clip")]
pub mod bioclip2;

#[cfg(feature = "siglip")]
pub mod siglip2_base_patch16_224;

#[cfg(feature = "siglip")]
pub mod siglip2_so400m_patch14_384;

#[cfg(feature = "ppocr")]
pub mod pp_ocrv5;

#[cfg(feature = "ppocr")]
pub mod pp_ocrv5_server;

#[cfg(feature = "ppocr")]
pub mod pp_ocrv6_small;

#[cfg(feature = "insightface")]
pub mod antelopev2;
