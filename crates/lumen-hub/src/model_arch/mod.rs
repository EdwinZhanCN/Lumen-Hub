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

#[cfg(feature = "insightface")]
pub mod antelopev2;
