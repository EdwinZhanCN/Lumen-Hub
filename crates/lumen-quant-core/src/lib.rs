//! Shared selective weight-only int8 quantization primitives.
//!
//! Both `lumen-hub` (runtime fp16q8 quantization after loading) and
//! `lumen-convert` (offline artifact generation) use these building blocks so
//! the quantization policy stays in sync across both paths.

use burn::module::{ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::ops::QuantizedTensor;
use burn::tensor::quantization::{
    QTensorPrimitive, QuantLevel, QuantScheme, QuantStore, QuantValue,
};
use burn::tensor::Tensor;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quantization granularity / dtype configuration.
#[derive(Clone, Copy, Debug)]
pub struct QuantConfig {
    /// `Some(cap)` quantizes per-block along the last axis with block size = the
    /// largest divisor of the last dim that is `<= cap` (capped at 255, the u8
    /// block-size limit); `None` quantizes per-tensor. Smaller cap → higher
    /// accuracy, more scale overhead. Default 32 (~0.999 cosine on SigLIP).
    pub block_cap: Option<usize>,
    /// Also quantize bare rank-2 constants with at least this many elements (e.g.
    /// token-embedding tables emitted by burn-import as `Gather` constants).
    ///
    /// DEFAULT `None`: quantizing an embedding requires `q_gather` at inference,
    /// which is `unimplemented!()` in burn-cubecl — so a quantized embedding panics
    /// on every GPU backend (works only on ndarray CPU). Since int8's whole point is
    /// GPU, embeddings must stay fp32. Only enable this for CPU-only / disk-size use.
    pub large_const_min_numel: Option<usize>,
    /// Keep `Linear` layers with any dimension above this in fp32. These are
    /// typically the flatten→embedding output projection of a CNN embedding model
    /// (e.g. ArcFace's 25088→512): a huge fan-in accumulates int8 noise across all
    /// summed terms, and being the *output* layer it goes straight into the
    /// L2-normalized embedding → the cosine collapses. Standard PTQ practice keeps
    /// the output projection in full precision. Transformer MLPs (≤ a few thousand)
    /// are well below this. `None` disables the rule.
    pub skip_linear_dim_above: Option<usize>,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            block_cap: Some(32),
            large_const_min_numel: None,
            skip_linear_dim_above: Some(8192),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Default symmetric Q8 scheme for backend `B`.
pub fn q8_scheme<B: Backend>() -> QuantScheme {
    <QuantizedTensor<B> as QTensorPrimitive>::default_scheme().with_value(QuantValue::Q8S)
}

/// Largest divisor of `last` in `[2, min(cap, 255)]` that is also a multiple of
/// `multiple` (the packing alignment required by the backend's quant store).
/// Returns `None` when no valid block size exists (caller falls back to per-tensor).
pub fn block_size_for(last: usize, cap: usize, multiple: usize) -> Option<u8> {
    let multiple = multiple.max(1);
    (2..=last.min(cap).min(255))
        .rev()
        .find(|d| last % d == 0 && d % multiple == 0)
        .map(|d| d as u8)
}

/// Returns `true` when the quantized tensor can be stored in the backend's packed
/// format. CubeCL's `PackedU32` packs along the last dimension; if the last dim
/// isn't divisible by the packing factor the tensor must stay in float.
pub fn is_quant_store_packable(scheme: &QuantScheme, dims: &[usize]) -> bool {
    match scheme.store {
        QuantStore::PackedU32(_) => dims
            .last()
            .is_some_and(|last| last % scheme.num_quants() == 0),
        _ => true,
    }
}

/// Cosine similarity between two equal-length embedding vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

// ---------------------------------------------------------------------------
// Weight container classification (shared between runtime and offline paths)
// ---------------------------------------------------------------------------

/// The kind of weight container currently being traversed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightKind {
    /// `Linear` — eligible for the large-fan-in output-projection skip rule.
    Linear,
    /// Conv* — quantized per-block, no skip rule.
    Conv,
    /// Not a weight module.
    None,
}

impl WeightKind {
    pub fn from_container_type(container_type: &str) -> Self {
        match container_type {
            "Struct:Linear" => Self::Linear,
            "Struct:Conv1d"
            | "Struct:Conv2d"
            | "Struct:Conv3d"
            | "Struct:ConvTranspose1d"
            | "Struct:ConvTranspose2d"
            | "Struct:ConvTranspose3d" => Self::Conv,
            _ => Self::None,
        }
    }
}

// ---------------------------------------------------------------------------
// SelectiveQuantizer — offline use (with error tracking)
// ---------------------------------------------------------------------------

/// [`ModuleMapper`] that quantizes only weight-bearing module parameters.
///
/// Maintains a container-type stack via `enter_module`/`exit_module`; a float
/// param is quantized only when its immediate container is a weight module
/// (`Linear`/`Conv*`) and its rank is `>= 2` (which skips 1-D biases, norm params,
/// and BatchNorm running stats).
pub struct SelectiveQuantizer {
    scheme: QuantScheme,
    config: QuantConfig,
    stack: Vec<WeightKind>,
    pub quantized: usize,
    pub skipped: usize,
    err_sum: f32,
    pub err_max: f32,
}

impl SelectiveQuantizer {
    pub fn new(scheme: QuantScheme, config: QuantConfig) -> Self {
        Self {
            scheme,
            config,
            stack: Vec::new(),
            quantized: 0,
            skipped: 0,
            err_sum: 0.0,
            err_max: 0.0,
        }
    }

    pub fn mean_err(&self) -> f32 {
        if self.quantized == 0 {
            0.0
        } else {
            self.err_sum / self.quantized as f32
        }
    }
}

impl<B: Backend> ModuleMapper<B> for SelectiveQuantizer {
    fn enter_module(&mut self, _name: &str, container_type: &str) {
        self.stack.push(WeightKind::from_container_type(container_type));
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        use burn::tensor::ElementConversion;

        if D < 2 {
            self.skipped += 1;
            return param;
        }
        let kind = self.stack.last().copied().unwrap_or(WeightKind::None);
        let in_weight_module = kind != WeightKind::None;
        let original = param.val();
        let dims = original.dims();
        let numel: usize = dims.iter().product();
        let is_large_const = self
            .config
            .large_const_min_numel
            .is_some_and(|min| numel >= min);
        let skip_output_linear = kind == WeightKind::Linear
            && self
                .config
                .skip_linear_dim_above
                .is_some_and(|max| dims.iter().any(|&d| d > max));
        if (in_weight_module || is_large_const) && !skip_output_linear {
            self.quantized += 1;
            let last = original.dims()[D - 1];
            let multiple = self.scheme.num_quants();
            let scheme = if in_weight_module {
                match self
                    .config
                    .block_cap
                    .and_then(|cap| block_size_for(last, cap, multiple))
                {
                    Some(b) => self.scheme.with_level(QuantLevel::block([b])),
                    None => self.scheme.with_level(QuantLevel::Tensor),
                }
            } else {
                self.scheme.with_level(QuantLevel::Tensor)
            };

            if !is_quant_store_packable(&scheme, &dims) {
                self.skipped += 1;
                self.quantized -= 1;
                return param;
            }

            let quant = original.clone().quantize_dynamic(&scheme);
            let deq = quant.clone().dequantize();
            let num: f32 = (original.clone() - deq).abs().sum().into_scalar().elem();
            let den: f32 = original.abs().sum().into_scalar().elem::<f32>() + 1e-12;
            let err = num / den;
            self.err_sum += err;
            self.err_max = self.err_max.max(err);
            param.map(|_| quant)
        } else {
            self.skipped += 1;
            param
        }
    }
}

// ---------------------------------------------------------------------------
// RuntimeQ8Quantizer — lightweight runtime-only quantizer (no error tracking)
// ---------------------------------------------------------------------------

/// Lightweight runtime quantizer that applies Q8 to eligible weights after
/// loading fp16 burnpacks. Same policy as [`SelectiveQuantizer`] but without
/// reconstruction-error tracking (which requires dequantize + extra allocs).
pub struct RuntimeQ8Quantizer<B: Backend> {
    scheme: QuantScheme,
    block_cap: usize,
    skip_linear_dim_above: Option<usize>,
    stack: Vec<WeightKind>,
    _backend: core::marker::PhantomData<B>,
}

impl<B: Backend> RuntimeQ8Quantizer<B> {
    pub fn new(config: &QuantConfig) -> Self {
        Self {
            scheme: q8_scheme::<B>(),
            block_cap: config.block_cap.unwrap_or(32),
            skip_linear_dim_above: config.skip_linear_dim_above,
            stack: Vec::new(),
            _backend: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> ModuleMapper<B> for RuntimeQ8Quantizer<B> {
    fn enter_module(&mut self, _name: &str, container_type: &str) {
        self.stack.push(WeightKind::from_container_type(container_type));
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D < 2 {
            return param;
        }

        let kind = self
            .stack
            .last()
            .copied()
            .unwrap_or(WeightKind::None);
        if kind == WeightKind::None {
            return param;
        }

        let tensor = param.val();
        let dims = tensor.dims();
        let skip_output_linear = kind == WeightKind::Linear
            && self
                .skip_linear_dim_above
                .is_some_and(|max| dims.iter().any(|&dim| dim > max));
        if skip_output_linear {
            return param;
        }

        let last = dims[D - 1];
        let multiple = self.scheme.num_quants();
        let scheme = match block_size_for(last, self.block_cap, multiple) {
            Some(block) => self.scheme.with_level(QuantLevel::block([block])),
            None => self.scheme.with_level(QuantLevel::Tensor),
        };

        if !is_quant_store_packable(&scheme, &dims) {
            return param;
        }

        param.map(|value| value.quantize_dynamic(&scheme))
    }
}
