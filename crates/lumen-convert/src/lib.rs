//! Selective weight-only int8 quantization for Burn modules.
//!
//! This crate re-exports the shared quantization primitives from
//! `lumen-quant-core` and adds the server model architecture bindings
//! generated at build time from ONNX.

pub mod server;

pub use lumen_quant_core::{
    QuantConfig, RuntimeQ8Quantizer, SelectiveQuantizer, WeightKind, block_size_for, cosine,
    is_quant_store_packable, q8_scheme,
};
