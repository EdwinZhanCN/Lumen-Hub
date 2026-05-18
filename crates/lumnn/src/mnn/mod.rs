//! MNN (Mobile Neural Network) inference backend.
//!
//! This module provides [`MnnNode`], which executes MNN-format models via the
//! `lumnn-mnn-sys` FFI crate. It implements the [`MLNode`] trait, making it
//! composable in [`MLPipeline`](crate::core::pipeline::MLPipeline) graphs.
//!
//! # Backend selection
//!
//! The inference backend is chosen at compile time through Cargo features:
//!
//! | Feature         | Backend       | Platform          |
//! |-----------------|---------------|-------------------|
//! | `mnn-vulkan`    | Vulkan        | Cross-platform    |
//! | `mnn-cuda`      | CUDA          | NVIDIA GPU only   |
//! | `mnn-metal`     | Metal         | Apple only        |
//! | `mnn-opencl`    | OpenCL        | Cross-platform    |
//! | `mnn-opengl`    | OpenGL        | Limited           |
//! | `mnn-coreml`    | CoreML        | Apple only        |
//!
//! When [`MLContext::accelerated`](crate::core::context::MLContext::accelerated)
//! is `false`, all features are ignored and the CPU backend is used.
//!
//! # Data format
//!
//! MnnNode exposes the named model inputs and outputs reported by MNN. It
//! currently supports `Float32`, `Int32`, and `Int64` host tensors.

pub mod convert;
pub mod node;

pub use node::MnnNode;
