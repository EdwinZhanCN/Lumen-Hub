//! Compile-time Burn backend selection.
//!
//! lumen-hub runs every model through [Burn](https://burn.dev). The concrete
//! backend is chosen at build time via cargo features. Exactly one backend is
//! active per build; when several features are enabled the highest-priority one
//! wins so that, for example, `--features metal` overrides the default `cpu`.
//!
//! Priority (high to low): `cuda` > `rocm` > `vulkan` > `metal` > `wgpu` > `cpu`.

#[cfg(feature = "cuda")]
pub type Backend = burn::backend::Cuda;
#[cfg(feature = "cuda")]
pub const BACKEND_NAME: &str = "cuda";

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub type Backend = burn::backend::Rocm;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub const BACKEND_NAME: &str = "rocm";

#[cfg(all(feature = "vulkan", not(any(feature = "cuda", feature = "rocm"))))]
pub type Backend = burn::backend::Vulkan;
#[cfg(all(feature = "vulkan", not(any(feature = "cuda", feature = "rocm"))))]
pub const BACKEND_NAME: &str = "vulkan";

#[cfg(all(
    feature = "metal",
    not(any(feature = "cuda", feature = "rocm", feature = "vulkan"))
))]
pub type Backend = burn::backend::Metal;
#[cfg(all(
    feature = "metal",
    not(any(feature = "cuda", feature = "rocm", feature = "vulkan"))
))]
pub const BACKEND_NAME: &str = "metal";

#[cfg(all(
    feature = "wgpu",
    not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal"
    ))
))]
pub type Backend = burn::backend::Wgpu;
#[cfg(all(
    feature = "wgpu",
    not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal"
    ))
))]
pub const BACKEND_NAME: &str = "wgpu";

#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
)))]
pub type Backend = burn::backend::Flex;
#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
)))]
pub const BACKEND_NAME: &str = "cpu";

/// Device type for the selected backend.
pub type Device = burn::tensor::Device<Backend>;

/// Configures backend runtime defaults before any device/server is created.
pub fn configure_runtime() {
    configure_cubecl_runtime();
    configure_wgpu_runtime();
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
))]
fn configure_cubecl_runtime() {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut config = CubeClRuntimeConfig::default();
    // Each CubeCL stream owns a separate memory pool; keep inference on one pool.
    // Experiment escape hatch: LUMEN_GPU_MAX_STREAMS overrides for A/B runs.
    config.streaming.max_streams = std::env::var("LUMEN_GPU_MAX_STREAMS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    CubeClRuntimeConfig::set(config);
}

#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu"
)))]
fn configure_cubecl_runtime() {}

#[cfg(all(feature = "vulkan", not(any(feature = "cuda", feature = "rocm"))))]
fn configure_wgpu_runtime() {
    configure_wgpu_runtime_for::<burn::backend::wgpu::graphics::Vulkan>();
}

#[cfg(all(
    feature = "metal",
    not(any(feature = "cuda", feature = "rocm", feature = "vulkan"))
))]
fn configure_wgpu_runtime() {
    configure_wgpu_runtime_for::<burn::backend::wgpu::graphics::Metal>();
}

#[cfg(all(
    feature = "wgpu",
    not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal"
    ))
))]
fn configure_wgpu_runtime() {
    configure_wgpu_runtime_for::<burn::backend::wgpu::graphics::AutoGraphicsApi>();
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    not(any(feature = "vulkan", feature = "metal", feature = "wgpu"))
))]
fn configure_wgpu_runtime() {}

#[cfg(any(feature = "vulkan", feature = "metal", feature = "wgpu"))]
fn configure_wgpu_runtime_for<G>()
where
    G: burn::backend::wgpu::graphics::GraphicsApi,
{
    use burn::backend::wgpu::{MemoryConfiguration, RuntimeOptions, init_setup};

    // Experiment escape hatch: LUMEN_GPU_MEMORY_STRATEGY=subslices restores the
    // CubeCL default sliced pools for A/B runs against ExclusivePages.
    let memory_config = match std::env::var("LUMEN_GPU_MEMORY_STRATEGY").as_deref() {
        Ok("subslices") => MemoryConfiguration::SubSlices,
        _ => MemoryConfiguration::ExclusivePages,
    };
    let device = default_device();
    let options = RuntimeOptions {
        memory_config,
        ..RuntimeOptions::default()
    };
    let _ = init_setup::<G>(&device, options);
}

/// Returns the default device for the selected backend.
pub fn default_device() -> Device {
    Device::default()
}

/// Releases cached backend memory that is no longer referenced by live tensors.
///
/// GPU backends such as Metal route this to CubeCL's memory manager. CPU
/// backends currently treat it as a no-op.
pub fn cleanup_memory(device: &Device) {
    <Backend as burn::prelude::Backend>::memory_cleanup(device);
    let _ = <Backend as burn::prelude::Backend>::sync(device);
}
