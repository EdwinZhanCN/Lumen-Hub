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
pub type Backend = burn::backend::NdArray<f32>;
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

/// Returns the default device for the selected backend.
pub fn default_device() -> Device {
    Device::default()
}
