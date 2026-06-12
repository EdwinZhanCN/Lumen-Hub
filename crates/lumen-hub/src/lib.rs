// Burn's generated model code (model_arch) references `alloc::vec::Vec`.
extern crate alloc;

pub mod backend;
pub mod daemon;
pub mod inference_worker;
pub mod model_arch;
pub mod model_download;
pub mod service;
pub mod warmup;

pub mod models {
    #[cfg(feature = "clip")]
    pub mod bioclip;
    #[cfg(feature = "insightface")]
    pub mod insightface;
    #[cfg(feature = "ppocr")]
    pub mod ppocr;
    #[cfg(feature = "siglip")]
    pub mod siglip;
}
