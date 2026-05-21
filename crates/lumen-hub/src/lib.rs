pub mod daemon;
pub mod model_download;
pub mod service;
pub mod warmup;

pub mod models {
    #[cfg(feature = "clip")]
    pub mod clip;
    #[cfg(feature = "insightface")]
    pub mod insightface;
    #[cfg(feature = "ppocr")]
    pub mod ppocr;
    #[cfg(feature = "siglip")]
    pub mod siglip;
}
