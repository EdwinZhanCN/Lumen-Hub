pub mod daemon;
pub mod service;

pub mod models {
    #[cfg(feature = "clip")]
    pub mod clip;
    #[cfg(feature = "fastvlm")]
    pub mod fastvlm;
    #[cfg(feature = "ppocr")]
    pub mod ppocr;
    #[cfg(feature = "siglip")]
    pub mod siglip;
}
