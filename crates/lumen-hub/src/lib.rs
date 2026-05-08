pub mod daemon;
pub mod service;

pub mod models {
    #[cfg(feature = "clip")]
    pub mod clip;
    #[cfg(feature = "siglip")]
    pub mod siglip;
}
