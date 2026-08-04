//! Canonical Lumen capability catalog and deployment presets.
//!
//! Docker, the native launcher, and every user-facing configurator must read
//! these definitions instead of maintaining their own service/model tables.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityTerm {
    pub service: &'static str,
    pub zh_cn: &'static str,
    pub en: &'static str,
}

pub const CAPABILITIES: [CapabilityTerm; 4] = [
    CapabilityTerm {
        service: "siglip",
        zh_cn: "图像语义分析",
        en: "Image Semantic Analysis",
    },
    CapabilityTerm {
        service: "face",
        zh_cn: "人物识别",
        en: "Person Recognition",
    },
    CapabilityTerm {
        service: "ocr",
        zh_cn: "OCR文字识别",
        en: "OCR Text Recognition",
    },
    CapabilityTerm {
        service: "bioclip",
        zh_cn: "BioCLIP物种识别",
        en: "BioCLIP Species Recognition",
    },
];

pub const SERVICE_ORDER: [&str; 4] = ["siglip", "face", "ocr", "bioclip"];

pub const SIGLIP_BASE_MODEL: &str = "siglip2-base-patch16-224";
pub const SIGLIP_BRAVE_MODEL: &str = "siglip2-so400m-patch14-384";
pub const FACE_DEFAULT_MODEL: &str = "antelopev2";
pub const OCR_DEFAULT_MODEL: &str = "pp-ocrv6-small";
pub const BIOCLIP_DEFAULT_MODEL: &str = "bioclip-2";
pub const BIOCLIP_CORE_DATASET: &str = "TreeOfLife200MCore";
pub const BIOCLIP_FULL_DATASET: &str = "TreeOfLife200M";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preset {
    pub name: &'static str,
    pub components: &'static [&'static str],
    pub siglip_model: &'static str,
    pub bioclip_dataset: Option<&'static str>,
    pub min_ram_gb: u64,
    pub min_vram_gb: u64,
    pub min_disk_gb: u64,
}

const PRESETS: [Preset; 3] = [
    // RAM/VRAM/disk are measured guidance (Apple M2 Pro, Metal, fp16q8).
    // Weights and BioCLIP catalogs are memory-mapped, so model size lands on
    // disk and cold faults rather than becoming permanently resident RAM.
    Preset {
        name: "minimal",
        components: &["siglip", "face"],
        siglip_model: SIGLIP_BASE_MODEL,
        bioclip_dataset: None,
        min_ram_gb: 4,
        min_vram_gb: 2,
        min_disk_gb: 2,
    },
    Preset {
        name: "basic",
        components: &SERVICE_ORDER,
        siglip_model: SIGLIP_BASE_MODEL,
        bioclip_dataset: Some(BIOCLIP_CORE_DATASET),
        min_ram_gb: 6,
        min_vram_gb: 3,
        min_disk_gb: 6,
    },
    Preset {
        name: "brave",
        components: &SERVICE_ORDER,
        siglip_model: SIGLIP_BRAVE_MODEL,
        bioclip_dataset: Some(BIOCLIP_FULL_DATASET),
        min_ram_gb: 8,
        min_vram_gb: 4,
        min_disk_gb: 10,
    },
];

impl Preset {
    pub fn all() -> &'static [Self] {
        &PRESETS
    }

    pub fn by_name(name: &str) -> Option<Self> {
        PRESETS.iter().copied().find(|preset| preset.name == name)
    }

    pub fn includes(self, component: &str) -> bool {
        self.components.contains(&component)
    }

    pub fn display_title(self) -> &'static str {
        match self.name {
            "minimal" => "minimal (最小)",
            "basic" => "basic (基础)",
            "brave" => "brave (激进)",
            _ => self.name,
        }
    }

    pub fn label(self) -> String {
        let capabilities = self
            .components
            .iter()
            .filter_map(|service| capability_term(service))
            .map(|term| format!("{} / {}", term.zh_cn, term.en))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{} — {} — RAM {} GB, GPU/Unified {} GB",
            self.display_title(),
            capabilities,
            self.min_ram_gb,
            self.min_vram_gb
        )
    }
}

pub fn capability_term(service: &str) -> Option<&'static CapabilityTerm> {
    CAPABILITIES.iter().find(|term| term.service == service)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_reference_the_canonical_capability_catalog() {
        for preset in Preset::all() {
            for service in preset.components {
                assert!(
                    capability_term(service).is_some(),
                    "preset {} references unknown service {service}",
                    preset.name
                );
            }
        }
    }

    #[test]
    fn capability_terms_are_exact() {
        assert_eq!(CAPABILITIES[0].zh_cn, "图像语义分析");
        assert_eq!(CAPABILITIES[0].en, "Image Semantic Analysis");
        assert_eq!(CAPABILITIES[1].zh_cn, "人物识别");
        assert_eq!(CAPABILITIES[1].en, "Person Recognition");
        assert_eq!(CAPABILITIES[2].zh_cn, "OCR文字识别");
        assert_eq!(CAPABILITIES[2].en, "OCR Text Recognition");
        assert_eq!(CAPABILITIES[3].zh_cn, "BioCLIP物种识别");
        assert_eq!(CAPABILITIES[3].en, "BioCLIP Species Recognition");
    }
}
