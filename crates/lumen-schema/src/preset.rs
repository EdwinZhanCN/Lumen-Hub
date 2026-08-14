//! Canonical Lumen capability catalog and deployment presets.
//!
//! Docker, the native launcher, and every user-facing configurator must read
//! these definitions instead of maintaining their own service/model tables.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayLang {
    En,
    ZhCn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityTerm {
    pub service: &'static str,
    pub zh_cn: &'static str,
    pub en: &'static str,
}

impl CapabilityTerm {
    pub fn label(self, lang: DisplayLang) -> &'static str {
        match lang {
            DisplayLang::En => self.en,
            DisplayLang::ZhCn => self.zh_cn,
        }
    }
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

/// Canonical model options per capability service, in display order.
/// The Docker env parser, CLI, and release manifest all read these tables
/// instead of maintaining parallel allow-lists.
pub const SIGLIP_MODELS: [&str; 2] = [SIGLIP_BASE_MODEL, SIGLIP_BRAVE_MODEL];
pub const FACE_MODELS: [&str; 1] = [FACE_DEFAULT_MODEL];
pub const OCR_MODELS: [&str; 1] = [OCR_DEFAULT_MODEL];
pub const BIOCLIP_MODELS: [&str; 1] = [BIOCLIP_DEFAULT_MODEL];

/// Canonical BioCLIP dataset options, in display order.
pub const BIOCLIP_DATASETS: [&str; 2] = [BIOCLIP_CORE_DATASET, BIOCLIP_FULL_DATASET];

/// Model options for a capability service. Unknown services have no options.
pub fn models_for(service: &str) -> Option<&'static [&'static str]> {
    match service {
        "siglip" => Some(&SIGLIP_MODELS),
        "face" => Some(&FACE_MODELS),
        "ocr" => Some(&OCR_MODELS),
        "bioclip" => Some(&BIOCLIP_MODELS),
        _ => None,
    }
}

/// Resolve user-supplied service names to the canonical catalog, in catalog order.
pub fn intern_services<'a>(
    services: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<&'static str>, String> {
    let mut selected = Vec::new();
    for service in services {
        let canonical = SERVICE_ORDER
            .iter()
            .copied()
            .find(|candidate| *candidate == service)
            .ok_or_else(|| {
                format!(
                    "unknown capability service `{service}`; expected one of: {}",
                    SERVICE_ORDER.join(", ")
                )
            })?;
        if selected.contains(&canonical) {
            return Err(format!(
                "capability service `{canonical}` is listed more than once"
            ));
        }
        selected.push(canonical);
    }
    if selected.is_empty() {
        return Err("a config must select at least one capability service".to_owned());
    }
    Ok(SERVICE_ORDER
        .iter()
        .copied()
        .filter(|service| selected.contains(service))
        .collect())
}

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

    pub fn display_title(self, lang: DisplayLang) -> &'static str {
        match (self.name, lang) {
            ("minimal", DisplayLang::En) => "minimal",
            ("minimal", DisplayLang::ZhCn) => "最小",
            ("basic", DisplayLang::En) => "basic",
            ("basic", DisplayLang::ZhCn) => "基础",
            ("brave", DisplayLang::En) => "brave",
            ("brave", DisplayLang::ZhCn) => "激进",
            _ => self.name,
        }
    }

    pub fn capability_summary(self, lang: DisplayLang) -> String {
        self.components
            .iter()
            .filter_map(|service| capability_term(service))
            .map(|term| term.label(lang))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn label(self, lang: DisplayLang) -> String {
        format!(
            "{} — {}",
            self.display_title(lang),
            self.capability_summary(lang)
        )
    }
}

pub fn capability_term(service: &str) -> Option<&'static CapabilityTerm> {
    CAPABILITIES.iter().find(|term| term.service == service)
}

/// Rust service package/crate name backing each capability service.
pub fn service_package(service: &str) -> Option<&'static str> {
    match service {
        "siglip" => Some("siglip"),
        "face" => Some("insightface"),
        "ocr" => Some("ppocr"),
        "bioclip" => Some("clip"),
        _ => None,
    }
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
    fn every_service_has_a_package() {
        for service in SERVICE_ORDER {
            assert!(
                service_package(service).is_some(),
                "service {service} has no package mapping"
            );
        }
    }

    #[test]
    fn every_service_has_model_options() {
        for service in SERVICE_ORDER {
            let models = models_for(service).expect("service has model options");
            assert!(!models.is_empty(), "service {service} has no models");
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

    #[test]
    fn labels_are_single_language() {
        let minimal = Preset::by_name("minimal").unwrap();
        assert_eq!(minimal.display_title(DisplayLang::En), "minimal");
        assert_eq!(minimal.display_title(DisplayLang::ZhCn), "最小");
        assert_eq!(
            CAPABILITIES[0].label(DisplayLang::En),
            "Image Semantic Analysis"
        );
        assert_eq!(CAPABILITIES[0].label(DisplayLang::ZhCn), "图像语义分析");

        let en = minimal.label(DisplayLang::En);
        let zh = minimal.label(DisplayLang::ZhCn);
        assert!(en.contains("Image Semantic Analysis"));
        assert!(en.contains("Person Recognition"));
        assert!(!en.contains("图像语义分析"));
        assert!(!en.contains("最小"));
        assert!(zh.contains("图像语义分析"));
        assert!(zh.contains("人物识别"));
        assert!(!zh.contains("Image Semantic Analysis"));
        assert!(!zh.contains("minimal"));
    }

    #[test]
    fn intern_services_canonicalizes_order_and_rejects_duplicates() {
        assert_eq!(
            intern_services(["bioclip", "siglip"]).unwrap(),
            vec!["siglip", "bioclip"]
        );
        assert!(intern_services(["siglip", "siglip"]).is_err());
        assert!(intern_services(["gpu"]).is_err());
        assert!(intern_services(Vec::<&str>::new()).is_err());
    }
}
