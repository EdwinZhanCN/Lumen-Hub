//! Canonical Lumen config rendering.
//!
//! Every managed entry point supplies only deployment intent. This module owns
//! the runtime semantics: service definitions, models, precision, server
//! defaults, and validation.

use super::{
    BatchingConfig, Deployment, LumenConfig, Mdns, Metadata, Mode, ModelConfig, Region, Runtime,
    ServerConfig, ServiceConfig, ServiceName,
};
use crate::preset::{
    BIOCLIP_DATASETS, BIOCLIP_DEFAULT_MODEL, FACE_DEFAULT_MODEL, OCR_DEFAULT_MODEL, Preset,
    SERVICE_ORDER, SIGLIP_MODELS, service_package,
};

pub const CONFIG_VERSION: &str = "0.1.0";
pub const MODEL_PRECISION: &str = "fp16q8";

/// The deployment boundary determines network exposure. It does not alter
/// model or service semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigTarget {
    /// Bind on all interfaces for Docker and standalone LAN deployments.
    Network,
    /// Bind only on loopback for an application-managed local child process.
    Desktop,
}

impl ConfigTarget {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "network" => Ok(Self::Network),
            "desktop" => Ok(Self::Desktop),
            other => Err(format!(
                "unsupported config target `{other}`; expected `network` or `desktop`"
            )),
        }
    }

    fn host(self) -> &'static str {
        match self {
            Self::Network => "0.0.0.0",
            Self::Desktop => "127.0.0.1",
        }
    }

    fn mdns(self) -> Mdns {
        Mdns {
            enabled: matches!(self, Self::Network),
            service_name: None,
        }
    }
}

/// Machine-specific intent accepted by the canonical renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderOptions<'a> {
    /// `other` or `cn`.
    pub region: &'a str,
    /// Absolute model cache directory.
    pub cache_dir: &'a str,
    /// Network exposure owned by the installation path.
    pub target: ConfigTarget,
}

/// Render an official preset as a complete validated config.
pub fn preset_config(preset: Preset, options: &RenderOptions<'_>) -> Result<LumenConfig, String> {
    build_config(
        preset.components,
        Some(preset.siglip_model),
        preset.bioclip_dataset,
        options,
    )
}

/// Render a custom selection as a complete validated config.
/// `siglip_model` defaults to the base model when omitted.
pub fn custom_config(
    services: &[&str],
    siglip_model: Option<&str>,
    bioclip_dataset: Option<&str>,
    options: &RenderOptions<'_>,
) -> Result<LumenConfig, String> {
    build_config(services, siglip_model, bioclip_dataset, options)
}

/// Serialize a validated config in the one format emitted by managed entry
/// points. Hand-written configuration remains supported by the daemon parser.
pub fn to_yaml(config: &LumenConfig) -> Result<String, String> {
    config
        .validate_config()
        .map_err(|error| format!("config failed validation: {error}"))?;
    let mut yaml = serde_yaml::to_string(config)
        .map_err(|error| format!("failed to serialize config: {error}"))?;
    if !yaml.ends_with('\n') {
        yaml.push('\n');
    }
    Ok(yaml)
}

pub fn preset_yaml(preset: Preset, options: &RenderOptions<'_>) -> Result<String, String> {
    to_yaml(&preset_config(preset, options)?)
}

pub fn custom_yaml(
    services: &[&str],
    siglip_model: Option<&str>,
    bioclip_dataset: Option<&str>,
    options: &RenderOptions<'_>,
) -> Result<String, String> {
    to_yaml(&custom_config(
        services,
        siglip_model,
        bioclip_dataset,
        options,
    )?)
}

fn build_config(
    services: &[&str],
    siglip_model: Option<&str>,
    bioclip_dataset: Option<&str>,
    options: &RenderOptions<'_>,
) -> Result<LumenConfig, String> {
    if services.is_empty() {
        return Err("a config must select at least one capability service".to_owned());
    }
    for service in services {
        if !SERVICE_ORDER.contains(service) {
            return Err(format!(
                "unknown capability service `{service}`; expected one of: {}",
                SERVICE_ORDER.join(", ")
            ));
        }
    }

    let region = match options.region {
        "cn" => Region::Cn,
        "other" => Region::Other,
        other => {
            return Err(format!(
                "unsupported region `{other}`; expected `other` or `cn`"
            ));
        }
    };
    let cache_dir = options.cache_dir.trim();
    if cache_dir.is_empty() {
        return Err("cache directory must not be empty".to_owned());
    }
    if !is_absolute_cache_dir(cache_dir) {
        return Err(format!("cache directory `{cache_dir}` must be absolute"));
    }

    let siglip_model = siglip_model.unwrap_or(crate::preset::SIGLIP_BASE_MODEL);
    if !SIGLIP_MODELS.contains(&siglip_model) {
        return Err(format!(
            "unsupported SigLIP model `{siglip_model}`; expected {}",
            SIGLIP_MODELS.join(" or ")
        ));
    }
    if let Some(dataset) = bioclip_dataset
        && !BIOCLIP_DATASETS.contains(&dataset)
    {
        return Err(format!(
            "unsupported BioCLIP dataset `{dataset}`; expected {}",
            BIOCLIP_DATASETS.join(" or ")
        ));
    }
    if services.contains(&"bioclip") && bioclip_dataset.is_none() {
        return Err("BioCLIP requires an explicit dataset".to_owned());
    }
    if !services.contains(&"bioclip") && bioclip_dataset.is_some() {
        return Err("a BioCLIP dataset was supplied while the service is disabled".to_owned());
    }

    let mut service_configs = std::collections::BTreeMap::new();
    for service in SERVICE_ORDER {
        let enabled = services.contains(&service);
        let model_config = match service {
            "siglip" => ModelConfig {
                model: siglip_model.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(MODEL_PRECISION.to_owned()),
            },
            "face" => ModelConfig {
                model: FACE_DEFAULT_MODEL.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(MODEL_PRECISION.to_owned()),
            },
            "ocr" => ModelConfig {
                model: OCR_DEFAULT_MODEL.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(MODEL_PRECISION.to_owned()),
            },
            "bioclip" => ModelConfig {
                model: BIOCLIP_DEFAULT_MODEL.to_owned(),
                runtime: Runtime::Burn,
                dataset: bioclip_dataset.map(str::to_owned),
                precision: Some(MODEL_PRECISION.to_owned()),
            },
            _ => unreachable!("SERVICE_ORDER is the fixed four-service catalog"),
        };
        let mut models = std::collections::BTreeMap::new();
        models.insert("default".to_owned(), model_config);
        service_configs.insert(
            service.to_owned(),
            ServiceConfig {
                enabled,
                package: service_package(service)
                    .expect("service has a package")
                    .to_owned(),
                models,
            },
        );
    }

    let config = LumenConfig {
        metadata: Metadata {
            version: CONFIG_VERSION.to_owned(),
            region,
            cache_dir: cache_dir.to_owned(),
        },
        deployment: Deployment {
            mode: Mode::Hub,
            service: None,
            services: Some(
                services
                    .iter()
                    .map(|service| ServiceName((*service).to_owned()))
                    .collect(),
            ),
        },
        server: ServerConfig {
            host: options.target.host().to_owned(),
            port: 50051,
            mdns: options.target.mdns(),
            batching: BatchingConfig {
                enabled: false,
                max_batch_size: 8,
                queue_latency_ms: 2,
            },
        },
        services: service_configs,
    };

    config
        .validate_config()
        .map_err(|error| format!("rendered config failed validation: {error}"))?;
    Ok(config)
}

fn is_absolute_cache_dir(path: &str) -> bool {
    if std::path::Path::new(path).is_absolute() {
        return true;
    }

    let bytes = path.as_bytes();
    let drive_absolute = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/');
    let unc_absolute = path.strip_prefix(r"\\").is_some_and(|rest| {
        let mut parts = rest.split(['\\', '/']).filter(|part| !part.is_empty());
        parts.next().is_some() && parts.next().is_some()
    });
    drive_absolute || unc_absolute
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{BIOCLIP_FULL_DATASET, Preset, SIGLIP_BASE_MODEL, SIGLIP_BRAVE_MODEL};

    fn options<'a>() -> RenderOptions<'a> {
        RenderOptions {
            region: "other",
            cache_dir: "/models",
            target: ConfigTarget::Network,
        }
    }

    #[test]
    fn minimal_preset_selects_siglip_and_face() {
        let config = preset_config(Preset::by_name("minimal").unwrap(), &options()).unwrap();
        assert_eq!(config.deployment_service_names(), vec!["siglip", "face"]);
        assert_eq!(
            config.services["siglip"].models["default"].model,
            SIGLIP_BASE_MODEL
        );
        assert!(config.services["face"].enabled);
        assert!(!config.services["ocr"].enabled);
        assert!(!config.services["bioclip"].enabled);
    }

    #[test]
    fn brave_preset_selects_large_model_and_full_dataset() {
        let config = preset_config(Preset::by_name("brave").unwrap(), &options()).unwrap();
        assert_eq!(
            config.services["siglip"].models["default"].model,
            SIGLIP_BRAVE_MODEL
        );
        assert_eq!(
            config.services["bioclip"].models["default"]
                .dataset
                .as_deref(),
            Some(BIOCLIP_FULL_DATASET)
        );
    }

    #[test]
    fn rejects_relative_cache_directory() {
        let options = RenderOptions {
            region: "other",
            cache_dir: "models",
            target: ConfigTarget::Desktop,
        };
        let error = preset_config(Preset::by_name("basic").unwrap(), &options).unwrap_err();
        assert!(error.contains("must be absolute"));
    }

    #[test]
    fn accepts_windows_absolute_cache_directory_on_every_host() {
        for cache_dir in [
            r"C:\Users\edwin\.lumen\models",
            r"\\model-server\lumen\models",
        ] {
            let options = RenderOptions {
                region: "other",
                cache_dir,
                target: ConfigTarget::Desktop,
            };
            let config = preset_config(Preset::by_name("minimal").unwrap(), &options).unwrap();
            assert_eq!(config.metadata.cache_dir, options.cache_dir);
        }
    }

    #[test]
    fn rejects_windows_drive_relative_cache_directory() {
        for cache_dir in [r"C:models", r"\models"] {
            let options = RenderOptions {
                region: "other",
                cache_dir,
                target: ConfigTarget::Desktop,
            };
            let error = preset_config(Preset::by_name("minimal").unwrap(), &options).unwrap_err();
            assert!(error.contains("must be absolute"));
        }
    }

    #[test]
    fn desktop_target_is_loopback_only() {
        let mut desktop = options();
        desktop.target = ConfigTarget::Desktop;
        let config = preset_config(Preset::by_name("basic").unwrap(), &desktop).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert!(!config.server.mdns.enabled);
    }

    #[test]
    fn custom_selection_renders_only_selected_services() {
        let config = custom_config(
            &["siglip", "bioclip"],
            Some(SIGLIP_BRAVE_MODEL),
            Some(BIOCLIP_FULL_DATASET),
            &options(),
        )
        .unwrap();
        assert_eq!(config.deployment_service_names(), vec!["siglip", "bioclip"]);
        assert!(!config.services["face"].enabled);
    }

    #[test]
    fn rejects_invalid_custom_intent() {
        assert!(custom_config(&[], None, None, &options()).is_err());
        assert!(custom_config(&["gpu"], None, None, &options()).is_err());
        assert!(custom_config(&["siglip"], Some("future-model"), None, &options()).is_err());
        assert!(custom_config(&["bioclip"], None, None, &options()).is_err());
        assert!(custom_config(&["siglip"], None, Some(BIOCLIP_FULL_DATASET), &options()).is_err());
    }

    #[test]
    fn rejects_unsupported_region_and_target() {
        let mut bad_region = options();
        bad_region.region = "eu";
        assert!(preset_config(Preset::by_name("basic").unwrap(), &bad_region).is_err());
        assert!(ConfigTarget::parse("container").is_err());
    }

    #[test]
    fn yaml_round_trip_preserves_the_canonical_config() {
        let config = preset_config(Preset::by_name("basic").unwrap(), &options()).unwrap();
        let yaml = to_yaml(&config).unwrap();
        let parsed: LumenConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, config);
    }
}
