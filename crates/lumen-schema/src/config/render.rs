//! Canonical Lumen config rendering.
//!
//! Every entry point that produces a runtime config from a preset or a custom
//! selection (native launcher, Docker env resolver, CLI, and the fixture
//! generator) must end up with the same `LumenConfig`. This module is that
//! single implementation: callers pass a preset or a custom selection plus
//! environment options and receive a validated config.

use super::{
    BatchingConfig, Deployment, LumenConfig, Metadata, Mode, ModelConfig, Region, Runtime,
    ServerConfig, ServiceConfig, ServiceName,
};
use crate::preset::{
    BIOCLIP_DEFAULT_MODEL, FACE_DEFAULT_MODEL, OCR_DEFAULT_MODEL, Preset, SERVICE_ORDER,
    service_package,
};

/// Environment options that differ per machine/entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderOptions<'a> {
    /// `other` or `cn`.
    pub region: &'a str,
    /// Absolute model cache directory.
    pub cache_dir: &'a str,
    /// Config metadata version written into the rendered config.
    pub metadata_version: &'a str,
    /// Runtime string for every model, currently always `burn`.
    pub runtime: &'a str,
    /// Precision string for every model, currently always `fp16q8`.
    pub precision: &'a str,
}

/// Render the official preset selection as a complete validated config.
pub fn preset_config(preset: Preset, options: &RenderOptions<'_>) -> Result<LumenConfig, String> {
    build_config(
        preset.components,
        Some(preset.siglip_model),
        preset.bioclip_dataset,
        options,
    )
}

/// Render a custom selection (at least one capability) as a complete
/// validated config. `siglip_model` defaults to the base model when omitted.
pub fn custom_config(
    services: &[&str],
    siglip_model: Option<&str>,
    bioclip_dataset: Option<&str>,
    options: &RenderOptions<'_>,
) -> Result<LumenConfig, String> {
    build_config(services, siglip_model, bioclip_dataset, options)
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
    if options.runtime != "burn" {
        return Err(format!(
            "unsupported runtime `{}`; the only supported runtime is `burn`",
            options.runtime
        ));
    }
    if options.precision != "fp16q8" {
        return Err(format!(
            "unsupported precision `{}`; the only supported precision is `fp16q8`",
            options.precision
        ));
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
    let siglip_model = siglip_model.unwrap_or(crate::preset::SIGLIP_BASE_MODEL);

    let mut service_configs = std::collections::BTreeMap::new();
    for service in SERVICE_ORDER {
        let enabled = services.contains(&service);
        let model_config = match service {
            "siglip" => ModelConfig {
                model: siglip_model.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(options.precision.to_owned()),
            },
            "face" => ModelConfig {
                model: FACE_DEFAULT_MODEL.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(options.precision.to_owned()),
            },
            "ocr" => ModelConfig {
                model: OCR_DEFAULT_MODEL.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(options.precision.to_owned()),
            },
            "bioclip" => ModelConfig {
                model: BIOCLIP_DEFAULT_MODEL.to_owned(),
                runtime: Runtime::Burn,
                dataset: bioclip_dataset.map(str::to_owned),
                precision: Some(options.precision.to_owned()),
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
            version: options.metadata_version.to_owned(),
            region,
            cache_dir: options.cache_dir.to_owned(),
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
            host: "0.0.0.0".to_owned(),
            port: 50051,
            mdns: Default::default(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{BIOCLIP_FULL_DATASET, Preset, SIGLIP_BASE_MODEL, SIGLIP_BRAVE_MODEL};

    fn options<'a>() -> RenderOptions<'a> {
        RenderOptions {
            region: "other",
            cache_dir: "/models",
            metadata_version: "0.1.0",
            runtime: "burn",
            precision: "fp16q8",
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
    fn rejects_empty_or_unknown_service_selections() {
        assert!(custom_config(&[], None, None, &options()).is_err());
        assert!(custom_config(&["gpu"], None, None, &options()).is_err());
    }

    #[test]
    fn rejects_unsupported_runtime_and_region() {
        let mut bad_runtime = options();
        bad_runtime.runtime = "onnx";
        assert!(preset_config(Preset::by_name("basic").unwrap(), &bad_runtime).is_err());

        let mut bad_region = options();
        bad_region.region = "eu";
        assert!(preset_config(Preset::by_name("basic").unwrap(), &bad_region).is_err());
    }

    #[test]
    fn cn_region_renders_without_changing_service_catalog() {
        let mut cn = options();
        cn.region = "cn";
        let config = preset_config(Preset::by_name("basic").unwrap(), &cn).unwrap();
        assert_eq!(config.metadata.region, Region::Cn);
        assert_eq!(config.deployment_service_names(), SERVICE_ORDER);
    }
}
