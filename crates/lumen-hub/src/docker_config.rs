//! Strict Docker environment adapter.
//!
//! Docker owns only deployment intent. The canonical schema renderer owns the
//! complete runtime configuration; this module never mutates a base YAML file.

use std::env;

use lumen_schema::{
    BIOCLIP_CORE_DATASET, ConfigTarget, LumenConfig, Preset, RenderOptions, SERVICE_ORDER,
    custom_config, preset_config,
};
use thiserror::Error;

const REGION_VAR: &str = "LUMEN_REGION";
const PRESET_VAR: &str = "LUMEN_PRESET";
const SERVICES_VAR: &str = "LUMEN_SERVICES";
const SIGLIP_MODEL_VAR: &str = "LUMEN_SIGLIP_MODEL";
const BIOCLIP_DATASET_VAR: &str = "LUMEN_BIOCLIP_DATASET";
const ENV_VARS: [&str; 5] = [
    REGION_VAR,
    PRESET_VAR,
    SERVICES_VAR,
    SIGLIP_MODEL_VAR,
    BIOCLIP_DATASET_VAR,
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DockerConfigInput {
    pub region: Option<String>,
    pub preset: Option<String>,
    pub services: Option<String>,
    pub siglip_model: Option<String>,
    pub bioclip_dataset: Option<String>,
}

impl DockerConfigInput {
    pub fn from_process_env() -> Result<Self, DockerConfigError> {
        Ok(Self {
            region: read_env(REGION_VAR)?,
            preset: read_env(PRESET_VAR)?,
            services: read_env(SERVICES_VAR)?,
            siglip_model: read_env(SIGLIP_MODEL_VAR)?,
            bioclip_dataset: read_env(BIOCLIP_DATASET_VAR)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.region.is_none()
            && self.preset.is_none()
            && self.services.is_none()
            && self.siglip_model.is_none()
            && self.bioclip_dataset.is_none()
    }

    pub fn for_preset(name: &str) -> Self {
        Self {
            region: Some("other".to_owned()),
            preset: Some(name.to_owned()),
            ..Self::default()
        }
    }

    pub fn render(&self, cache_dir: &str) -> Result<LumenConfig, DockerConfigError> {
        let preset_name = self
            .preset
            .as_deref()
            .ok_or(DockerConfigError::MissingVariable { name: PRESET_VAR })?;
        let options = RenderOptions {
            region: self.region.as_deref().unwrap_or("other"),
            cache_dir,
            target: ConfigTarget::Network,
        };

        if preset_name == "custom" {
            let services_value = self
                .services
                .as_deref()
                .ok_or(DockerConfigError::MissingCustomServices)?;
            let services = parse_services(services_value)?;
            let dataset = if services.contains(&"bioclip") {
                Some(
                    self.bioclip_dataset
                        .as_deref()
                        .unwrap_or(BIOCLIP_CORE_DATASET),
                )
            } else {
                self.bioclip_dataset.as_deref()
            };
            return custom_config(&services, self.siglip_model.as_deref(), dataset, &options)
                .map_err(DockerConfigError::Render);
        }

        if let Some(name) = self.first_custom_var() {
            return Err(DockerConfigError::CustomValueWithPreset {
                name,
                preset: preset_name.to_owned(),
            });
        }
        let preset =
            Preset::by_name(preset_name).ok_or_else(|| DockerConfigError::InvalidValue {
                name: PRESET_VAR,
                value: preset_name.to_owned(),
                expected: "minimal, basic, brave, or custom".to_owned(),
            })?;
        preset_config(preset, &options).map_err(DockerConfigError::Render)
    }

    fn first_custom_var(&self) -> Option<&'static str> {
        [
            (self.services.as_ref(), SERVICES_VAR),
            (self.siglip_model.as_ref(), SIGLIP_MODEL_VAR),
            (self.bioclip_dataset.as_ref(), BIOCLIP_DATASET_VAR),
        ]
        .into_iter()
        .find_map(|(value, name)| value.map(|_| name))
    }
}

pub fn has_process_env() -> bool {
    ENV_VARS.iter().any(|name| env::var_os(name).is_some())
}

fn read_env(name: &'static str) -> Result<Option<String>, DockerConfigError> {
    match env::var(name) {
        Ok(value) if value.trim().is_empty() => Err(DockerConfigError::EmptyVariable { name }),
        Ok(value) => Ok(Some(value.trim().to_owned())),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(DockerConfigError::NonUnicode { name }),
    }
}

fn parse_services(value: &str) -> Result<Vec<&'static str>, DockerConfigError> {
    let mut selected = Vec::new();
    for raw in value.split(',') {
        let service = raw.trim();
        let canonical = SERVICE_ORDER
            .iter()
            .copied()
            .find(|candidate| *candidate == service)
            .ok_or_else(|| DockerConfigError::InvalidValue {
                name: SERVICES_VAR,
                value: value.to_owned(),
                expected: "a comma-separated subset of siglip, face, ocr, bioclip".to_owned(),
            })?;
        if selected.contains(&canonical) {
            return Err(DockerConfigError::DuplicateService {
                service: canonical.to_owned(),
            });
        }
        selected.push(canonical);
    }
    if selected.is_empty() {
        return Err(DockerConfigError::MissingCustomServices);
    }
    Ok(SERVICE_ORDER
        .iter()
        .copied()
        .filter(|service| selected.contains(service))
        .collect())
}

#[derive(Debug, Error)]
pub enum DockerConfigError {
    #[error("environment variable `{name}` must contain valid UTF-8")]
    NonUnicode { name: &'static str },

    #[error("environment variable `{name}` must not be empty")]
    EmptyVariable { name: &'static str },

    #[error("missing required Docker environment variable `{name}`")]
    MissingVariable { name: &'static str },

    #[error("invalid `{name}` value `{value}`; expected {expected}")]
    InvalidValue {
        name: &'static str,
        value: String,
        expected: String,
    },

    #[error("`{name}` cannot be used with `LUMEN_PRESET={preset}`; choose `custom`")]
    CustomValueWithPreset { name: &'static str, preset: String },

    #[error("`LUMEN_PRESET=custom` requires a non-empty `LUMEN_SERVICES`")]
    MissingCustomServices,

    #[error("service `{service}` is listed more than once in `LUMEN_SERVICES`")]
    DuplicateService { service: String },

    #[error("invalid Docker configuration: {0}")]
    Render(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_schema::{Region, SIGLIP_BRAVE_MODEL};

    #[test]
    fn official_presets_are_rendered_by_the_canonical_schema() {
        for preset in Preset::all() {
            let config = DockerConfigInput::for_preset(preset.name)
                .render("/models")
                .unwrap();
            assert_eq!(config.deployment_service_names(), preset.components);
            assert_eq!(config.server.host, "0.0.0.0");
        }
    }

    #[test]
    fn custom_selects_services_and_overrides() {
        let input = DockerConfigInput {
            region: Some("cn".to_owned()),
            preset: Some("custom".to_owned()),
            services: Some("bioclip, siglip".to_owned()),
            siglip_model: Some(SIGLIP_BRAVE_MODEL.to_owned()),
            bioclip_dataset: Some("TreeOfLife200M".to_owned()),
        };
        let config = input.render("/models").unwrap();
        assert_eq!(config.metadata.region, Region::Cn);
        assert_eq!(config.deployment_service_names(), vec!["siglip", "bioclip"]);
        assert_eq!(
            config.services["siglip"].models["default"].model,
            SIGLIP_BRAVE_MODEL
        );
    }

    #[test]
    fn custom_bioclip_defaults_to_the_core_dataset() {
        let input = DockerConfigInput {
            preset: Some("custom".to_owned()),
            services: Some("bioclip".to_owned()),
            ..Default::default()
        };
        let config = input.render("/models").unwrap();
        assert_eq!(
            config.services["bioclip"].models["default"]
                .dataset
                .as_deref(),
            Some(BIOCLIP_CORE_DATASET)
        );
    }

    #[test]
    fn preset_rejects_custom_values() {
        let input = DockerConfigInput {
            preset: Some("basic".to_owned()),
            services: Some("siglip".to_owned()),
            ..Default::default()
        };
        assert!(matches!(
            input.render("/models"),
            Err(DockerConfigError::CustomValueWithPreset { .. })
        ));
    }

    #[test]
    fn custom_rejects_duplicate_or_unknown_services() {
        for services in ["siglip,siglip", "siglip,gpu"] {
            let input = DockerConfigInput {
                preset: Some("custom".to_owned()),
                services: Some(services.to_owned()),
                ..Default::default()
            };
            assert!(input.render("/models").is_err());
        }
    }
}
