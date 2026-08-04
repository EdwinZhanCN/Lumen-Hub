//! Strict, allow-listed Docker environment overrides.
//!
//! Native and mounted-config launches remain file-driven when none of these
//! variables are present. The published Docker Compose files opt in by setting
//! `LUMEN_REGION` and `LUMEN_PRESET`.

use std::env;

use lumen_schema::{
    BIOCLIP_DATASETS, BIOCLIP_DEFAULT_MODEL, ConfigValidationError, FACE_DEFAULT_MODEL,
    LumenConfig, Mode, OCR_DEFAULT_MODEL, Preset, Region, SERVICE_ORDER, ServiceName, models_for,
};
use thiserror::Error;

pub const REGION_VAR: &str = "LUMEN_REGION";
pub const PRESET_VAR: &str = "LUMEN_PRESET";
pub const SERVICES_VAR: &str = "LUMEN_SERVICES";
pub const SIGLIP_MODEL_VAR: &str = "LUMEN_SIGLIP_MODEL";
pub const FACE_MODEL_VAR: &str = "LUMEN_FACE_MODEL";
pub const OCR_MODEL_VAR: &str = "LUMEN_OCR_MODEL";
pub const BIOCLIP_MODEL_VAR: &str = "LUMEN_BIOCLIP_MODEL";
pub const BIOCLIP_DATASET_VAR: &str = "LUMEN_BIOCLIP_DATASET";

const CUSTOM_VARS: [&str; 6] = [
    SERVICES_VAR,
    SIGLIP_MODEL_VAR,
    FACE_MODEL_VAR,
    OCR_MODEL_VAR,
    BIOCLIP_MODEL_VAR,
    BIOCLIP_DATASET_VAR,
];
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DockerConfigInput {
    region: Option<String>,
    preset: Option<String>,
    services: Option<String>,
    siglip_model: Option<String>,
    face_model: Option<String>,
    ocr_model: Option<String>,
    bioclip_model: Option<String>,
    bioclip_dataset: Option<String>,
}

impl DockerConfigInput {
    pub fn for_preset(preset: impl Into<String>) -> Self {
        Self {
            preset: Some(preset.into()),
            ..Self::default()
        }
    }

    pub fn from_process_env() -> Result<Self, DockerConfigError> {
        Ok(Self {
            region: read_env(REGION_VAR)?,
            preset: read_env(PRESET_VAR)?,
            services: read_env(SERVICES_VAR)?,
            siglip_model: read_env(SIGLIP_MODEL_VAR)?,
            face_model: read_env(FACE_MODEL_VAR)?,
            ocr_model: read_env(OCR_MODEL_VAR)?,
            bioclip_model: read_env(BIOCLIP_MODEL_VAR)?,
            bioclip_dataset: read_env(BIOCLIP_DATASET_VAR)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.region.is_none() && self.preset.is_none() && self.custom_values().all(Option::is_none)
    }

    pub fn apply(&self, config: &mut LumenConfig) -> Result<(), DockerConfigError> {
        if let Some(region) = &self.region {
            config.metadata.region = match region.as_str() {
                "other" => Region::Other,
                "cn" => Region::Cn,
                _ => return Err(invalid_value(REGION_VAR, region, "other or cn")),
            };
        }

        match self.preset.as_deref() {
            None => {
                if let Some(name) = self.first_custom_var() {
                    return Err(DockerConfigError::RequiresPreset { name });
                }
            }
            Some("custom") => self.apply_custom(config)?,
            Some(value) => {
                let preset = Preset::by_name(value).ok_or_else(|| {
                    invalid_value(PRESET_VAR, value, "minimal, basic, brave, or custom")
                })?;
                self.reject_custom_vars(value)?;
                apply_preset(config, preset)?;
            }
        }

        config.validate_config()?;
        Ok(())
    }

    fn apply_custom(&self, config: &mut LumenConfig) -> Result<(), DockerConfigError> {
        let raw_services = self
            .services
            .as_deref()
            .ok_or(DockerConfigError::MissingCustomServices)?;
        let services = parse_services(raw_services)?;

        apply_services(config, &services)?;
        apply_optional_model(
            config,
            &services,
            "siglip",
            SIGLIP_MODEL_VAR,
            self.siglip_model.as_deref(),
        )?;
        apply_optional_model(
            config,
            &services,
            "face",
            FACE_MODEL_VAR,
            self.face_model.as_deref(),
        )?;
        apply_optional_model(
            config,
            &services,
            "ocr",
            OCR_MODEL_VAR,
            self.ocr_model.as_deref(),
        )?;
        apply_optional_model(
            config,
            &services,
            "bioclip",
            BIOCLIP_MODEL_VAR,
            self.bioclip_model.as_deref(),
        )?;
        apply_optional_dataset(config, &services, self.bioclip_dataset.as_deref())?;
        Ok(())
    }

    fn reject_custom_vars(&self, preset: &str) -> Result<(), DockerConfigError> {
        if let Some(name) = self.first_custom_var() {
            return Err(DockerConfigError::CustomValueWithPreset {
                name,
                preset: preset.to_owned(),
            });
        }
        Ok(())
    }

    fn custom_values(&self) -> impl Iterator<Item = &Option<String>> {
        [
            &self.services,
            &self.siglip_model,
            &self.face_model,
            &self.ocr_model,
            &self.bioclip_model,
            &self.bioclip_dataset,
        ]
        .into_iter()
    }

    fn first_custom_var(&self) -> Option<&'static str> {
        self.custom_values()
            .zip(CUSTOM_VARS)
            .find_map(|(value, name)| value.as_ref().map(|_| name))
    }
}

fn read_env(name: &'static str) -> Result<Option<String>, DockerConfigError> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(DockerConfigError::NonUnicode { name }),
    }
}

fn apply_preset(config: &mut LumenConfig, preset: Preset) -> Result<(), DockerConfigError> {
    apply_services(config, preset.components)?;

    set_default_model(config, "siglip", preset.siglip_model)?;
    set_default_model(config, "face", FACE_DEFAULT_MODEL)?;
    if preset.includes("ocr") {
        set_default_model(config, "ocr", OCR_DEFAULT_MODEL)?;
    }
    if preset.includes("bioclip") {
        set_default_model(config, "bioclip", BIOCLIP_DEFAULT_MODEL)?;
        set_default_dataset(
            config,
            "bioclip",
            preset
                .bioclip_dataset
                .expect("a preset containing BioCLIP must select a dataset"),
        )?;
    }
    Ok(())
}

fn parse_services(value: &str) -> Result<Vec<&'static str>, DockerConfigError> {
    let mut selected = Vec::new();

    for raw in value.split(',') {
        let service = raw.trim();
        let canonical = SERVICE_ORDER
            .iter()
            .copied()
            .find(|candidate| *candidate == service)
            .ok_or_else(|| {
                invalid_value(
                    SERVICES_VAR,
                    value,
                    "a comma-separated subset of siglip, face, ocr, bioclip",
                )
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

fn apply_services(config: &mut LumenConfig, services: &[&str]) -> Result<(), DockerConfigError> {
    for service_name in SERVICE_ORDER {
        let service = config.services.get_mut(service_name).ok_or_else(|| {
            DockerConfigError::MissingServiceDefinition {
                service: service_name.to_owned(),
            }
        })?;
        service.enabled = services.contains(&service_name);
    }

    config.deployment.mode = Mode::Hub;
    config.deployment.service = None;
    config.deployment.services = Some(
        services
            .iter()
            .map(|service| ServiceName((*service).to_owned()))
            .collect(),
    );
    Ok(())
}

fn apply_optional_model(
    config: &mut LumenConfig,
    services: &[&str],
    service: &'static str,
    variable: &'static str,
    value: Option<&str>,
) -> Result<(), DockerConfigError> {
    let Some(value) = value else {
        return Ok(());
    };
    require_selected(services, service, variable)?;
    let allowed =
        models_for(service).ok_or_else(|| DockerConfigError::MissingServiceDefinition {
            service: service.to_owned(),
        })?;
    if !allowed.contains(&value) {
        return Err(invalid_value(variable, value, &allowed.join(" or ")));
    }
    set_default_model(config, service, value)
}

fn apply_optional_dataset(
    config: &mut LumenConfig,
    services: &[&str],
    value: Option<&str>,
) -> Result<(), DockerConfigError> {
    let Some(value) = value else {
        return Ok(());
    };
    require_selected(services, "bioclip", BIOCLIP_DATASET_VAR)?;
    if !BIOCLIP_DATASETS.contains(&value) {
        return Err(invalid_value(
            BIOCLIP_DATASET_VAR,
            value,
            "TreeOfLife200MCore or TreeOfLife200M",
        ));
    }
    set_default_dataset(config, "bioclip", value)
}

fn require_selected(
    services: &[&str],
    service: &'static str,
    variable: &'static str,
) -> Result<(), DockerConfigError> {
    if services.contains(&service) {
        Ok(())
    } else {
        Err(DockerConfigError::ModelForDisabledService { variable, service })
    }
}

fn set_default_model(
    config: &mut LumenConfig,
    service: &str,
    model: &str,
) -> Result<(), DockerConfigError> {
    let model_config = default_model_mut(config, service)?;
    model_config.model = model.to_owned();
    Ok(())
}

fn set_default_dataset(
    config: &mut LumenConfig,
    service: &str,
    dataset: &str,
) -> Result<(), DockerConfigError> {
    let model_config = default_model_mut(config, service)?;
    model_config.dataset = Some(dataset.to_owned());
    Ok(())
}

fn default_model_mut<'a>(
    config: &'a mut LumenConfig,
    service: &str,
) -> Result<&'a mut lumen_schema::ModelConfig, DockerConfigError> {
    config
        .services
        .get_mut(service)
        .ok_or_else(|| DockerConfigError::MissingServiceDefinition {
            service: service.to_owned(),
        })?
        .models
        .get_mut("default")
        .ok_or_else(|| DockerConfigError::MissingDefaultModel {
            service: service.to_owned(),
        })
}

fn invalid_value(name: &'static str, value: &str, expected: &str) -> DockerConfigError {
    DockerConfigError::InvalidValue {
        name,
        value: value.to_owned(),
        expected: expected.to_owned(),
    }
}

#[derive(Debug, Error)]
pub enum DockerConfigError {
    #[error("environment variable `{name}` must contain valid UTF-8")]
    NonUnicode { name: &'static str },

    #[error("invalid `{name}` value `{value}`; expected {expected}")]
    InvalidValue {
        name: &'static str,
        value: String,
        expected: String,
    },

    #[error("`{name}` requires `LUMEN_PRESET=custom`")]
    RequiresPreset { name: &'static str },

    #[error("`{name}` cannot be used with `LUMEN_PRESET={preset}`; choose `custom`")]
    CustomValueWithPreset { name: &'static str, preset: String },

    #[error("`LUMEN_PRESET=custom` requires a non-empty `LUMEN_SERVICES`")]
    MissingCustomServices,

    #[error("service `{service}` is listed more than once in `LUMEN_SERVICES`")]
    DuplicateService { service: String },

    #[error("`{variable}` configures `{service}`, but that service is not selected")]
    ModelForDisabledService {
        variable: &'static str,
        service: &'static str,
    },

    #[error("base config does not define required Docker service `{service}`")]
    MissingServiceDefinition { service: String },

    #[error("base config service `{service}` does not define the `default` model alias")]
    MissingDefaultModel { service: String },

    #[error("Docker environment produced an invalid config: {0}")]
    Config(#[from] ConfigValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_schema::{
        BIOCLIP_CORE_DATASET, BIOCLIP_FULL_DATASET, SIGLIP_BASE_MODEL, SIGLIP_BRAVE_MODEL,
    };

    fn base_config() -> LumenConfig {
        serde_yaml::from_str(include_str!(
            "../../../packaging/docker/config.default.yaml"
        ))
        .expect("Docker base config parses")
    }

    fn input(preset: &str) -> DockerConfigInput {
        DockerConfigInput::for_preset(preset)
    }

    fn default_model<'a>(config: &'a LumenConfig, service: &str) -> &'a lumen_schema::ModelConfig {
        &config.services[service].models["default"]
    }

    #[test]
    fn empty_input_preserves_file_config() {
        let mut config = base_config();
        let original = config.clone();

        DockerConfigInput::default().apply(&mut config).unwrap();

        assert_eq!(config, original);
    }

    #[test]
    fn minimal_selects_only_semantic_and_face() {
        let mut config = base_config();

        input("minimal").apply(&mut config).unwrap();

        assert_eq!(config.deployment_service_names(), vec!["siglip", "face"]);
        assert_eq!(default_model(&config, "siglip").model, SIGLIP_BASE_MODEL);
    }

    #[test]
    fn basic_selects_all_services_and_core_catalog() {
        let mut config = base_config();

        input("basic").apply(&mut config).unwrap();

        assert_eq!(config.deployment_service_names(), SERVICE_ORDER);
        assert_eq!(default_model(&config, "siglip").model, SIGLIP_BASE_MODEL);
        assert_eq!(
            default_model(&config, "bioclip").dataset.as_deref(),
            Some(BIOCLIP_CORE_DATASET)
        );
    }

    #[test]
    fn brave_selects_large_semantic_model_and_full_catalog() {
        let mut config = base_config();

        input("brave").apply(&mut config).unwrap();

        assert_eq!(default_model(&config, "siglip").model, SIGLIP_BRAVE_MODEL);
        assert_eq!(
            default_model(&config, "bioclip").dataset.as_deref(),
            Some(BIOCLIP_FULL_DATASET)
        );
    }

    #[test]
    fn custom_selects_services_and_models() {
        let mut config = base_config();
        let input = DockerConfigInput {
            region: Some("cn".to_owned()),
            preset: Some("custom".to_owned()),
            services: Some("bioclip, siglip".to_owned()),
            siglip_model: Some(SIGLIP_BRAVE_MODEL.to_owned()),
            bioclip_model: Some(BIOCLIP_DEFAULT_MODEL.to_owned()),
            bioclip_dataset: Some(BIOCLIP_FULL_DATASET.to_owned()),
            ..Default::default()
        };

        input.apply(&mut config).unwrap();

        assert_eq!(config.metadata.region, Region::Cn);
        assert_eq!(config.deployment_service_names(), vec!["siglip", "bioclip"]);
        assert_eq!(default_model(&config, "siglip").model, SIGLIP_BRAVE_MODEL);
        assert_eq!(
            default_model(&config, "bioclip").dataset.as_deref(),
            Some(BIOCLIP_FULL_DATASET)
        );
        assert!(!config.services["face"].enabled);
        assert!(!config.services["ocr"].enabled);
    }

    #[test]
    fn preset_rejects_custom_values() {
        let mut config = base_config();
        let input = DockerConfigInput {
            preset: Some("basic".to_owned()),
            services: Some("siglip".to_owned()),
            ..Default::default()
        };

        let error = input.apply(&mut config).unwrap_err();

        assert!(matches!(
            error,
            DockerConfigError::CustomValueWithPreset { .. }
        ));
    }

    #[test]
    fn custom_rejects_model_for_unselected_service() {
        let mut config = base_config();
        let input = DockerConfigInput {
            preset: Some("custom".to_owned()),
            services: Some("face".to_owned()),
            siglip_model: Some(SIGLIP_BASE_MODEL.to_owned()),
            ..Default::default()
        };

        let error = input.apply(&mut config).unwrap_err();

        assert!(matches!(
            error,
            DockerConfigError::ModelForDisabledService { .. }
        ));
    }

    #[test]
    fn custom_requires_at_least_one_service() {
        let mut config = base_config();

        let error = input("custom").apply(&mut config).unwrap_err();

        assert!(matches!(error, DockerConfigError::MissingCustomServices));
    }

    /// Environment options mirroring packaging/docker/config.default.yaml.
    fn parity_options() -> lumen_schema::RenderOptions<'static> {
        lumen_schema::RenderOptions {
            region: "other",
            cache_dir: "/models",
            metadata_version: "0.1.0",
            runtime: "burn",
            precision: "fp16q8",
        }
    }

    /// The Docker base image catalog keeps default model metadata on disabled
    /// services (for example BioCLIP's core dataset), while the canonical
    /// render leaves them neutral. The entry-point contract is: metadata,
    /// deployment, server, and every selected service are identical.
    fn assert_config_parity(docker: &LumenConfig, canonical: &LumenConfig) {
        assert_eq!(docker.metadata, canonical.metadata);
        assert_eq!(docker.deployment, canonical.deployment);
        assert_eq!(docker.server, canonical.server);
        for (service, docker_config) in &docker.services {
            let canonical_config = &canonical.services[service];
            assert_eq!(
                docker_config.enabled, canonical_config.enabled,
                "service {service} enabled flag"
            );
            if docker_config.enabled {
                assert_eq!(
                    docker_config, canonical_config,
                    "selected service {service} must be identical"
                );
            }
        }
    }

    /// The Docker env resolver and the canonical schema renderer must agree on
    /// every official preset.
    #[test]
    fn docker_env_matches_canonical_preset_render() {
        for preset in Preset::all() {
            let mut via_docker = base_config();
            DockerConfigInput::for_preset(preset.name)
                .apply(&mut via_docker)
                .unwrap();
            let via_schema =
                lumen_schema::preset_config(*preset, &parity_options()).expect("canonical render");
            assert_config_parity(&via_docker, &via_schema);
        }
    }

    /// Same parity guarantee for a representative custom combination.
    #[test]
    fn docker_env_matches_canonical_custom_render() {
        let mut via_docker = base_config();
        let input = DockerConfigInput {
            region: Some("cn".to_owned()),
            preset: Some("custom".to_owned()),
            services: Some("bioclip, siglip".to_owned()),
            siglip_model: Some(SIGLIP_BRAVE_MODEL.to_owned()),
            bioclip_model: Some(BIOCLIP_DEFAULT_MODEL.to_owned()),
            bioclip_dataset: Some(BIOCLIP_FULL_DATASET.to_owned()),
            ..Default::default()
        };
        input.apply(&mut via_docker).unwrap();

        let mut options = parity_options();
        options.region = "cn";
        let via_schema = lumen_schema::custom_config(
            &["siglip", "bioclip"],
            Some(SIGLIP_BRAVE_MODEL),
            Some(BIOCLIP_FULL_DATASET),
            &options,
        )
        .expect("canonical render");
        assert_config_parity(&via_docker, &via_schema);
    }

    /// The committed fixtures under fixtures/config/ are the stable goldens
    /// shared by every entry point; they must equal the canonical render.
    #[test]
    fn committed_config_fixtures_match_canonical_render() {
        // Fixtures are rendered with the canonical machine-independent path.
        let fixture_options = lumen_schema::RenderOptions {
            cache_dir: "~/.lumen/models",
            ..parity_options()
        };
        let fixtures: &[(&str, &str)] = &[
            (
                "minimal",
                include_str!("../../../fixtures/config/minimal.yaml"),
            ),
            ("basic", include_str!("../../../fixtures/config/basic.yaml")),
            ("brave", include_str!("../../../fixtures/config/brave.yaml")),
            (
                "custom-siglip-bioclip",
                include_str!("../../../fixtures/config/custom-siglip-bioclip.yaml"),
            ),
            (
                "custom-face-ocr",
                include_str!("../../../fixtures/config/custom-face-ocr.yaml"),
            ),
            (
                "custom-siglip",
                include_str!("../../../fixtures/config/custom-siglip.yaml"),
            ),
        ];

        for (name, raw) in fixtures {
            let fixture: LumenConfig =
                serde_yaml::from_str(raw).unwrap_or_else(|e| panic!("{name} fixture parses: {e}"));
            let expected = if let Some(preset) = Preset::by_name(name) {
                lumen_schema::preset_config(preset, &fixture_options).expect("canonical render")
            } else {
                match *name {
                    "custom-siglip-bioclip" => lumen_schema::custom_config(
                        &["siglip", "bioclip"],
                        Some(SIGLIP_BRAVE_MODEL),
                        Some(BIOCLIP_FULL_DATASET),
                        &fixture_options,
                    )
                    .expect("canonical render"),
                    "custom-face-ocr" => {
                        lumen_schema::custom_config(&["face", "ocr"], None, None, &fixture_options)
                            .expect("canonical render")
                    }
                    "custom-siglip" => {
                        lumen_schema::custom_config(&["siglip"], None, None, &fixture_options)
                            .expect("canonical render")
                    }
                    other => panic!("unhandled fixture {other}"),
                }
            };
            assert_eq!(fixture, expected, "fixture {name}");
        }
    }
}
