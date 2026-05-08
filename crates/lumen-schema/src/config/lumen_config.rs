use std::{collections::BTreeMap, sync::LazyLock};

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

static SEMVER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+\.\d+\.\d+$").expect("valid semver regex"));
static SERVICE_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9_]*$").expect("valid service name regex"));
static MDNS_SERVICE_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9-]*$").expect("valid mDNS service name regex"));
static PACKAGE_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9_]*$").expect("valid package name regex"));
static RKNN_DEVICE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^rk\d+$").expect("valid RKNN device regex"));

#[derive(Debug, thiserror::Error)]
pub enum ConfigValidationError {
    #[error("config json parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("config field validation failed: {0}")]
    Fields(#[from] validator::ValidationErrors),

    #[error("config validation failed: {0}")]
    Invalid(String),
}

/// Platform region selection.
///
/// `cn` uses ModelScope. `other` is reserved for future Hugging Face routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    Cn,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    /// Configuration version (semantic versioning).
    #[validate(regex(path = "*SEMVER_RE"))]
    pub version: String,

    /// Platform region selection.
    pub region: Region,

    /// Model cache directory path (supports shell-style `~` expansion by callers).
    #[validate(length(min = 1))]
    pub cache_dir: String,
}

/// Deployment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Single,
    Hub,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct ServiceName(#[schemars(regex(pattern = "^[a-z][a-z0-9_]*$"))] pub String);

impl ServiceName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<ServiceName> for String {
    fn from(value: ServiceName) -> Self {
        value.0
    }
}

impl Validate for ServiceName {
    fn validate(&self) -> Result<(), ValidationErrors> {
        if SERVICE_NAME_RE.is_match(&self.0) {
            return Ok(());
        }

        let mut errors = ValidationErrors::new();
        errors.add("root", ValidationError::new("regex"));
        Err(errors)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    /// Deployment mode.
    pub mode: Mode,

    /// Service name for single mode.
    #[validate(nested)]
    #[serde(default)]
    pub service: Option<ServiceName>,

    /// Service names for hub mode.
    #[validate(nested)]
    #[serde(default)]
    pub services: Option<Vec<ServiceName>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Mdns {
    /// Enable mDNS service discovery.
    #[serde(default)]
    pub enabled: bool,

    /// mDNS service name, required when `enabled` is true.
    #[validate(regex(path = "*MDNS_SERVICE_NAME_RE"))]
    #[serde(default)]
    pub service_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// gRPC server port.
    #[validate(range(min = 1024, max = 65535))]
    pub port: u16,

    /// Server bind address.
    #[serde(default = "default_host")]
    pub host: String,

    #[validate(nested)]
    #[serde(default)]
    pub mdns: Option<Mdns>,
}

/// Model runtime type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    Onnx,
    CandleOnnx,
    Rknn,
}

impl Runtime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::CandleOnnx => "candle_onnx",
            Self::Rknn => "rknn",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model repository name.
    #[validate(length(min = 1))]
    pub model: String,

    /// Model runtime type.
    pub runtime: Runtime,

    /// RKNN device identifier, required when `runtime` is `rknn`.
    #[validate(regex(path = "*RKNN_DEVICE_RE"))]
    #[serde(default)]
    pub rknn_device: Option<String>,

    /// Dataset name for zero-shot classification.
    #[serde(default)]
    pub dataset: Option<String>,

    /// Preferred precision for running the model.
    #[serde(default)]
    pub precision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServiceConfig {
    /// Whether to load this service.
    pub enabled: bool,

    /// Rust service package/crate name.
    #[validate(regex(path = "*PACKAGE_NAME_RE"))]
    pub package: String,

    /// Model configurations (alias -> config).
    #[validate(nested)]
    pub models: BTreeMap<String, ModelConfig>,
}

/// Unified configuration schema for all Lumen ML services.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LumenConfig {
    #[validate(nested)]
    pub metadata: Metadata,

    #[validate(nested)]
    pub deployment: Deployment,

    #[validate(nested)]
    pub server: ServerConfig,

    /// Service definitions keyed by service name.
    #[validate(nested)]
    pub services: BTreeMap<String, ServiceConfig>,
}

impl LumenConfig {
    pub fn from_json_str(input: &str) -> Result<Self, ConfigValidationError> {
        let config = serde_json::from_str::<Self>(input)?;
        config.validate_config()?;
        Ok(config)
    }

    pub fn validate_config(&self) -> Result<(), ConfigValidationError> {
        self.validate()?;
        self.validate_service_keys()?;
        self.validate_model_aliases()?;
        self.validate_deployment()?;
        self.validate_mdns()?;
        self.validate_runtime_requirements()?;
        Ok(())
    }

    pub fn service_enabled(&self, service_name: &str) -> bool {
        self.services
            .get(service_name)
            .map(|service| service.enabled)
            .unwrap_or(false)
    }

    pub fn deployment_service_names(&self) -> Vec<&str> {
        match self.deployment.mode {
            Mode::Single => self
                .deployment
                .service
                .as_ref()
                .map(|service| vec![service.as_str()])
                .unwrap_or_default(),
            Mode::Hub => self
                .deployment
                .services
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(ServiceName::as_str)
                .collect(),
        }
    }

    fn validate_service_keys(&self) -> Result<(), ConfigValidationError> {
        if self.services.is_empty() {
            return Err(ConfigValidationError::Invalid(
                "services must contain at least one service definition".to_owned(),
            ));
        }

        for service_name in self.services.keys() {
            if !SERVICE_NAME_RE.is_match(service_name) {
                return Err(ConfigValidationError::Invalid(format!(
                    "service key `{service_name}` must match ^[a-z][a-z0-9_]*$"
                )));
            }
        }

        Ok(())
    }

    fn validate_model_aliases(&self) -> Result<(), ConfigValidationError> {
        for (service_name, service) in &self.services {
            if service.models.is_empty() {
                return Err(ConfigValidationError::Invalid(format!(
                    "service `{service_name}` must define at least one model"
                )));
            }

            for alias in service.models.keys() {
                if alias.is_empty() {
                    return Err(ConfigValidationError::Invalid(format!(
                        "service `{service_name}` contains an empty model alias"
                    )));
                }
            }
        }

        Ok(())
    }

    fn validate_deployment(&self) -> Result<(), ConfigValidationError> {
        match self.deployment.mode {
            Mode::Single => {
                let service = self.deployment.service.as_ref().ok_or_else(|| {
                    ConfigValidationError::Invalid(
                        "deployment.service is required when deployment.mode is single".to_owned(),
                    )
                })?;
                self.validate_deployed_service(service.as_str())?;
            }
            Mode::Hub => {
                let services = self.deployment.services.as_ref().ok_or_else(|| {
                    ConfigValidationError::Invalid(
                        "deployment.services is required when deployment.mode is hub".to_owned(),
                    )
                })?;

                if services.is_empty() {
                    return Err(ConfigValidationError::Invalid(
                        "deployment.services must contain at least one service when deployment.mode is hub"
                            .to_owned(),
                    ));
                }

                for service in services {
                    self.validate_deployed_service(service.as_str())?;
                }
            }
        }

        Ok(())
    }

    fn validate_deployed_service(&self, service_name: &str) -> Result<(), ConfigValidationError> {
        let service = self.services.get(service_name).ok_or_else(|| {
            ConfigValidationError::Invalid(format!(
                "deployment references unknown service `{service_name}`"
            ))
        })?;

        if !service.enabled {
            return Err(ConfigValidationError::Invalid(format!(
                "deployment references disabled service `{service_name}`"
            )));
        }

        Ok(())
    }

    fn validate_mdns(&self) -> Result<(), ConfigValidationError> {
        if let Some(mdns) = &self.server.mdns {
            if mdns.enabled && mdns.service_name.is_none() {
                return Err(ConfigValidationError::Invalid(
                    "server.mdns.service_name is required when server.mdns.enabled is true"
                        .to_owned(),
                ));
            }
        }

        Ok(())
    }

    fn validate_runtime_requirements(&self) -> Result<(), ConfigValidationError> {
        for (service_name, service) in &self.services {
            for (alias, model) in &service.models {
                if model.runtime == Runtime::Rknn && model.rknn_device.is_none() {
                    return Err(ConfigValidationError::Invalid(format!(
                        "services.{service_name}.models.{alias}.rknn_device is required when runtime is rknn"
                    )));
                }
            }
        }

        Ok(())
    }
}

fn default_host() -> String {
    "0.0.0.0".to_owned()
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use validator::Validate;

    use super::{LumenConfig, Mode, Runtime, ServerConfig};

    #[test]
    fn parses_and_validates_single_service_config() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "onnx",
                                "precision": "fp32"
                            }
                        }
                    }
                }
            })
            .to_string(),
        )
        .expect("valid config parses");

        assert_eq!(config.deployment.mode, Mode::Single);
        assert_eq!(config.server.host, "0.0.0.0");
        assert!(config.service_enabled("clip"));
        assert_eq!(config.deployment_service_names(), vec!["clip"]);
    }

    #[test]
    fn validates_field_constraints() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1",
                    "region": "cn",
                    "cache_dir": ""
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 80
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "onnx"
                            }
                        }
                    }
                }
            })
            .to_string(),
        );

        assert!(config.is_err());
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "onnx"
                            }
                        }
                    }
                },
                "extra": "forbidden"
            })
            .to_string(),
        );

        assert!(config.is_err());
    }

    #[test]
    fn rejects_hub_deployment_without_services() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "hub",
                    "services": []
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "onnx"
                            }
                        }
                    }
                }
            })
            .to_string(),
        );

        assert!(config.is_err());
    }

    #[test]
    fn rejects_deployment_referencing_disabled_service() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip": {
                        "enabled": false,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "onnx"
                            }
                        }
                    }
                }
            })
            .to_string(),
        );

        assert!(config.is_err());
    }

    #[test]
    fn requires_mdns_service_name_when_enabled() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 50051,
                    "mdns": {
                        "enabled": true
                    }
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "onnx"
                            }
                        }
                    }
                }
            })
            .to_string(),
        );

        assert!(config.is_err());
    }

    #[test]
    fn requires_rknn_device_for_rknn_runtime() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "rknn"
                            }
                        }
                    }
                }
            })
            .to_string(),
        );

        assert!(config.is_err());
    }

    #[test]
    fn accepts_valid_rknn_runtime() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "clip"
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip": {
                        "enabled": true,
                        "package": "lumen_clip",
                        "models": {
                            "default": {
                                "model": "ViT-B-32",
                                "runtime": "rknn",
                                "rknn_device": "rk3588"
                            }
                        }
                    }
                }
            })
            .to_string(),
        )
        .expect("valid rknn config parses");

        let model = &config.services["clip"].models["default"];
        assert_eq!(model.runtime, Runtime::Rknn);
        assert_eq!(model.rknn_device.as_deref(), Some("rk3588"));
    }

    #[test]
    fn accepts_candle_onnx_runtime_without_rknn_device() {
        let config = LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "1.0.0",
                    "region": "cn",
                    "cache_dir": "~/.lumen/models"
                },
                "deployment": {
                    "mode": "single",
                    "service": "siglip"
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "siglip": {
                        "enabled": true,
                        "package": "lumen_siglip",
                        "models": {
                            "default": {
                                "model": "siglip-base",
                                "runtime": "candle_onnx"
                            }
                        }
                    }
                }
            })
            .to_string(),
        )
        .expect("valid candle_onnx config parses");

        let model = &config.services["siglip"].models["default"];
        assert_eq!(model.runtime, Runtime::CandleOnnx);
        assert_eq!(model.runtime.as_str(), "candle_onnx");
        assert!(model.rknn_device.is_none());
    }

    #[test]
    fn server_config_validates_port_range() {
        let server = ServerConfig {
            port: 50051,
            host: "127.0.0.1".to_owned(),
            mdns: None,
        };

        assert!(server.validate().is_ok());
    }
}
