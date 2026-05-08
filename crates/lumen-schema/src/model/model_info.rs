use std::{collections::BTreeMap, sync::LazyLock};

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

static SEMVER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+\.\d+\.\d+$").expect("valid semver regex"));
static RUNTIME_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9_-]+$").expect("valid runtime key regex"));

#[derive(Debug, thiserror::Error)]
pub enum ModelInfoValidationError {
    #[error("model info json parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("model info field validation failed: {0}")]
    Fields(#[from] validator::ValidationErrors),

    #[error("model info validation failed: {0}")]
    Invalid(String),
}

/// Runtime-facing schema for model assets consumed directly by Lumen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    /// Stable model asset identifier.
    #[validate(length(min = 1, max = 100))]
    pub name: String,

    /// Semantic version of this asset package.
    #[validate(regex(path = "*SEMVER_RE"))]
    pub version: String,

    /// Short human-readable description of the asset package.
    #[validate(length(min = 1, max = 500))]
    pub description: String,

    /// High-level task family, for example `clip`, `face`, `ocr`, or `vlm`.
    #[validate(length(min = 1))]
    pub model_type: String,

    /// Weak source reference for runtime introspection only.
    #[validate(nested)]
    pub source: ModelSource,

    /// Runtime inventory available to Lumen.
    #[validate(nested)]
    pub runtimes: RuntimeInventory,

    /// Task-specific weak contract area.
    #[serde(default)]
    pub task_metadata: Option<BTreeMap<String, serde_json::Value>>,

    /// Optional descriptive metadata for humans and tooling.
    #[validate(nested)]
    #[serde(default)]
    pub metadata: Option<ModelMetadata>,
}

impl ModelInfo {
    pub fn from_json_str(input: &str) -> Result<Self, ModelInfoValidationError> {
        let model_info = serde_json::from_str::<Self>(input)?;
        model_info.validate_model_info()?;
        Ok(model_info)
    }

    pub fn validate_model_info(&self) -> Result<(), ModelInfoValidationError> {
        self.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SourceFormat {
    Huggingface,
    Openclip,
    Modelscope,
    Custom,
    Local,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelSource {
    pub format: SourceFormat,

    /// Upstream identifier or local logical identifier.
    #[validate(length(min = 1))]
    pub repo_id: String,
}

/// Runtime inventory keyed by runtime name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct RuntimeInventory(#[schemars(length(min = 1))] pub BTreeMap<String, RuntimeSpec>);

impl RuntimeInventory {
    pub fn as_map(&self) -> &BTreeMap<String, RuntimeSpec> {
        &self.0
    }
}

impl Validate for RuntimeInventory {
    fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();

        if self.0.is_empty() {
            errors.add("root", ValidationError::new("length"));
        }

        for (runtime, spec) in &self.0 {
            if !RUNTIME_KEY_RE.is_match(runtime) {
                errors.add("root", ValidationError::new("runtime_key"));
            }

            if spec.validate().is_err() {
                errors.add("root", ValidationError::new("runtime_spec"));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSpec {
    /// Whether this runtime has packaged artifacts for the model.
    pub available: bool,

    /// Logical model components, such as `vision`, `text`, `detection`, or `recognition`.
    #[schemars(length(min = 1))]
    pub components: Vec<String>,

    /// Packaged artifact precisions, such as `fp32`, `fp16`, or `int8`.
    #[schemars(length(min = 1))]
    pub precisions: Vec<String>,
}

impl Validate for RuntimeSpec {
    fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = ValidationErrors::new();

        validate_nonempty_unique_strings("components", &self.components, &mut errors);
        validate_nonempty_unique_strings("precisions", &self.precisions, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata {
    #[serde(default)]
    pub license: Option<String>,

    #[serde(default)]
    pub author: Option<String>,

    #[serde(default)]
    pub created_at: Option<String>,

    #[serde(default)]
    pub updated_at: Option<String>,

    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

fn validate_nonempty_unique_strings(
    field_name: &'static str,
    values: &[String],
    errors: &mut ValidationErrors,
) {
    if values.is_empty() {
        errors.add(field_name, ValidationError::new("length"));
        return;
    }

    let mut seen = std::collections::BTreeSet::new();
    for value in values {
        if value.is_empty() {
            errors.add(field_name, ValidationError::new("length"));
        }
        if !seen.insert(value) {
            errors.add(field_name, ValidationError::new("unique"));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use validator::Validate;

    use super::{ModelInfo, SourceFormat};

    #[test]
    fn parses_valid_model_info() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package for image and text embedding.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
                },
                "runtimes": {
                    "onnx": {
                        "available": true,
                        "components": ["vision", "text"],
                        "precisions": ["fp32", "fp16"]
                    }
                },
                "task_metadata": {
                    "embedding_dim": 512,
                    "tasks": ["clip_image_embed", "clip_text_embed"]
                },
                "metadata": {
                    "license": "mit",
                    "author": "Lumen",
                    "created_at": "2026-04-17",
                    "updated_at": "2026-04-17T00:00:00Z",
                    "tags": ["clip", "openclip", "onnx"]
                }
            })
            .to_string(),
        )
        .expect("valid model info parses");

        assert_eq!(model_info.name, "openclip-vit-b-32");
        assert_eq!(model_info.source.format, SourceFormat::Huggingface);
        assert!(model_info.runtimes.as_map().contains_key("onnx"));
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32"
                },
                "runtimes": {
                    "onnx": {
                        "available": true,
                        "components": ["vision"],
                        "precisions": ["fp32"]
                    }
                },
                "extra": "forbidden"
            })
            .to_string(),
        );

        assert!(model_info.is_err());
    }

    #[test]
    fn validates_required_string_lengths_and_version_pattern() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "",
                "version": "1",
                "description": "",
                "model_type": "",
                "source": {
                    "format": "custom",
                    "repo_id": ""
                },
                "runtimes": {
                    "onnx": {
                        "available": true,
                        "components": ["vision"],
                        "precisions": ["fp32"]
                    }
                }
            })
            .to_string(),
        );

        assert!(model_info.is_err());
    }

    #[test]
    fn validates_runtime_inventory_is_not_empty() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32"
                },
                "runtimes": {}
            })
            .to_string(),
        );

        assert!(model_info.is_err());
    }

    #[test]
    fn validates_runtime_key_pattern() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32"
                },
                "runtimes": {
                    "bad runtime": {
                        "available": true,
                        "components": ["vision"],
                        "precisions": ["fp32"]
                    }
                }
            })
            .to_string(),
        );

        assert!(model_info.is_err());
    }

    #[test]
    fn validates_runtime_components_and_precisions() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32"
                },
                "runtimes": {
                    "onnx": {
                        "available": true,
                        "components": [],
                        "precisions": []
                    }
                }
            })
            .to_string(),
        );

        assert!(model_info.is_err());
    }

    #[test]
    fn rejects_extra_metadata_fields() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32"
                },
                "runtimes": {
                    "onnx": {
                        "available": true,
                        "components": ["vision"],
                        "precisions": ["fp32"]
                    }
                },
                "metadata": {
                    "license": "mit",
                    "extra": "forbidden"
                }
            })
            .to_string(),
        );

        assert!(model_info.is_err());
    }

    #[test]
    fn runtime_spec_validate_can_be_called_directly() {
        let model_info = ModelInfo::from_json_str(
            &json!({
                "name": "openclip-vit-b-32",
                "version": "1.0.0",
                "description": "OpenCLIP ViT-B-32 asset package.",
                "model_type": "clip",
                "source": {
                    "format": "huggingface",
                    "repo_id": "laion/CLIP-ViT-B-32"
                },
                "runtimes": {
                    "onnx": {
                        "available": true,
                        "components": ["vision"],
                        "precisions": ["fp32"]
                    }
                }
            })
            .to_string(),
        )
        .expect("valid model info parses");

        let spec = &model_info.runtimes.as_map()["onnx"];
        assert!(spec.validate().is_ok());
    }
}
