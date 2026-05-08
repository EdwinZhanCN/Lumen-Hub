use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::SchemaEncodeError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Label {
    /// The classification label or class name.
    pub label: String,

    /// Confidence score for this label.
    pub score: f32,
}

/// Classification response schema for Lumen services. Returns ranked labels
/// with confidence scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelsV1 {
    /// Array of classification results.
    #[validate(nested)]
    pub labels: Vec<Label>,

    /// Identifier of the model that generated the classification.
    #[validate(length(min = 1))]
    pub model_id: String,
}

impl LabelsV1 {
    pub fn new(labels: Vec<Label>, model_id: impl Into<String>) -> Self {
        Self {
            labels,
            model_id: model_id.into(),
        }
    }

    pub fn to_json_bytes(&self) -> Result<Bytes, SchemaEncodeError> {
        self.validate()?;
        Ok(Bytes::from(serde_json::to_vec(self)?))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use validator::Validate;

    use super::{Label, LabelsV1};

    #[test]
    fn allows_empty_labels_list() {
        let response = LabelsV1::new(Vec::new(), "imagenet-resnet50");

        assert!(response.validate().is_ok());
    }

    #[test]
    fn validates_model_id_min_length() {
        let response = LabelsV1::new(Vec::new(), "");

        assert!(response.validate().is_err());
    }

    #[test]
    fn does_not_constrain_label_score_range() {
        let response = LabelsV1::new(
            vec![Label {
                label: "cat".to_owned(),
                score: 42.0,
            }],
            "classifier",
        );

        assert!(response.validate().is_ok());
    }

    #[test]
    fn forbids_extra_label_fields() {
        let value = json!({
            "label": "cat",
            "score": 0.9,
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<Label>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn forbids_extra_response_fields() {
        let value = json!({
            "labels": [],
            "model_id": "imagenet-resnet50",
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<LabelsV1>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn encodes_valid_labels_schema_as_json_bytes() {
        let response = LabelsV1::new(
            vec![Label {
                label: "cat".to_owned(),
                score: 0.9,
            }],
            "imagenet-resnet50",
        );

        let bytes = response
            .to_json_bytes()
            .expect("valid labels response encodes");
        let decoded: LabelsV1 =
            serde_json::from_slice(&bytes).expect("encoded labels response decodes");

        assert_eq!(decoded, response);
    }
}
