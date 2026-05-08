use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::SchemaEncodeError;

/// Universal schema for embedding responses across all Lumen services
/// (face, clip, ocr, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingV1 {
    /// Embedding vector.
    #[validate(length(min = 1))]
    pub vector: Vec<f32>,

    /// Embedding dimension.
    #[validate(range(min = 1))]
    pub dim: usize,

    /// Model identifier that generated the embedding.
    #[validate(length(min = 1))]
    pub model_id: String,
}

impl EmbeddingV1 {
    pub fn new(vector: Vec<f32>, model_id: impl Into<String>) -> Self {
        let dim = vector.len();
        Self {
            vector,
            dim,
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

    use super::EmbeddingV1;

    #[test]
    fn validates_minimum_field_constraints() {
        let invalid = EmbeddingV1 {
            vector: Vec::new(),
            dim: 0,
            model_id: String::new(),
        };

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn forbids_extra_fields() {
        let value = json!({
            "vector": [0.1, 0.2],
            "dim": 2,
            "model_id": "clip-vit-b-32",
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<EmbeddingV1>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn encodes_valid_schema_as_json_bytes() {
        let embedding = EmbeddingV1::new(vec![0.1, 0.2], "clip-vit-b-32");

        let bytes = embedding.to_json_bytes().expect("valid embedding encodes");
        let decoded: EmbeddingV1 =
            serde_json::from_slice(&bytes).expect("encoded embedding decodes");

        assert_eq!(decoded, embedding);
    }
}
