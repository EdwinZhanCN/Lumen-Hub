use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::SchemaEncodeError;

/// Reason why generation terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    EosToken,
    StopSequence,
    Error,
}

/// Optional metadata about the generation process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    /// Sampling temperature used for generation.
    #[validate(range(min = 0.0))]
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter used for generation.
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Maximum tokens allowed for generation.
    #[validate(range(min = 1))]
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Random seed used for generation (if deterministic).
    #[serde(default)]
    pub seed: Option<i64>,

    /// Time taken to generate the response in milliseconds.
    #[validate(range(min = 0.0))]
    #[serde(default)]
    pub generation_time_ms: Option<f32>,

    /// Number of chunks in streaming generation (if applicable).
    #[validate(range(min = 0))]
    #[serde(default)]
    pub streaming_chunks: Option<usize>,
}

/// Universal schema for text generation responses across Lumen VLM services.
/// Returns generated text with metadata about the generation process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextGenerationV1 {
    /// Generated text content.
    #[validate(length(min = 0))]
    pub text: String,

    /// Reason why generation terminated.
    pub finish_reason: FinishReason,

    /// Number of tokens generated in the response.
    #[validate(range(min = 0))]
    pub generated_tokens: usize,

    /// Number of tokens in the input prompt.
    #[validate(range(min = 0))]
    #[serde(default)]
    pub input_tokens: Option<usize>,

    /// Identifier of the model that generated the text.
    #[validate(length(min = 1))]
    pub model_id: String,

    /// Optional metadata about the generation process.
    #[validate(nested)]
    #[serde(default)]
    pub metadata: Option<Metadata>,
}

impl TextGenerationV1 {
    pub fn new(
        text: impl Into<String>,
        finish_reason: FinishReason,
        generated_tokens: usize,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            finish_reason,
            generated_tokens,
            input_tokens: None,
            model_id: model_id.into(),
            metadata: None,
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

    use super::{FinishReason, Metadata, TextGenerationV1};

    #[test]
    fn serializes_finish_reason_as_snake_case_string() {
        let encoded =
            serde_json::to_string(&FinishReason::EosToken).expect("finish reason serializes");

        assert_eq!(encoded, "\"eos_token\"");
    }

    #[test]
    fn validates_metadata_constraints() {
        let invalid = Metadata {
            temperature: Some(-0.1),
            top_p: Some(1.1),
            max_tokens: Some(0),
            seed: None,
            generation_time_ms: Some(-1.0),
            streaming_chunks: None,
        };

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn validates_response_constraints() {
        let invalid = TextGenerationV1 {
            text: String::new(),
            finish_reason: FinishReason::Stop,
            generated_tokens: 0,
            input_tokens: None,
            model_id: String::new(),
            metadata: None,
        };

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn allows_empty_generated_text() {
        let response = TextGenerationV1::new("", FinishReason::Stop, 0, "llm");

        assert!(response.validate().is_ok());
    }

    #[test]
    fn validates_nested_metadata_when_present() {
        let response = TextGenerationV1 {
            text: "hello".to_owned(),
            finish_reason: FinishReason::Stop,
            generated_tokens: 1,
            input_tokens: Some(0),
            model_id: "llm".to_owned(),
            metadata: Some(Metadata {
                temperature: None,
                top_p: Some(2.0),
                max_tokens: None,
                seed: None,
                generation_time_ms: None,
                streaming_chunks: None,
            }),
        };

        assert!(response.validate().is_err());
    }

    #[test]
    fn forbids_extra_metadata_fields() {
        let value = json!({
            "temperature": 0.7,
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<Metadata>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn forbids_extra_response_fields() {
        let value = json!({
            "text": "hello",
            "finish_reason": "stop",
            "generated_tokens": 1,
            "model_id": "llm",
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<TextGenerationV1>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn encodes_valid_text_generation_schema_as_json_bytes() {
        let mut response = TextGenerationV1::new("hello", FinishReason::StopSequence, 1, "llm");
        response.input_tokens = Some(3);
        response.metadata = Some(Metadata {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(128),
            seed: Some(42),
            generation_time_ms: Some(12.5),
            streaming_chunks: Some(0),
        });

        let bytes = response
            .to_json_bytes()
            .expect("valid text generation response encodes");
        let decoded: TextGenerationV1 =
            serde_json::from_slice(&bytes).expect("encoded text generation response decodes");

        assert_eq!(decoded, response);
    }
}
