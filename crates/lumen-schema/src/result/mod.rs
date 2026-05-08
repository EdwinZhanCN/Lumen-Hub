mod embedding_v1;
mod face_v1;
mod labels_v1;
mod ocr_v1;
mod text_generation_v1;

pub use embedding_v1::EmbeddingV1;
pub use face_v1::{BboxItem, Face, FaceV1};
pub use labels_v1::{Label, LabelsV1};
pub use ocr_v1::{BoxItem, Item as OcrItem, OCRV1};
pub use text_generation_v1::{FinishReason, Metadata as TextGenerationMetadata, TextGenerationV1};

#[derive(Debug, thiserror::Error)]
pub enum SchemaEncodeError {
    #[error("schema validation failed: {0}")]
    Validation(#[from] validator::ValidationErrors),

    #[error("schema json serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}
