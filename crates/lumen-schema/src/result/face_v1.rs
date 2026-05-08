use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

use super::SchemaEncodeError;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema)]
pub struct BboxItem(#[schemars(range(min = 0.0))] pub f32);

impl From<f32> for BboxItem {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl Validate for BboxItem {
    fn validate(&self) -> Result<(), ValidationErrors> {
        if self.0 >= 0.0 {
            return Ok(());
        }

        let mut errors = ValidationErrors::new();
        errors.add("root", ValidationError::new("range"));
        Err(errors)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Face {
    /// Bounding box coordinates [x1, y1, x2, y2] where (x1, y1) is top-left
    /// corner and (x2, y2) is bottom-right corner.
    #[validate(length(min = 4, max = 4), nested)]
    pub bbox: Vec<BboxItem>,

    /// Detection confidence score.
    #[validate(range(min = 0.0, max = 1.0))]
    pub confidence: f32,

    /// Facial landmark points (optional).
    #[serde(default)]
    pub landmarks: Option<Vec<f32>>,

    /// Face embedding vector for recognition/comparison (optional).
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// Universal schema for face detection and embedding responses across Lumen services.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FaceV1 {
    /// Detected faces with bounding boxes and metadata.
    #[validate(nested)]
    pub faces: Vec<Face>,

    /// Number of detected faces.
    #[validate(range(min = 0))]
    pub count: usize,

    /// Model identifier.
    #[validate(length(min = 1))]
    pub model_id: String,
}

impl FaceV1 {
    pub fn new(faces: Vec<Face>, model_id: impl Into<String>) -> Self {
        let count = faces.len();
        Self {
            faces,
            count,
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

    use super::{BboxItem, Face, FaceV1};

    #[test]
    fn serializes_bbox_items_as_numbers() {
        let item = BboxItem::from(12.5);

        let encoded = serde_json::to_string(&item).expect("bbox item serializes");

        assert_eq!(encoded, "12.5");
    }

    #[test]
    fn validates_face_constraints() {
        let invalid = Face {
            bbox: vec![BboxItem(-1.0), BboxItem(1.0), BboxItem(2.0)],
            confidence: 1.5,
            landmarks: None,
            embedding: None,
        };

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn allows_optional_face_vectors_to_be_absent() {
        let value = json!({
            "bbox": [0.0, 1.0, 2.0, 3.0],
            "confidence": 0.9
        });

        let parsed = serde_json::from_value::<Face>(value).expect("optional fields are absent");

        assert_eq!(parsed.landmarks, None);
        assert_eq!(parsed.embedding, None);
    }

    #[test]
    fn forbids_extra_face_fields() {
        let value = json!({
            "bbox": [0.0, 1.0, 2.0, 3.0],
            "confidence": 0.9,
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<Face>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn encodes_valid_face_schema_as_json_bytes() {
        let face = Face {
            bbox: vec![BboxItem(0.0), BboxItem(1.0), BboxItem(2.0), BboxItem(3.0)],
            confidence: 0.9,
            landmarks: Some(vec![0.1, 0.2]),
            embedding: None,
        };
        let response = FaceV1::new(vec![face], "retinaface-r50");

        let bytes = response
            .to_json_bytes()
            .expect("valid face response encodes");
        let decoded: FaceV1 =
            serde_json::from_slice(&bytes).expect("encoded face response decodes");

        assert_eq!(decoded, response);
    }
}
