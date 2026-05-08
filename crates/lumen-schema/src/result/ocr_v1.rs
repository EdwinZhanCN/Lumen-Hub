use bytes::Bytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

use super::SchemaEncodeError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BoxItem(
    /// Point coordinates [x, y].
    #[schemars(length(min = 2, max = 2))]
    pub Vec<i64>,
);

impl From<[i64; 2]> for BoxItem {
    fn from(value: [i64; 2]) -> Self {
        Self(value.into())
    }
}

impl Validate for BoxItem {
    fn validate(&self) -> Result<(), ValidationErrors> {
        if self.0.len() == 2 {
            return Ok(());
        }

        let mut errors = ValidationErrors::new();
        errors.add("root", ValidationError::new("length"));
        Err(errors)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Item {
    /// Polygon coordinates defining the text region (usually 4 points for
    /// rotated rectangle: TL, TR, BR, BL).
    #[serde(rename = "box")]
    #[schemars(rename = "box")]
    #[validate(length(min = 3), nested)]
    pub box_: Vec<BoxItem>,

    /// Recognized text content.
    pub text: String,

    /// Recognition confidence score.
    #[validate(range(min = 0.0, max = 1.0))]
    pub confidence: f32,
}

/// Universal schema for OCR text detection and recognition responses across
/// Lumen services.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OCRV1 {
    /// Detected text regions with content and metadata.
    #[validate(nested)]
    pub items: Vec<Item>,

    /// Number of detected text regions.
    #[validate(range(min = 0))]
    pub count: usize,

    /// Model identifier (combined detection and recognition models).
    #[validate(length(min = 1))]
    pub model_id: String,
}

impl OCRV1 {
    pub fn new(items: Vec<Item>, model_id: impl Into<String>) -> Self {
        let count = items.len();
        Self {
            items,
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

    use super::{BoxItem, Item, OCRV1};

    #[test]
    fn serializes_box_items_as_coordinate_arrays() {
        let item = BoxItem::from([12, 34]);

        let encoded = serde_json::to_string(&item).expect("box item serializes");

        assert_eq!(encoded, "[12,34]");
    }

    #[test]
    fn validates_item_constraints() {
        let invalid = Item {
            box_: vec![BoxItem::from([0, 0]), BoxItem::from([1, 1])],
            text: "hello".to_owned(),
            confidence: 1.5,
        };

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn forbids_extra_item_fields() {
        let value = json!({
            "box": [[0, 0], [1, 0], [1, 1]],
            "text": "hello",
            "confidence": 0.9,
            "extra": "forbidden"
        });

        let parsed = serde_json::from_value::<Item>(value);

        assert!(parsed.is_err());
    }

    #[test]
    fn validates_model_id_min_length() {
        let response = OCRV1::new(Vec::new(), "");

        assert!(response.validate().is_err());
    }

    #[test]
    fn allows_empty_items_list() {
        let response = OCRV1::new(Vec::new(), "paddleocr");

        assert!(response.validate().is_ok());
    }

    #[test]
    fn encodes_valid_ocr_schema_as_json_bytes() {
        let item = Item {
            box_: vec![
                BoxItem::from([0, 0]),
                BoxItem::from([10, 0]),
                BoxItem::from([10, 10]),
            ],
            text: "hello".to_owned(),
            confidence: 0.9,
        };
        let response = OCRV1::new(vec![item], "paddleocr");

        let bytes = response
            .to_json_bytes()
            .expect("valid ocr response encodes");
        let decoded: OCRV1 = serde_json::from_slice(&bytes).expect("encoded ocr response decodes");

        assert_eq!(decoded, response);
    }
}
