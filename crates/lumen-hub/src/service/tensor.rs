use std::collections::HashMap;

use bytes::Bytes;

use crate::service::{ServiceError, ServiceResult, TaskRequest};

pub const DEFAULT_TENSOR_MIME: &str = "application/octet-stream";

pub const META_INPUT_KIND: &str = "lumen.input.kind";
pub const META_OUTPUT_KIND: &str = "lumen.output.kind";
pub const INPUT_KIND_RAW: &str = "raw";
pub const INPUT_KIND_TENSOR: &str = "tensor";
pub const OUTPUT_KIND_TENSOR: &str = "tensor";

pub const META_TENSOR_DTYPE: &str = "lumen.tensor.dtype";
pub const META_TENSOR_SHAPE: &str = "lumen.tensor.shape";
pub const META_TENSOR_LAYOUT: &str = "lumen.tensor.layout";
pub const META_TENSOR_FORMAT: &str = "lumen.tensor.format";
pub const META_TENSOR_BYTE_ORDER: &str = "lumen.tensor.byte_order";

pub const META_OUTPUT_TENSOR_DTYPE: &str = "lumen.output.tensor.dtype";
pub const META_OUTPUT_TENSOR_SHAPE: &str = "lumen.output.tensor.shape";
pub const META_OUTPUT_TENSOR_LAYOUT: &str = "lumen.output.tensor.layout";
pub const META_OUTPUT_TENSOR_FORMAT: &str = "lumen.output.tensor.format";
pub const META_OUTPUT_TENSOR_BYTE_ORDER: &str = "lumen.output.tensor.byte_order";

pub const META_PREPROCESS_ID: &str = "lumen.preprocess.id";
pub const META_PREPROCESS_SKIP: &str = "lumen.preprocess.skip";
pub const META_MODEL_ID: &str = "lumen.model.id";
pub const META_MODEL_VERSION: &str = "lumen.model.version";

pub const META_SOURCE_WIDTH: &str = "lumen.source.width";
pub const META_SOURCE_HEIGHT: &str = "lumen.source.height";
pub const META_LETTERBOX_SCALE: &str = "lumen.letterbox.scale";
pub const META_LETTERBOX_PAD_X: &str = "lumen.letterbox.pad_x";
pub const META_LETTERBOX_PAD_Y: &str = "lumen.letterbox.pad_y";

pub const PREPROCESS_BIOCLIP2_224_IMAGE: &str = "bioclip2_224_image_v1";
pub const PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE: &str = "siglip2_base_patch16_224_image_v1";
pub const PREPROCESS_SIGLIP2_SO400M_PATCH14_384_IMAGE: &str = "siglip2_so400m_patch14_384_image_v1";
pub const PREPROCESS_CLIP_IMAGE: &str = PREPROCESS_BIOCLIP2_224_IMAGE;
pub const PREPROCESS_SIGLIP_IMAGE: &str = PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE;
pub const PREPROCESS_PPOCR_DET: &str = "ppocr_det_v1";
pub const PREPROCESS_INSIGHTFACE_DET: &str = "insightface_det_v1";

pub const IMAGE_TENSOR_LAYOUT: &str = "NCHW";

pub const TENSOR_FORMAT_CONTIGUOUS: &str = "contiguous";
pub const TENSOR_BYTE_ORDER_LITTLE: &str = "little";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub layout: String,
    pub format: String,
    pub byte_order: String,
}

#[derive(Debug, Clone)]
pub struct TensorValidationOptions<'a> {
    pub dtype: &'a str,
    pub layout: &'a str,
    pub preprocess_id: &'a str,
}

#[derive(Debug, Clone)]
pub struct FixedShapeTensorValidationOptions<'a> {
    pub dtype: &'a str,
    pub layout: &'a str,
    pub preprocess_id: &'a str,
    pub expected_shape: &'a [usize],
}

#[derive(Debug, Clone)]
pub struct DynamicDetTensorValidationOptions<'a> {
    pub dtype: &'a str,
    pub preprocess_id: &'a str,
}

pub fn is_tensor_input_request(request: &TaskRequest) -> bool {
    normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() == Some(INPUT_KIND_TENSOR)
}

pub fn validate_tensor_request(
    request: &TaskRequest,
    options: TensorValidationOptions<'_>,
) -> ServiceResult<TensorDescriptor> {
    if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() != Some(INPUT_KIND_TENSOR) {
        return Err(ServiceError::InvalidArgument(format!(
            "tensor request metadata `{META_INPUT_KIND}` must be `{INPUT_KIND_TENSOR}`"
        )));
    }
    if normalized_mime(&request.payload_mime) != DEFAULT_TENSOR_MIME {
        return Err(ServiceError::InvalidArgument(format!(
            "tensor request payload_mime must be `{DEFAULT_TENSOR_MIME}`"
        )));
    }
    require_meta(&request.meta, META_PREPROCESS_SKIP).and_then(|value| {
        if value == "true" {
            Ok(())
        } else {
            Err(ServiceError::InvalidArgument(format!(
                "tensor request metadata `{META_PREPROCESS_SKIP}` must be `true`"
            )))
        }
    })?;
    require_exact(&request.meta, META_TENSOR_DTYPE, options.dtype)?;
    require_exact(&request.meta, META_TENSOR_LAYOUT, options.layout)?;
    require_exact(&request.meta, META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)?;
    require_exact(
        &request.meta,
        META_TENSOR_BYTE_ORDER,
        TENSOR_BYTE_ORDER_LITTLE,
    )?;
    require_exact(&request.meta, META_PREPROCESS_ID, options.preprocess_id)?;

    let shape_text = require_meta(&request.meta, META_TENSOR_SHAPE)?;
    let shape = parse_shape(shape_text)?;
    let expected_bytes = shape
        .iter()
        .try_fold(element_size(options.dtype), |acc, dim| {
            acc.checked_mul(*dim)
        })
        .ok_or_else(|| {
            ServiceError::InvalidArgument("tensor request shape byte length overflow".to_owned())
        })?;
    if request.payload.len() != expected_bytes {
        return Err(ServiceError::InvalidArgument(format!(
            "tensor request payload byte length mismatch: expected {expected_bytes}, got {}",
            request.payload.len()
        )));
    }

    Ok(TensorDescriptor {
        dtype: options.dtype.to_owned(),
        shape,
        layout: options.layout.to_owned(),
        format: TENSOR_FORMAT_CONTIGUOUS.to_owned(),
        byte_order: TENSOR_BYTE_ORDER_LITTLE.to_owned(),
    })
}

pub fn validate_fixed_shape_tensor_request(
    request: &TaskRequest,
    options: FixedShapeTensorValidationOptions<'_>,
) -> ServiceResult<TensorDescriptor> {
    let descriptor = validate_tensor_request(
        request,
        TensorValidationOptions {
            dtype: options.dtype,
            layout: options.layout,
            preprocess_id: options.preprocess_id,
        },
    )?;
    if descriptor.shape != options.expected_shape {
        return Err(ServiceError::InvalidArgument(format!(
            "tensor shape must be {:?}, got {:?}",
            options.expected_shape, descriptor.shape
        )));
    }
    Ok(descriptor)
}

pub fn validate_dynamic_det_tensor_request(
    request: &TaskRequest,
    options: DynamicDetTensorValidationOptions<'_>,
) -> ServiceResult<TensorDescriptor> {
    let descriptor = validate_tensor_request(
        request,
        TensorValidationOptions {
            dtype: options.dtype,
            layout: IMAGE_TENSOR_LAYOUT,
            preprocess_id: options.preprocess_id,
        },
    )?;
    if descriptor.shape.len() != 4
        || descriptor.shape[0] != 1
        || descriptor.shape[1] != 3
        || !is_multiple_of_32(descriptor.shape[2])
        || !is_multiple_of_32(descriptor.shape[3])
    {
        return Err(ServiceError::InvalidArgument(format!(
            "detection tensor shape must be [1,3,H,W] with H and W multiples of 32, got {:?}",
            descriptor.shape
        )));
    }
    Ok(descriptor)
}

pub fn parse_source_dimensions(request: &TaskRequest) -> ServiceResult<SourceDimensions> {
    let width = parse_positive_u32_meta(request, META_SOURCE_WIDTH)?;
    let height = parse_positive_u32_meta(request, META_SOURCE_HEIGHT)?;
    Ok(SourceDimensions { width, height })
}

pub fn parse_letterbox_transform(request: &TaskRequest) -> ServiceResult<LetterboxTransform> {
    let scale = parse_finite_f32_meta(request, META_LETTERBOX_SCALE)?;
    let pad_x = parse_finite_f32_meta(request, META_LETTERBOX_PAD_X)?;
    let pad_y = parse_finite_f32_meta(request, META_LETTERBOX_PAD_Y)?;
    if scale <= 0.0 {
        return Err(ServiceError::InvalidArgument(format!(
            "`{META_LETTERBOX_SCALE}` must be greater than zero"
        )));
    }
    Ok(LetterboxTransform {
        scale,
        pad_x,
        pad_y,
    })
}

pub fn tensor_response_meta(
    dtype: &str,
    shape: &[usize],
    layout: &str,
    preprocess_id: &str,
    model_id: &str,
    model_version: &str,
) -> HashMap<String, String> {
    HashMap::from([
        (META_OUTPUT_KIND.to_owned(), OUTPUT_KIND_TENSOR.to_owned()),
        (META_OUTPUT_TENSOR_DTYPE.to_owned(), dtype.to_owned()),
        (META_OUTPUT_TENSOR_SHAPE.to_owned(), shape_json(shape)),
        (META_OUTPUT_TENSOR_LAYOUT.to_owned(), layout.to_owned()),
        (
            META_OUTPUT_TENSOR_FORMAT.to_owned(),
            TENSOR_FORMAT_CONTIGUOUS.to_owned(),
        ),
        (
            META_OUTPUT_TENSOR_BYTE_ORDER.to_owned(),
            TENSOR_BYTE_ORDER_LITTLE.to_owned(),
        ),
        (META_PREPROCESS_ID.to_owned(), preprocess_id.to_owned()),
        (META_MODEL_ID.to_owned(), model_id.to_owned()),
        (META_MODEL_VERSION.to_owned(), model_version.to_owned()),
    ])
}

pub fn f32_to_le_bytes(values: &[f32]) -> Bytes {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Bytes::from(bytes)
}

pub fn f16_to_le_bytes(values: &[half::f16]) -> Bytes {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Bytes::from(bytes)
}

pub fn i64_to_le_bytes(values: &[i64]) -> Bytes {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Bytes::from(bytes)
}

pub fn bytes_to_f32_le(bytes: &[u8]) -> ServiceResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(ServiceError::InvalidArgument(
            "fp32 tensor payload byte length is not divisible by 4".to_owned(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length is 4")))
        .collect())
}

pub fn bytes_to_f16_le(bytes: &[u8]) -> ServiceResult<Vec<half::f16>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<half::f16>()) {
        return Err(ServiceError::InvalidArgument(
            "fp16 tensor payload byte length is not divisible by 2".to_owned(),
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            half::f16::from_bits(u16::from_le_bytes(
                chunk.try_into().expect("chunk length is 2"),
            ))
        })
        .collect())
}

pub fn bytes_to_i64_le(bytes: &[u8]) -> ServiceResult<Vec<i64>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<i64>()) {
        return Err(ServiceError::InvalidArgument(
            "int64 tensor payload byte length is not divisible by 8".to_owned(),
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunk length is 8")))
        .collect())
}

pub fn shape_json(shape: &[usize]) -> String {
    serde_json::to_string(shape).expect("shape serializes")
}

fn parse_shape(input: &str) -> ServiceResult<Vec<usize>> {
    let shape = serde_json::from_str::<Vec<usize>>(input).map_err(|err| {
        ServiceError::InvalidArgument(format!("invalid tensor shape metadata `{input}`: {err}"))
    })?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(ServiceError::InvalidArgument(
            "tensor shape must contain positive dimensions".to_owned(),
        ));
    }
    Ok(shape)
}

fn element_size(dtype: &str) -> usize {
    match dtype {
        "fp32" => 4,
        "fp16" => 2,
        "uint8" => 1,
        "int64" => 8,
        _ => 0,
    }
}

fn require_exact(
    meta: &HashMap<String, String>,
    key: &'static str,
    expected: &str,
) -> ServiceResult<()> {
    let value = require_meta(meta, key)?;
    if value != expected {
        return Err(ServiceError::InvalidArgument(format!(
            "tensor metadata `{key}` must be `{expected}`, got `{value}`"
        )));
    }
    Ok(())
}

fn require_meta<'a>(
    meta: &'a HashMap<String, String>,
    key: &'static str,
) -> ServiceResult<&'a str> {
    meta.get(key)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ServiceError::InvalidArgument(format!("missing tensor metadata `{key}`")))
}

fn normalized_meta(value: Option<&String>) -> Option<String> {
    value.map(|value| value.trim().to_ascii_lowercase())
}

fn normalized_mime(value: &str) -> String {
    value
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

fn is_multiple_of_32(value: usize) -> bool {
    value >= 32 && value.is_multiple_of(32)
}

fn parse_positive_u32_meta(request: &TaskRequest, key: &'static str) -> ServiceResult<u32> {
    let value = require_meta(&request.meta, key)?;
    let parsed = value.parse::<u32>().map_err(|err| {
        ServiceError::InvalidArgument(format!("invalid `{key}` `{value}`: {err}"))
    })?;
    if parsed == 0 {
        return Err(ServiceError::InvalidArgument(format!(
            "`{key}` must be greater than zero"
        )));
    }
    Ok(parsed)
}

fn parse_finite_f32_meta(request: &TaskRequest, key: &'static str) -> ServiceResult<f32> {
    let value = require_meta(&request.meta, key)?;
    let parsed = value.parse::<f32>().map_err(|err| {
        ServiceError::InvalidArgument(format!("invalid `{key}` `{value}`: {err}"))
    })?;
    if !parsed.is_finite() {
        return Err(ServiceError::InvalidArgument(format!(
            "`{key}` must be finite"
        )));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_tensor_request_metadata_and_size() {
        let request = TaskRequest::new(vec![0; 1 * 7 * 896 * 4], DEFAULT_TENSOR_MIME)
            .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
            .with_meta(META_TENSOR_DTYPE, "fp32")
            .with_meta(META_TENSOR_SHAPE, "[1,7,896]")
            .with_meta(META_TENSOR_LAYOUT, "BSH")
            .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
            .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
            .with_meta(META_PREPROCESS_ID, "decoder_inputs_v1")
            .with_meta(META_PREPROCESS_SKIP, "true");

        let descriptor = validate_tensor_request(
            &request,
            TensorValidationOptions {
                dtype: "fp32",
                layout: "BSH",
                preprocess_id: "decoder_inputs_v1",
            },
        )
        .unwrap();

        assert_eq!(descriptor.shape, vec![1, 7, 896]);
    }

    #[test]
    fn rejects_wrong_tensor_layout() {
        let request = TaskRequest::new(vec![0; 4], DEFAULT_TENSOR_MIME)
            .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
            .with_meta(META_TENSOR_DTYPE, "fp32")
            .with_meta(META_TENSOR_SHAPE, "[1]")
            .with_meta(META_TENSOR_LAYOUT, "NCHW")
            .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
            .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
            .with_meta(META_PREPROCESS_ID, "decoder_inputs_v1")
            .with_meta(META_PREPROCESS_SKIP, "true");

        assert!(
            validate_tensor_request(
                &request,
                TensorValidationOptions {
                    dtype: "fp32",
                    layout: "BSH",
                    preprocess_id: "decoder_inputs_v1",
                },
            )
            .is_err()
        );
    }

    #[test]
    fn validates_dynamic_det_tensor_shape() {
        let request = TaskRequest::new(vec![0; 1 * 3 * 736 * 1280 * 4], DEFAULT_TENSOR_MIME)
            .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
            .with_meta(META_TENSOR_DTYPE, "fp32")
            .with_meta(META_TENSOR_SHAPE, "[1,3,736,1280]")
            .with_meta(META_TENSOR_LAYOUT, IMAGE_TENSOR_LAYOUT)
            .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
            .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
            .with_meta(META_PREPROCESS_ID, PREPROCESS_PPOCR_DET)
            .with_meta(META_PREPROCESS_SKIP, "true");

        let descriptor = validate_dynamic_det_tensor_request(
            &request,
            DynamicDetTensorValidationOptions {
                dtype: "fp32",
                preprocess_id: PREPROCESS_PPOCR_DET,
            },
        )
        .unwrap();

        assert_eq!(descriptor.shape, vec![1, 3, 736, 1280]);
    }

    #[test]
    fn rejects_det_tensor_shape_not_aligned_to_32() {
        let request = TaskRequest::new(vec![0; 1 * 3 * 100 * 100 * 4], DEFAULT_TENSOR_MIME)
            .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
            .with_meta(META_TENSOR_DTYPE, "fp32")
            .with_meta(META_TENSOR_SHAPE, "[1,3,100,100]")
            .with_meta(META_TENSOR_LAYOUT, IMAGE_TENSOR_LAYOUT)
            .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
            .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
            .with_meta(META_PREPROCESS_ID, PREPROCESS_PPOCR_DET)
            .with_meta(META_PREPROCESS_SKIP, "true");

        assert!(
            validate_dynamic_det_tensor_request(
                &request,
                DynamicDetTensorValidationOptions {
                    dtype: "fp32",
                    preprocess_id: PREPROCESS_PPOCR_DET,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn parses_source_and_letterbox_metadata() {
        let request = TaskRequest::new(Vec::<u8>::new(), DEFAULT_TENSOR_MIME)
            .with_meta(META_SOURCE_WIDTH, "1920")
            .with_meta(META_SOURCE_HEIGHT, "1080")
            .with_meta(META_LETTERBOX_SCALE, "0.3333333")
            .with_meta(META_LETTERBOX_PAD_X, "0")
            .with_meta(META_LETTERBOX_PAD_Y, "140");

        let source = parse_source_dimensions(&request).unwrap();
        let transform = parse_letterbox_transform(&request).unwrap();

        assert_eq!(source.width, 1920);
        assert_eq!(source.height, 1080);
        assert_eq!(transform.pad_y, 140.0);
    }
}
