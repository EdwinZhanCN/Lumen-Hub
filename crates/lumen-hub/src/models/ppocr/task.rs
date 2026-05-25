use std::{collections::HashMap, env, sync::Arc};

use async_trait::async_trait;
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use imageproc::geometric_transformations::{Interpolation, Projection, warp_into};
use lumen_schema::OCRV1;
use lumnn::core::{
    context::MLContext,
    node::{MLNode, MLNodeRef},
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
};
use serde::Deserialize;

use super::nodes::{CtcDecodeNode, DBPostProcessNode};
use crate::service::{
    DEFAULT_TENSOR_MIME, META_MODEL_ID, PREPROCESS_PPOCR_DET, ServiceError, ServiceResult,
    TaskHandler, TaskRequest, TaskResult, TaskSpec, bytes_to_f32_le, is_tensor_input_request,
    parse_source_dimensions, validate_dynamic_det_tensor_request,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SUPPORTED_IMAGE_MIMES: [&str; 4] = ["image/jpeg", "image/png", "image/webp", "image/avif"];

// ---------------------------------------------------------------------------
// Detection config (deserialized from model_info.json)
// ---------------------------------------------------------------------------

/// Detection sub-config for PP-OCR DBNet model.
#[derive(Debug, Clone, Deserialize)]
pub struct PpocrDetConfig {
    /// ONNX component name.
    pub component: String,
    /// ONNX input node name.
    pub input_name: String,
    /// ONNX output node name.
    pub output_name: String,
    /// Per-channel mean for normalization (ImageNet default).
    #[serde(default = "default_det_mean")]
    pub mean: [f32; 3],
    /// Per-channel std for normalization (ImageNet default).
    #[serde(default = "default_det_std")]
    pub std: [f32; 3],
    /// Rescale factor applied before mean/std normalization.
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// Maximum side length for detection resize.
    #[serde(default = "default_limit_side_len")]
    pub limit_side_len: u32,
    /// Probability threshold for binary segmentation.
    #[serde(default = "default_thresh")]
    pub thresh: f32,
    /// Minimum box score to keep a detection.
    #[serde(default = "default_box_thresh")]
    pub box_thresh: f32,
    /// Unclip ratio for box expansion.
    #[serde(default = "default_unclip_ratio")]
    pub unclip_ratio: f32,
}

fn default_det_mean() -> [f32; 3] {
    [0.485, 0.456, 0.406]
}
fn default_det_std() -> [f32; 3] {
    [0.229, 0.224, 0.225]
}
fn default_scale() -> f32 {
    1.0 / 255.0
}
fn default_limit_side_len() -> u32 {
    960
}
fn default_thresh() -> f32 {
    0.3
}
fn default_box_thresh() -> f32 {
    0.6
}
fn default_unclip_ratio() -> f32 {
    1.5
}

// ---------------------------------------------------------------------------
// Recognition config (deserialized from model_info.json)
// ---------------------------------------------------------------------------

/// Recognition sub-config for PP-OCR SVTR/CRNN model.
#[derive(Debug, Clone, Deserialize)]
pub struct PpocrRecConfig {
    /// ONNX component name.
    pub component: String,
    /// ONNX input node name.
    pub input_name: String,
    /// ONNX output node name (CTC logits / softmax).
    pub output_name: String,
    /// Per-channel mean for normalization.
    #[serde(default = "default_rec_mean")]
    pub mean: [f32; 3],
    /// Per-channel std for normalization.
    #[serde(default = "default_rec_std")]
    pub std: [f32; 3],
    /// Rescale factor applied before mean/std normalization.
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// Expected input shape [C, H, W_max] for the recognition model.
    #[serde(default = "default_image_shape")]
    pub image_shape: [usize; 3],
    /// Filename of the character dictionary (in model root dir).
    #[serde(default = "default_character_dict_path")]
    pub character_dict_path: String,
    /// Whether to append a space character to the vocabulary.
    #[serde(default = "default_use_space_char")]
    pub use_space_char: bool,
    /// CTC blank token index.
    #[serde(default = "default_blank_id")]
    pub blank_id: i64,
}

fn default_rec_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_rec_std() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_image_shape() -> [usize; 3] {
    [3, 48, 320]
}
fn default_character_dict_path() -> String {
    "ppocrv5_dict.txt".to_owned()
}
fn default_use_space_char() -> bool {
    true
}
fn default_blank_id() -> i64 {
    0
}

// ---------------------------------------------------------------------------
// PpocrTask
// ---------------------------------------------------------------------------

/// End-to-end PP-OCR task: detection → post-process → recognition → decode.
pub struct PpocrTask {
    spec: TaskSpec,
    context: Arc<MLContext>,
    model_id: String,
    det_config: PpocrDetConfig,
    rec_config: PpocrRecConfig,
    det_node: MLNodeRef,
    rec_node: MLNodeRef,
    db_node: DBPostProcessNode,
    ctc_node: Arc<CtcDecodeNode>,
    vocab: Vec<String>,
}

impl PpocrTask {
    pub fn new(
        name: impl Into<String>,
        context: Arc<MLContext>,
        model_id: String,
        det_config: PpocrDetConfig,
        rec_config: PpocrRecConfig,
        det_node: MLNodeRef,
        rec_node: MLNodeRef,
        db_node: DBPostProcessNode,
        ctc_node: CtcDecodeNode,
        vocab: Vec<String>,
    ) -> ServiceResult<Self> {
        let name = name.into();

        let spec = TaskSpec::new(&name, "PP-OCR end-to-end text detection and recognition")
            .with_input_mimes(image_input_mimes_with_tensor())
            .with_output_mime("application/json;schema=ocr_v1")
            .with_metadata("output_schema", "ocr_v1")
            .with_metadata(META_MODEL_ID, &model_id)
            .with_limit("det_limit_side_len", det_config.limit_side_len.to_string())
            .with_limit("rec_image_height", rec_config.image_shape[1].to_string());

        Ok(Self {
            spec,
            context,
            model_id,
            det_config,
            rec_config,
            det_node,
            rec_node,
            db_node,
            ctc_node: Arc::new(ctc_node),
            vocab,
        })
    }
}

#[async_trait]
impl TaskHandler for PpocrTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let results = if is_tensor_input_request(&request) {
            self.run_ocr_from_det_tensor(&request).await?
        } else {
            let image = image::load_from_memory(&request.payload).map_err(|e| {
                ServiceError::InvalidArgument(format!("failed to decode image: {e}"))
            })?;
            self.run_ocr_from_image(&image).await?
        };

        let json_bytes = results
            .to_json_bytes()
            .map_err(|e| ServiceError::Internal(format!("failed to serialize OCR result: {e}")))?;

        Ok(
            TaskResult::new(json_bytes, "application/json;schema=ocr_v1")
                .with_result_schema("ocr_v1")
                .with_meta(META_MODEL_ID, &self.model_id),
        )
    }
}

impl PpocrTask {
    async fn run_ocr_from_image(&self, image: &DynamicImage) -> ServiceResult<OCRV1> {
        let rgb = image.to_rgb8();
        let (src_h, src_w) = (rgb.height(), rgb.width());

        let (det_input, det_input_h, det_input_w, ratio_h, ratio_w) = det_preprocess(
            &rgb,
            self.det_config.limit_side_len,
            self.det_config.scale,
            &self.det_config.mean,
            &self.det_config.std,
        );

        self.run_ocr_pipeline(
            &rgb,
            &det_input,
            det_input_h,
            det_input_w,
            ratio_h,
            ratio_w,
            src_w,
            src_h,
            |box_coords| *box_coords,
        )
        .await
    }

    async fn run_ocr_from_det_tensor(&self, request: &TaskRequest) -> ServiceResult<OCRV1> {
        let descriptor = validate_dynamic_det_tensor_request(
            request,
            crate::service::DynamicDetTensorValidationOptions {
                dtype: "fp32",
                preprocess_id: PREPROCESS_PPOCR_DET,
            },
        )?;
        let source = parse_source_dimensions(request)?;
        let det_input = bytes_to_f32_le(&request.payload)?;
        let det_input_h = descriptor.shape[2];
        let det_input_w = descriptor.shape[3];
        let ratio_h = det_input_h as f32 / source.height as f32;
        let ratio_w = det_input_w as f32 / source.width as f32;
        let crop_canvas = denormalize_nchw_to_rgb(
            &det_input,
            det_input_h,
            det_input_w,
            self.det_config.scale,
            &self.det_config.mean,
            &self.det_config.std,
        );

        self.run_ocr_pipeline(
            &crop_canvas,
            &det_input,
            det_input_h,
            det_input_w,
            ratio_h,
            ratio_w,
            source.width,
            source.height,
            |box_coords| source_box_to_det_canvas(box_coords, ratio_w, ratio_h),
        )
        .await
    }

    async fn run_ocr_pipeline(
        &self,
        crop_image: &RgbImage,
        det_input: &[f32],
        det_input_h: usize,
        det_input_w: usize,
        ratio_h: f32,
        ratio_w: f32,
        src_w: u32,
        src_h: u32,
        map_box_for_crop: impl Fn(&[f32; 8]) -> [f32; 8],
    ) -> ServiceResult<OCRV1> {
        let det_output = run_detection_node(
            self.det_node.as_ref(),
            self.context.as_ref(),
            &self.det_config.input_name,
            &self.det_config.output_name,
            det_input,
            det_input_h,
            det_input_w,
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("detection inference failed: {e}")))?;

        let boxes = run_db_postprocess(
            &self.db_node,
            self.context.as_ref(),
            &det_output.values,
            &det_output.shape,
            ratio_h,
            ratio_w,
            src_h,
            src_w,
        )
        .await
        .map_err(|e| ServiceError::Internal(format!("DB post-process failed: {e}")))?;

        if ppocr_debug_enabled() {
            eprintln!(
                "ppocr_debug det_output_shape={:?} det_output_len={} boxes={}",
                det_output.shape,
                det_output.values.len(),
                boxes.len()
            );
        }

        if boxes.is_empty() {
            return Ok(OCRV1::new(vec![], &self.model_id));
        }

        // 4. Sort boxes (top-down, left-to-right)
        let sorted_boxes = sort_boxes(&boxes);

        // 5. For each box: crop → perspective warp → recognize → decode
        let mut items = Vec::with_capacity(sorted_boxes.len());

        for (box_idx, box_coords) in sorted_boxes.iter().enumerate() {
            let crop_box = map_box_for_crop(box_coords);
            let crop = match perspective_crop(crop_image, &crop_box) {
                Ok(c) => c,
                Err(err) => {
                    if ppocr_debug_enabled() {
                        eprintln!("ppocr_debug box_idx={} crop_error={}", box_idx, err);
                    }
                    continue;
                }
            };

            let rec_input = rec_preprocess(
                &crop,
                self.rec_config.image_shape[0],
                self.rec_config.image_shape[1] as u32,
                self.rec_config.image_shape[2] as u32,
                self.rec_config.scale,
                &self.rec_config.mean,
                &self.rec_config.std,
            );

            let rec_output = match run_recognition_node(
                self.rec_node.as_ref(),
                self.context.as_ref(),
                &self.rec_config.input_name,
                &self.rec_config.output_name,
                &rec_input,
                &self.rec_config.image_shape,
            )
            .await
            {
                Ok(l) => l,
                Err(err) => {
                    if ppocr_debug_enabled() {
                        eprintln!("ppocr_debug box_idx={} recognition_error={}", box_idx, err);
                    }
                    continue;
                }
            };

            let (indices, conf) = match run_ctc_decode(
                self.ctc_node.as_ref(),
                self.context.as_ref(),
                &rec_output.values,
                &rec_output.shape,
            )
            .await
            {
                Ok((idx, c)) => (idx, c),
                Err(err) => {
                    if ppocr_debug_enabled() {
                        eprintln!("ppocr_debug box_idx={} ctc_error={}", box_idx, err);
                    }
                    continue;
                }
            };

            let text = indices_to_text(&indices, &self.vocab, self.rec_config.blank_id);

            if ppocr_debug_enabled() {
                eprintln!(
                    "ppocr_debug box_idx={} crop={}x{} decoded_len={} conf={:.4} text={:?}",
                    box_idx,
                    crop.width(),
                    crop.height(),
                    text.chars().count(),
                    conf,
                    text
                );
            }

            if !text.is_empty() {
                let box_items: Vec<lumen_schema::BoxItem> = box_coords
                    .chunks(2)
                    .map(|pt| lumen_schema::BoxItem::from([pt[0] as i64, pt[1] as i64]))
                    .collect();

                items.push(lumen_schema::OcrItem {
                    box_: box_items,
                    text,
                    confidence: conf,
                });
            }
        }

        Ok(OCRV1::new(items, &self.model_id))
    }
}

// ---------------------------------------------------------------------------
// Detection preprocessing
// ---------------------------------------------------------------------------

fn det_preprocess(
    img: &RgbImage,
    limit_side_len: u32,
    scale: f32,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> (Vec<f32>, usize, usize, f32, f32) {
    let h = img.height() as f32;
    let w = img.width() as f32;

    let ratio = if h.max(w) > limit_side_len as f32 {
        if h > w {
            limit_side_len as f32 / h
        } else {
            limit_side_len as f32 / w
        }
    } else {
        1.0
    };

    let mut resize_h = ((h * ratio) as u32).max(32);
    let mut resize_w = ((w * ratio) as u32).max(32);
    resize_h = ((resize_h + 31) / 32 * 32).max(32);
    resize_w = ((resize_w + 31) / 32 * 32).max(32);

    let ratio_h = resize_h as f32 / h;
    let ratio_w = resize_w as f32 / w;

    let resized = image::imageops::resize(img, resize_w, resize_h, FilterType::CatmullRom);

    let mut tensor = vec![0.0f32; 3 * resize_h as usize * resize_w as usize];
    for y in 0..resize_h as usize {
        for x in 0..resize_w as usize {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 * scale;
                let normalized = (val - mean[c]) / std[c];
                tensor[c * resize_h as usize * resize_w as usize + y * resize_w as usize + x] =
                    normalized;
            }
        }
    }

    (
        tensor,
        resize_h as usize,
        resize_w as usize,
        ratio_h,
        ratio_w,
    )
}

// ---------------------------------------------------------------------------
// Detection ONNX inference
// ---------------------------------------------------------------------------

async fn run_detection_node(
    node: &dyn MLNode,
    context: &MLContext,
    input_name: &str,
    output_name: &str,
    input_data: &[f32],
    input_h: usize,
    input_w: usize,
) -> Result<DetectionOutput, String> {
    let _input_desc = node
        .input_descriptors()
        .get(input_name)
        .ok_or_else(|| format!("detection node missing input `{input_name}`"))?;

    let expected_len = 3 * input_h * input_w;
    if input_data.len() != expected_len {
        return Err(format!(
            "detection input length mismatch: expected {expected_len}, got {}",
            input_data.len()
        ));
    }

    let actual_shape = vec![1, 3, input_h, input_w];
    let packet_desc = MLPacketDescriptor::new(MLPacketDataType::Float32, actual_shape);
    let packet = MLPacket::from_host_tensor(
        Arc::new(context.clone()),
        packet_desc,
        HostTensor::Float32(input_data.to_vec()),
    )?;

    let mut outputs = node
        .execute(HashMap::from([(input_name.to_owned(), packet)]), context)
        .await?;

    let output_packet = outputs
        .remove(output_name)
        .ok_or_else(|| format!("detection node missing output `{output_name}`"))?;

    let (_ctx, output_desc, output_payload) = output_packet.into_parts()?;
    let values = match output_payload.to_host_tensor() {
        Ok(HostTensor::Float32(v)) => v,
        Ok(other) => return Err(format!("unexpected detection output dtype: {other:?}")),
        Err(e) => return Err(format!("detection output read failed: {e}")),
    };

    Ok(DetectionOutput {
        values,
        shape: output_desc.shape,
    })
}

struct DetectionOutput {
    values: Vec<f32>,
    shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// DB post-process execution
// ---------------------------------------------------------------------------

async fn run_db_postprocess(
    node: &DBPostProcessNode,
    context: &MLContext,
    prob_map: &[f32],
    prob_map_shape: &[usize],
    ratio_h: f32,
    ratio_w: f32,
    src_h: u32,
    src_w: u32,
) -> Result<Vec<[f32; 8]>, String> {
    let ctx = Arc::new(context.clone());

    let prob_desc = node
        .input_descriptors()
        .get("prob_map")
        .ok_or("DB node missing prob_map descriptor")?;
    if prob_map_shape.len() != 4 {
        return Err(format!(
            "detection output rank mismatch: expected 4D tensor, got shape {:?}",
            prob_map_shape
        ));
    }
    if prob_map_shape[0] != 1 || prob_map_shape[1] != 1 {
        return Err(format!(
            "detection output leading dims mismatch: expected [1, 1, H, W], got {:?}",
            prob_map_shape
        ));
    }

    let mut actual_prob_desc = prob_desc.clone();
    actual_prob_desc.shape = prob_map_shape.to_vec();

    let prob_packet = MLPacket::from_host_tensor(
        ctx.clone(),
        actual_prob_desc,
        HostTensor::Float32(prob_map.to_vec()),
    )?;

    let scalar_f32 = |ctx: &Arc<MLContext>, v: f32| -> Result<MLPacket, String> {
        MLPacket::from_host_tensor(
            Arc::clone(ctx),
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1]),
            HostTensor::Float32(vec![v]),
        )
    };
    let scalar_i64 = |ctx: &Arc<MLContext>, v: i64| -> Result<MLPacket, String> {
        MLPacket::from_host_tensor(
            Arc::clone(ctx),
            MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1]),
            HostTensor::Int64(vec![v]),
        )
    };

    let outputs = node
        .execute(
            HashMap::from([
                ("prob_map".to_owned(), prob_packet),
                ("ratio_h".to_owned(), scalar_f32(&ctx, ratio_h)?),
                ("ratio_w".to_owned(), scalar_f32(&ctx, ratio_w)?),
                ("src_h".to_owned(), scalar_i64(&ctx, src_h as i64)?),
                ("src_w".to_owned(), scalar_i64(&ctx, src_w as i64)?),
            ]),
            context,
        )
        .await?;

    let boxes_packet = outputs
        .get("boxes")
        .ok_or("DB node missing `boxes` output")?;

    let boxes_values = match boxes_packet.to_host_tensor().await {
        Ok(HostTensor::Float32(v)) => v,
        Ok(other) => return Err(format!("unexpected boxes dtype: {other:?}")),
        Err(e) => return Err(format!("boxes read failed: {e}")),
    };

    let num = boxes_values.len() / 8;
    let boxes: Vec<[f32; 8]> = (0..num)
        .map(|i| {
            let s = i * 8;
            [
                boxes_values[s],
                boxes_values[s + 1],
                boxes_values[s + 2],
                boxes_values[s + 3],
                boxes_values[s + 4],
                boxes_values[s + 5],
                boxes_values[s + 6],
                boxes_values[s + 7],
            ]
        })
        .collect();

    Ok(boxes)
}

// ---------------------------------------------------------------------------
// Box sorting
// ---------------------------------------------------------------------------

fn sort_boxes(boxes: &[[f32; 8]]) -> Vec<[f32; 8]> {
    let mut sorted: Vec<[f32; 8]> = boxes.to_vec();
    sorted.sort_by(|a, b| {
        let a_y = a[1].min(a[3]).min(a[5]).min(a[7]);
        let b_y = b[1].min(b[3]).min(b[5]).min(b[7]);
        let a_x = a[0].min(a[2]).min(a[4]).min(a[6]);
        let b_x = b[0].min(b[2]).min(b[4]).min(b[6]);
        if (a_y - b_y).abs() < 10.0 {
            a_x.partial_cmp(&b_x).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a_y.partial_cmp(&b_y).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    sorted
}

// ---------------------------------------------------------------------------
// Perspective crop
// ---------------------------------------------------------------------------

fn perspective_crop(img: &RgbImage, box_coords: &[f32; 8]) -> Result<RgbImage, String> {
    let pts: [(f32, f32); 4] = [
        (box_coords[0], box_coords[1]),
        (box_coords[2], box_coords[3]),
        (box_coords[4], box_coords[5]),
        (box_coords[6], box_coords[7]),
    ];

    let crop_w = dist(pts[0], pts[1]).max(dist(pts[2], pts[3])) as u32;
    let crop_h = dist(pts[0], pts[3]).max(dist(pts[1], pts[2])) as u32;

    if crop_w == 0 || crop_h == 0 {
        return Err("zero-size crop".to_owned());
    }

    let dst: [(f32, f32); 4] = [
        (0.0, 0.0),
        (crop_w as f32 - 1.0, 0.0),
        (crop_w as f32 - 1.0, crop_h as f32 - 1.0),
        (0.0, crop_h as f32 - 1.0),
    ];

    let projection =
        get_perspective_transform(&pts, &dst).ok_or("failed to compute perspective transform")?;

    let warped = warp_perspective(img, &projection, crop_w, crop_h);

    let (wh, ww) = (warped.height(), warped.width());
    if wh as f32 / ww as f32 >= 1.5 {
        Ok(image::imageops::rotate90(&warped))
    } else {
        Ok(warped)
    }
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn get_perspective_transform(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Option<Projection> {
    Projection::from_control_points(*src, *dst)
}

fn warp_perspective(img: &RgbImage, projection: &Projection, out_w: u32, out_h: u32) -> RgbImage {
    let mut out = RgbImage::new(out_w, out_h);
    warp_into(
        img,
        projection,
        Interpolation::Nearest,
        Rgb([0u8, 0u8, 0u8]),
        &mut out,
    );
    out
}

// ---------------------------------------------------------------------------
// Recognition preprocessing
// ---------------------------------------------------------------------------

fn rec_preprocess(
    img: &RgbImage,
    target_c: usize,
    target_h: u32,
    target_w: u32,
    scale: f32,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    assert_eq!(target_c, 3, "PP-OCR recognition expects RGB input");

    let h = img.height() as f32;
    let w = img.width() as f32;
    let ratio = target_h as f32 / h;
    let resize_w = ((w * ratio).ceil() as u32).clamp(1, target_w);

    let resized = image::imageops::resize(img, resize_w, target_h, FilterType::CatmullRom);

    let rh = target_h as usize;
    let rw = resize_w as usize;
    let tw = target_w as usize;
    let mut tensor = Vec::with_capacity(target_c * rh * tw);
    for c in 0..target_c {
        tensor.extend(std::iter::repeat_n(
            (0.0f32 * scale - mean[c]) / std[c],
            rh * tw,
        ));
    }

    for y in 0..rh {
        for x in 0..rw {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..target_c {
                let val = pixel[c] as f32 * scale;
                tensor[c * rh * tw + y * tw + x] = (val - mean[c]) / std[c];
            }
        }
    }

    tensor
}

// ---------------------------------------------------------------------------
// Recognition ONNX inference
// ---------------------------------------------------------------------------

async fn run_recognition_node(
    node: &dyn MLNode,
    context: &MLContext,
    input_name: &str,
    output_name: &str,
    input_data: &[f32],
    input_shape: &[usize; 3],
) -> Result<RecognitionOutput, String> {
    let _input_desc = node
        .input_descriptors()
        .get(input_name)
        .ok_or_else(|| format!("recognition node missing input `{input_name}`"))?;

    let [c, h, w] = *input_shape;
    let expected_len = c * h * w;
    if input_data.len() != expected_len {
        return Err(format!(
            "recognition input length mismatch: expected {expected_len}, got {}",
            input_data.len()
        ));
    }

    let actual_shape = vec![1, c, h, w];
    let packet_desc = MLPacketDescriptor::new(MLPacketDataType::Float32, actual_shape);
    let packet = MLPacket::from_host_tensor(
        Arc::new(context.clone()),
        packet_desc,
        HostTensor::Float32(input_data.to_vec()),
    )?;

    let mut outputs = node
        .execute(HashMap::from([(input_name.to_owned(), packet)]), context)
        .await?;

    let output_packet = outputs
        .remove(output_name)
        .ok_or_else(|| format!("recognition node missing output `{output_name}`"))?;

    let (_ctx, output_desc, output_payload) = output_packet.into_parts()?;
    let values = match output_payload.to_host_tensor() {
        Ok(HostTensor::Float32(v)) => v,
        Ok(other) => return Err(format!("unexpected recognition output dtype: {other:?}")),
        Err(e) => return Err(format!("recognition output read failed: {e}")),
    };

    Ok(RecognitionOutput {
        values,
        shape: output_desc.shape,
    })
}

struct RecognitionOutput {
    values: Vec<f32>,
    shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// CTC decode execution
// ---------------------------------------------------------------------------

async fn run_ctc_decode(
    node: &CtcDecodeNode,
    context: &MLContext,
    logits: &[f32],
    logits_shape: &[usize],
) -> Result<(Vec<i64>, f32), String> {
    let ctx = Arc::new(context.clone());

    let logit_desc = node
        .input_descriptors()
        .get("logits")
        .ok_or("CTC node missing logits descriptor")?;

    if logits_shape.len() != 3 {
        return Err(format!(
            "recognition output rank mismatch: expected 3D tensor, got shape {:?}",
            logits_shape
        ));
    }

    let mut actual_logit_desc = logit_desc.clone();
    actual_logit_desc.shape = logits_shape.to_vec();

    let logit_packet =
        MLPacket::from_host_tensor(ctx, actual_logit_desc, HostTensor::Float32(logits.to_vec()))?;

    let outputs = node
        .execute(
            HashMap::from([("logits".to_owned(), logit_packet)]),
            context,
        )
        .await?;

    let indices_packet = outputs
        .get("text_indices")
        .ok_or("CTC node missing `text_indices` output")?;
    let confidence_packet = outputs
        .get("confidence")
        .ok_or("CTC node missing `confidence` output")?;

    let indices = match indices_packet.to_host_tensor().await {
        Ok(HostTensor::Int64(v)) => v,
        Ok(other) => return Err(format!("unexpected indices dtype: {other:?}")),
        Err(e) => return Err(format!("indices read failed: {e}")),
    };

    let conf = match confidence_packet.to_host_tensor().await {
        Ok(HostTensor::Float32(v)) => *v.first().unwrap_or(&0.0),
        Ok(other) => return Err(format!("unexpected confidence dtype: {other:?}")),
        Err(e) => return Err(format!("confidence read failed: {e}")),
    };

    Ok((indices, conf))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn indices_to_text(indices: &[i64], vocab: &[String], blank_id: i64) -> String {
    indices
        .iter()
        .filter_map(|&idx| {
            let vocab_idx = if idx < 0 || idx == blank_id {
                return None;
            } else if idx > blank_id {
                idx - 1
            } else {
                idx
            };

            (vocab_idx >= 0 && (vocab_idx as usize) < vocab.len())
                .then(|| vocab[vocab_idx as usize].as_str())
        })
        .collect::<Vec<&str>>()
        .join("")
}

fn denormalize_nchw_to_rgb(
    values: &[f32],
    height: usize,
    width: usize,
    scale: f32,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> RgbImage {
    let plane = height * width;
    let mut image = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let mut rgb = [0u8; 3];
            for channel in 0..3 {
                let normalized = values[channel * plane + y * width + x];
                let pixel = normalized * std[channel] + mean[channel];
                rgb[channel] = (pixel / scale).round().clamp(0.0, 255.0) as u8;
            }
            image.put_pixel(x as u32, y as u32, Rgb(rgb));
        }
    }
    image
}

fn source_box_to_det_canvas(box_coords: &[f32; 8], ratio_w: f32, ratio_h: f32) -> [f32; 8] {
    let mut mapped = *box_coords;
    for index in (0..8).step_by(2) {
        mapped[index] *= ratio_w;
        mapped[index + 1] *= ratio_h;
    }
    mapped
}

fn image_input_mimes_with_tensor() -> Vec<String> {
    SUPPORTED_IMAGE_MIMES
        .iter()
        .copied()
        .chain(std::iter::once(DEFAULT_TENSOR_MIME))
        .map(str::to_owned)
        .collect()
}

fn ppocr_debug_enabled() -> bool {
    matches!(
        env::var("LUMEN_PPOCR_DEBUG").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_boxes_top_down() {
        let boxes = vec![
            [100.0, 100.0, 200.0, 100.0, 200.0, 120.0, 100.0, 120.0],
            [100.0, 10.0, 200.0, 10.0, 200.0, 30.0, 100.0, 30.0],
        ];
        let sorted = sort_boxes(&boxes);
        assert!(sorted[0][1] < sorted[1][1]);
    }

    #[test]
    fn test_dist() {
        let d = dist((0.0, 0.0), (3.0, 4.0));
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_perspective_transform_identity() {
        let src: [(f32, f32); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst: [(f32, f32); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let projection = get_perspective_transform(&src, &dst).unwrap();
        let point = projection * (0.25, 0.75);
        assert!((point.0 - 0.25).abs() < 1e-5);
        assert!((point.1 - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_indices_to_text() {
        let vocab = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        assert_eq!(indices_to_text(&[1, 2, 3], &vocab, 0), "abc");
    }

    #[test]
    fn test_indices_to_text_skips_invalid_indices() {
        let vocab = vec!["h".to_owned(), "i".to_owned()];
        assert_eq!(indices_to_text(&[1, 0, -1, 2, 99], &vocab, 0), "hi");
    }

    #[test]
    fn test_indices_to_text_with_nonzero_blank_id() {
        let vocab = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        assert_eq!(indices_to_text(&[0, 1, 3], &vocab, 2), "abc");
    }

    #[test]
    fn test_image_input_mimes_include_tensor_mime() {
        assert!(image_input_mimes_with_tensor().contains(&DEFAULT_TENSOR_MIME.to_owned()));
    }

    #[test]
    fn test_source_box_to_det_canvas_scales_coordinates() {
        let mapped =
            source_box_to_det_canvas(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 2.0, 0.5);
        assert_eq!(mapped, [20.0, 10.0, 60.0, 20.0, 100.0, 30.0, 140.0, 40.0]);
    }

    #[test]
    fn test_image_input_mimes_only_advertise_decodable_formats() {
        assert_eq!(
            image_input_mimes_with_tensor()
                .into_iter()
                .filter(|mime| mime != DEFAULT_TENSOR_MIME)
                .collect::<Vec<_>>(),
            vec![
                "image/jpeg".to_owned(),
                "image/png".to_owned(),
                "image/webp".to_owned(),
                "image/avif".to_owned()
            ]
        );
    }

    #[test]
    fn test_rec_preprocess_pads_to_configured_width() {
        let image = RgbImage::from_pixel(8, 16, Rgb([255, 255, 255]));
        let tensor = rec_preprocess(&image, 3, 48, 320, 1.0 / 255.0, &[0.5; 3], &[0.5; 3]);

        assert_eq!(tensor.len(), 3 * 48 * 320);
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor[48 * 320 - 1], -1.0);
    }

    #[test]
    fn test_invalid_image_bytes_report_decode_error() {
        let err = image::load_from_memory(b"not-an-image")
            .map_err(|e| ServiceError::InvalidArgument(format!("failed to decode image: {e}")))
            .unwrap_err();

        assert!(err.to_string().contains("failed to decode image"));
    }
}
