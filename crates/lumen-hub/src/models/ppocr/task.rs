use std::{env, sync::Arc};

use async_trait::async_trait;
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use imageproc::geometric_transformations::{Interpolation, Projection, warp_into};
use lumen_schema::OCRV1;
use serde::Deserialize;

use super::model::{PpocrClassificationModel, PpocrDetectionModel, PpocrRecognitionModel};
use super::postprocess::{ctc_greedy_decode, detect_boxes};
use crate::{
    inference_worker,
    service::{
        DEFAULT_TENSOR_MIME, META_MODEL_ID, PREPROCESS_PPOCR_DET, ServiceError, ServiceResult,
        TaskHandler, TaskRequest, TaskResult, TaskSpec, bytes_to_f32_le, is_tensor_input_request,
        parse_source_dimensions, validate_dynamic_det_tensor_request,
    },
};

const SUPPORTED_IMAGE_MIMES: [&str; 4] = ["image/jpeg", "image/png", "image/webp", "image/avif"];

// ---------------------------------------------------------------------------
// Detection config (deserialized from model_info.json)
// ---------------------------------------------------------------------------

/// Detection sub-config for PP-OCR DBNet model.
#[derive(Debug, Clone, Deserialize)]
pub struct PpocrDetConfig {
    /// Model component name.
    pub component: String,
    /// Input node name (kept for metadata compatibility).
    pub input_name: String,
    /// Output node name (kept for metadata compatibility).
    pub output_name: String,
    #[serde(default = "default_det_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_det_std")]
    pub std: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default = "default_limit_side_len")]
    pub limit_side_len: u32,
    #[serde(default = "default_thresh")]
    pub thresh: f32,
    #[serde(default = "default_box_thresh")]
    pub box_thresh: f32,
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
    pub component: String,
    pub input_name: String,
    pub output_name: String,
    #[serde(default = "default_rec_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_rec_std")]
    pub std: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default = "default_image_shape")]
    pub image_shape: [usize; 3],
    #[serde(default = "default_character_dict_path")]
    pub character_dict_path: String,
    #[serde(default = "default_use_space_char")]
    pub use_space_char: bool,
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
// Text-line orientation classification config (server pack only; optional)
// ---------------------------------------------------------------------------

/// Orientation classifier sub-config. When present, each detected crop is checked
/// for being upside-down (180°) and rotated before recognition.
#[derive(Debug, Clone, Deserialize)]
pub struct PpocrClsConfig {
    pub component: String,
    #[serde(default)]
    pub input_name: String,
    #[serde(default)]
    pub output_name: String,
    #[serde(default = "default_rec_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_rec_std")]
    pub std: [f32; 3],
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default = "default_cls_image_shape")]
    pub image_shape: [usize; 3],
    #[serde(default = "default_cls_labels")]
    pub labels: Vec<String>,
    #[serde(default = "default_cls_thresh")]
    pub thresh: f32,
}

fn default_cls_image_shape() -> [usize; 3] {
    [3, 80, 160]
}
fn default_cls_labels() -> Vec<String> {
    vec!["0".to_owned(), "180".to_owned()]
}
fn default_cls_thresh() -> f32 {
    0.9
}

// ---------------------------------------------------------------------------
// PpocrEngine: synchronous detection -> recognition pipeline
// ---------------------------------------------------------------------------

/// Owns the Burn models and config and performs the synchronous OCR pipeline.
struct PpocrEngine {
    model_id: String,
    det_config: PpocrDetConfig,
    rec_config: PpocrRecConfig,
    cls_config: Option<PpocrClsConfig>,
    det_model: PpocrDetectionModel,
    rec_model: PpocrRecognitionModel,
    cls_model: Option<PpocrClassificationModel>,
    vocab: Vec<String>,
}

impl PpocrEngine {
    fn run_from_image(&self, image: &DynamicImage) -> OCRV1 {
        let rgb = image.to_rgb8();
        let (src_h, src_w) = (rgb.height(), rgb.width());

        let (det_input, det_input_h, det_input_w, ratio_h, ratio_w) = det_preprocess(
            &rgb,
            self.det_config.limit_side_len,
            self.det_config.scale,
            &self.det_config.mean,
            &self.det_config.std,
        );

        self.run_pipeline(
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
    }

    fn run_from_det_tensor(
        &self,
        det_input: Vec<f32>,
        det_input_h: usize,
        det_input_w: usize,
        src_w: u32,
        src_h: u32,
    ) -> OCRV1 {
        let ratio_h = det_input_h as f32 / src_h as f32;
        let ratio_w = det_input_w as f32 / src_w as f32;
        let crop_canvas = denormalize_nchw_to_rgb(
            &det_input,
            det_input_h,
            det_input_w,
            self.det_config.scale,
            &self.det_config.mean,
            &self.det_config.std,
        );

        self.run_pipeline(
            &crop_canvas,
            &det_input,
            det_input_h,
            det_input_w,
            ratio_h,
            ratio_w,
            src_w,
            src_h,
            |box_coords| source_box_to_det_canvas(box_coords, ratio_w, ratio_h),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_pipeline(
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
    ) -> OCRV1 {
        let det = self.det_model.forward(det_input, det_input_h, det_input_w);
        if det.shape.len() != 4 {
            return OCRV1::new(vec![], &self.model_id);
        }
        let prob_h = det.shape[2] as u32;
        let prob_w = det.shape[3] as u32;

        let boxes = detect_boxes(
            &det.values,
            prob_h,
            prob_w,
            self.det_config.thresh,
            self.det_config.box_thresh,
            self.det_config.unclip_ratio,
            ratio_h,
            ratio_w,
            src_h,
            src_w,
        );

        if ppocr_debug_enabled() {
            eprintln!(
                "ppocr_debug det_output_shape={:?} det_output_len={} boxes={}",
                det.shape,
                det.values.len(),
                boxes.len()
            );
        }

        if boxes.is_empty() {
            return OCRV1::new(vec![], &self.model_id);
        }

        let sorted_boxes = sort_boxes(&boxes);
        let mut items = Vec::with_capacity(sorted_boxes.len());

        for (box_idx, box_coords) in sorted_boxes.iter().enumerate() {
            let crop_box = map_box_for_crop(box_coords);
            let crop = match perspective_crop(crop_image, &crop_box) {
                Ok(c) => c,
                Err(err) => {
                    if ppocr_debug_enabled() {
                        eprintln!("ppocr_debug box_idx={box_idx} crop_error={err}");
                    }
                    continue;
                }
            };

            let crop = self.maybe_orient(crop, box_idx);

            let [c, h, w] = self.rec_config.image_shape;
            let rec_input = rec_preprocess(
                &crop,
                c,
                h as u32,
                w as u32,
                self.rec_config.scale,
                &self.rec_config.mean,
                &self.rec_config.std,
            );

            let rec = self.rec_model.forward(&rec_input, c, h, w);
            if rec.shape.len() != 3 {
                continue;
            }
            let seq_len = rec.shape[1];
            let num_classes = rec.shape[2];
            let (indices, conf) =
                ctc_greedy_decode(&rec.values, seq_len, num_classes, self.rec_config.blank_id);
            let text = indices_to_text(&indices, &self.vocab, self.rec_config.blank_id);

            if ppocr_debug_enabled() {
                eprintln!(
                    "ppocr_debug box_idx={box_idx} crop={}x{} decoded_len={} conf={conf:.4} text={text:?}",
                    crop.width(),
                    crop.height(),
                    text.chars().count(),
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

        OCRV1::new(items, &self.model_id)
    }

    /// Corrects upside-down text crops using the orientation classifier (server
    /// pack only). Returns the crop rotated 180° when the classifier predicts the
    /// `180` label above its confidence threshold; otherwise the crop unchanged.
    fn maybe_orient(&self, crop: RgbImage, box_idx: usize) -> RgbImage {
        let (Some(cfg), Some(model)) = (&self.cls_config, &self.cls_model) else {
            return crop;
        };
        let [c, h, w] = cfg.image_shape;
        let input = rec_preprocess(&crop, c, h as u32, w as u32, cfg.scale, &cfg.mean, &cfg.std);
        let out = model.forward(&input, c, h, w);
        let (best, score) =
            out.values
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |acc, (i, &v)| {
                        if v > acc.1 { (i, v) } else { acc }
                    },
                );
        let upside_down = cfg.labels.get(best).is_some_and(|l| l == "180");
        if ppocr_debug_enabled() {
            eprintln!(
                "ppocr_debug box_idx={box_idx} cls_label={:?} score={score:.4} rotate={}",
                cfg.labels.get(best),
                upside_down && score >= cfg.thresh,
            );
        }
        if upside_down && score >= cfg.thresh {
            image::imageops::rotate180(&crop)
        } else {
            crop
        }
    }
}

// ---------------------------------------------------------------------------
// PpocrTask
// ---------------------------------------------------------------------------

/// End-to-end PP-OCR task: detection → post-process → recognition → decode.
pub struct PpocrTask {
    spec: TaskSpec,
    model_id: String,
    engine: Arc<PpocrEngine>,
}

impl PpocrTask {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        model_id: String,
        det_config: PpocrDetConfig,
        rec_config: PpocrRecConfig,
        cls_config: Option<PpocrClsConfig>,
        det_model: PpocrDetectionModel,
        rec_model: PpocrRecognitionModel,
        cls_model: Option<PpocrClassificationModel>,
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
            model_id: model_id.clone(),
            engine: Arc::new(PpocrEngine {
                model_id,
                det_config,
                rec_config,
                cls_config,
                det_model,
                rec_model,
                cls_model,
                vocab,
            }),
        })
    }
}

#[async_trait]
impl TaskHandler for PpocrTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let engine = Arc::clone(&self.engine);

        let results = if is_tensor_input_request(&request) {
            let descriptor = validate_dynamic_det_tensor_request(
                &request,
                crate::service::DynamicDetTensorValidationOptions {
                    dtype: "fp32",
                    preprocess_id: PREPROCESS_PPOCR_DET,
                },
            )?;
            let source = parse_source_dimensions(&request)?;
            let det_input = bytes_to_f32_le(&request.payload)?;
            let det_h = descriptor.shape[2];
            let det_w = descriptor.shape[3];
            run_blocking(move || {
                engine.run_from_det_tensor(det_input, det_h, det_w, source.width, source.height)
            })
            .await?
        } else {
            let image = image::load_from_memory(&request.payload).map_err(|e| {
                ServiceError::InvalidArgument(format!("failed to decode image: {e}"))
            })?;
            run_blocking(move || engine.run_from_image(&image)).await?
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

/// Runs a blocking inference closure on the dedicated inference worker.
async fn run_blocking<F, T>(f: F) -> ServiceResult<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    inference_worker::run(f)
        .await
        .map_err(|e| ServiceError::Internal(format!("inference worker failed: {e}")))
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
    resize_h = (resize_h.div_ceil(32) * 32).max(32);
    resize_w = (resize_w.div_ceil(32) * 32).max(32);

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
    fn test_indices_to_text() {
        let vocab = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        assert_eq!(indices_to_text(&[1, 2, 3], &vocab, 0), "abc");
    }

    #[test]
    fn test_indices_to_text_with_nonzero_blank_id() {
        let vocab = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        assert_eq!(indices_to_text(&[0, 1, 3], &vocab, 2), "abc");
    }

    #[test]
    fn test_source_box_to_det_canvas_scales_coordinates() {
        let mapped =
            source_box_to_det_canvas(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 2.0, 0.5);
        assert_eq!(mapped, [20.0, 10.0, 60.0, 20.0, 100.0, 30.0, 140.0, 40.0]);
    }

    #[test]
    fn test_rec_preprocess_pads_to_configured_width() {
        let image = RgbImage::from_pixel(8, 16, Rgb([255, 255, 255]));
        let tensor = rec_preprocess(&image, 3, 48, 320, 1.0 / 255.0, &[0.5; 3], &[0.5; 3]);
        assert_eq!(tensor.len(), 3 * 48 * 320);
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor[48 * 320 - 1], -1.0);
    }
}
