use std::sync::Arc;

use async_trait::async_trait;
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use lumen_schema::{BboxItem, Face, FaceV1};
use serde::Deserialize;

use super::metadata::{InsightFaceDetectionSpec, InsightFacePackSpec, InsightFaceRecognitionSpec};
use super::model::{InsightFaceDetectionModel, InsightFaceRecognitionModel, TensorOutput};
use crate::{
    inference_worker,
    service::{
        DEFAULT_TENSOR_MIME, FixedShapeTensorValidationOptions, IMAGE_TENSOR_LAYOUT,
        LetterboxTransform, META_MODEL_ID, PREPROCESS_INSIGHTFACE_DET, ServiceError, ServiceResult,
        TaskHandler, TaskRequest, TaskResult, TaskSpec, bytes_to_f32_le, is_tensor_input_request,
        parse_letterbox_transform, parse_source_dimensions, validate_fixed_shape_tensor_request,
    },
};

const SUPPORTED_IMAGE_MIMES: [&str; 4] = ["image/jpeg", "image/png", "image/webp", "image/avif"];
const JSON_MIME: &str = "application/json;schema=face_v1";
const NUM_ANCHORS: usize = 2;
const ARCFACE_TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Detection/recognition component config from `model_info.json`.
///
/// With Burn, model IO node names are fixed by the compiled graph, so these
/// fields are retained only for metadata compatibility (`component` selects the
/// `.bpk` filename).
#[derive(Debug, Clone, Deserialize)]
pub struct InsightFaceComponentConfig {
    pub component: String,
    #[serde(default)]
    pub input_name: String,
    #[serde(default)]
    pub output_name: Option<String>,
    #[serde(default)]
    pub output_names: Vec<String>,
}

/// Owns the Burn models + spec and performs the synchronous face pipeline.
struct InsightFaceEngine {
    model_id: String,
    pack: InsightFacePackSpec,
    det_model: InsightFaceDetectionModel,
    rec_model: InsightFaceRecognitionModel,
}

impl InsightFaceEngine {
    fn det_input_size(&self) -> (usize, usize) {
        (
            self.pack.detection.input_size[1],
            self.pack.detection.input_size[0],
        )
    }

    fn run_from_image(&self, image: &DynamicImage) -> ServiceResult<FaceV1> {
        let rgb = image.to_rgb8();
        let (det_input, transform) = det_preprocess(&rgb, &self.pack.detection);
        self.run_pipeline(
            &rgb,
            &det_input,
            transform,
            rgb.width(),
            rgb.height(),
            |landmarks| *landmarks,
            |bbox| *bbox,
        )
    }

    fn run_from_det_tensor(
        &self,
        det_input: Vec<f32>,
        transform: LetterboxTransform,
        source_w: u32,
        source_h: u32,
    ) -> ServiceResult<FaceV1> {
        let canvas = denormalize_det_nchw_to_rgb(
            &det_input,
            self.pack.detection.input_size[1],
            self.pack.detection.input_size[0],
            &self.pack.detection.mean,
            &self.pack.detection.std,
        );
        self.run_pipeline(
            &canvas,
            &det_input,
            transform,
            source_w,
            source_h,
            |landmarks| source_landmarks_to_canvas(landmarks, transform),
            |bbox| source_bbox_to_canvas(bbox, transform),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_pipeline(
        &self,
        crop_image: &RgbImage,
        det_input: &[f32],
        transform: LetterboxTransform,
        source_w: u32,
        source_h: u32,
        map_landmarks_for_crop: impl Fn(&[[f32; 2]; 5]) -> [[f32; 2]; 5],
        map_bbox_for_crop: impl Fn(&[f32; 4]) -> [f32; 4],
    ) -> ServiceResult<FaceV1> {
        let (det_h, det_w) = self.det_input_size();
        let det_outputs = self.det_model.forward(det_input, det_h, det_w);

        let candidates = scrfd_postprocess(
            &det_outputs,
            &self.pack.detection,
            &transform,
            source_w,
            source_h,
        )?;
        let detections = nms(candidates, self.pack.detection.nms_threshold);

        let mut faces = Vec::with_capacity(detections.len());
        for det in detections {
            let crop_landmarks = map_landmarks_for_crop(&det.landmarks);
            let aligned = if self.pack.recognition.align_landmarks {
                align_face(
                    crop_image,
                    &crop_landmarks,
                    self.pack.recognition.input_size,
                )
            } else {
                crop_face(
                    crop_image,
                    map_bbox_for_crop(&det.bbox),
                    self.pack.recognition.input_size,
                )
            };
            let rec_input = rec_preprocess(&aligned, &self.pack.recognition);
            let embedding = l2_normalize(self.rec_model.forward(
                &rec_input,
                self.pack.recognition.input_size[1],
                self.pack.recognition.input_size[0],
            ));

            faces.push(Face {
                bbox: det.bbox.iter().copied().map(BboxItem::from).collect(),
                confidence: det.score.clamp(0.0, 1.0),
                landmarks: Some(
                    det.landmarks
                        .iter()
                        .flat_map(|point| [point[0], point[1]])
                        .collect(),
                ),
                embedding: Some(embedding),
            });
        }

        Ok(FaceV1::new(faces, &self.model_id))
    }
}

pub struct InsightFaceTask {
    spec: TaskSpec,
    model_id: String,
    engine: Arc<InsightFaceEngine>,
}

impl InsightFaceTask {
    pub fn new(
        name: impl Into<String>,
        model_id: String,
        pack: InsightFacePackSpec,
        det_model: InsightFaceDetectionModel,
        rec_model: InsightFaceRecognitionModel,
    ) -> ServiceResult<Self> {
        let name = name.into();
        let spec = TaskSpec::new(
            &name,
            "InsightFace face detection, alignment, and embedding recognition",
        )
        .with_input_mimes(image_input_mimes_with_tensor())
        .with_output_mime(JSON_MIME)
        .with_metadata("output_schema", "face_v1")
        .with_metadata(META_MODEL_ID, &model_id)
        .with_limit("det_input_width", pack.detection.input_size[0].to_string())
        .with_limit("det_input_height", pack.detection.input_size[1].to_string())
        .with_limit("embedding_dim", pack.recognition.embedding_dim.to_string());

        Ok(Self {
            spec,
            model_id: model_id.clone(),
            engine: Arc::new(InsightFaceEngine {
                model_id,
                pack,
                det_model,
                rec_model,
            }),
        })
    }

    fn det_input_shape(&self) -> Vec<usize> {
        vec![
            1,
            3,
            self.engine.pack.detection.input_size[1],
            self.engine.pack.detection.input_size[0],
        ]
    }
}

#[async_trait]
impl TaskHandler for InsightFaceTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let engine = Arc::clone(&self.engine);

        let result = if is_tensor_input_request(&request) {
            let expected_shape = self.det_input_shape();
            validate_fixed_shape_tensor_request(
                &request,
                FixedShapeTensorValidationOptions {
                    dtype: "fp32",
                    layout: IMAGE_TENSOR_LAYOUT,
                    preprocess_id: PREPROCESS_INSIGHTFACE_DET,
                    expected_shape: &expected_shape,
                },
            )?;
            let source = parse_source_dimensions(&request)?;
            let transform = parse_letterbox_transform(&request)?;
            let det_input = bytes_to_f32_le(&request.payload)?;
            run_blocking(move || {
                engine.run_from_det_tensor(det_input, transform, source.width, source.height)
            })
            .await??
        } else {
            let image = decode_request_image(&request)?;
            run_blocking(move || engine.run_from_image(&image)).await??
        };

        let json_bytes = result.to_json_bytes().map_err(|e| {
            ServiceError::Internal(format!("failed to serialize face_v1 result: {e}"))
        })?;

        Ok(TaskResult::new(json_bytes, JSON_MIME)
            .with_result_schema("face_v1")
            .with_meta(META_MODEL_ID, &self.model_id))
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

fn image_input_mimes_with_tensor() -> Vec<String> {
    SUPPORTED_IMAGE_MIMES
        .iter()
        .copied()
        .chain(std::iter::once(DEFAULT_TENSOR_MIME))
        .map(str::to_owned)
        .collect()
}

fn decode_request_image(request: &TaskRequest) -> ServiceResult<DynamicImage> {
    if SUPPORTED_IMAGE_MIMES.contains(&request.payload_mime.as_str()) {
        return image::load_from_memory(&request.payload)
            .map_err(|e| ServiceError::InvalidArgument(format!("failed to decode image: {e}")));
    }

    Err(ServiceError::InvalidArgument(format!(
        "unsupported face_recognition input MIME `{}`; supported image MIME types: {}",
        request.payload_mime,
        SUPPORTED_IMAGE_MIMES.join(", ")
    )))
}

fn denormalize_det_nchw_to_rgb(
    values: &[f32],
    height: usize,
    width: usize,
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
                rgb[channel] = pixel.round().clamp(0.0, 255.0) as u8;
            }
            image.put_pixel(x as u32, y as u32, Rgb(rgb));
        }
    }
    image
}

fn source_landmarks_to_canvas(
    landmarks: &[[f32; 2]; 5],
    transform: LetterboxTransform,
) -> [[f32; 2]; 5] {
    let mut mapped = [[0.0; 2]; 5];
    for (dst, src) in mapped.iter_mut().zip(landmarks) {
        dst[0] = src[0] * transform.scale + transform.pad_x;
        dst[1] = src[1] * transform.scale + transform.pad_y;
    }
    mapped
}

fn source_bbox_to_canvas(bbox: &[f32; 4], transform: LetterboxTransform) -> [f32; 4] {
    [
        bbox[0] * transform.scale + transform.pad_x,
        bbox[1] * transform.scale + transform.pad_y,
        bbox[2] * transform.scale + transform.pad_x,
        bbox[3] * transform.scale + transform.pad_y,
    ]
}

fn det_preprocess(
    image: &RgbImage,
    spec: &InsightFaceDetectionSpec,
) -> (Vec<f32>, LetterboxTransform) {
    let target_w = spec.input_size[0] as u32;
    let target_h = spec.input_size[1] as u32;
    let (scale, pad_x, pad_y, resized) = if spec.letterbox {
        let scale =
            (target_w as f32 / image.width() as f32).min(target_h as f32 / image.height() as f32);
        let resize_w = (image.width() as f32 * scale).round().max(1.0) as u32;
        let resize_h = (image.height() as f32 * scale).round().max(1.0) as u32;
        let resized = image::imageops::resize(image, resize_w, resize_h, FilterType::CatmullRom);
        (
            scale,
            ((target_w - resize_w) / 2) as f32,
            ((target_h - resize_h) / 2) as f32,
            resized,
        )
    } else {
        let resized = image::imageops::resize(image, target_w, target_h, FilterType::CatmullRom);
        (target_w as f32 / image.width() as f32, 0.0, 0.0, resized)
    };

    let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([0, 0, 0]));
    image::imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    let mut tensor = vec![0.0f32; 3 * target_h as usize * target_w as usize];
    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let pixel = canvas.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[c * target_h as usize * target_w as usize + y * target_w as usize + x] =
                    (pixel[c] as f32 - spec.mean[c]) / spec.std[c];
            }
        }
    }

    (
        tensor,
        LetterboxTransform {
            scale,
            pad_x,
            pad_y,
        },
    )
}

#[derive(Debug, Clone)]
struct FaceDetection {
    bbox: [f32; 4],
    landmarks: [[f32; 2]; 5],
    score: f32,
}

fn scrfd_postprocess(
    outputs: &[TensorOutput],
    spec: &InsightFaceDetectionSpec,
    transform: &LetterboxTransform,
    src_w: u32,
    src_h: u32,
) -> ServiceResult<Vec<FaceDetection>> {
    if spec.detector_type != "scrfd" {
        return Err(ServiceError::InvalidArgument(format!(
            "unsupported InsightFace detector type `{}`",
            spec.detector_type
        )));
    }

    let mut faces = Vec::new();
    for mapping in &spec.outputs {
        let score = outputs.get(mapping.score).ok_or_else(|| {
            ServiceError::InvalidArgument("SCRFD score output index is out of range".to_owned())
        })?;
        let bbox = outputs.get(mapping.bbox).ok_or_else(|| {
            ServiceError::InvalidArgument("SCRFD bbox output index is out of range".to_owned())
        })?;
        let kps = outputs.get(mapping.kps).ok_or_else(|| {
            ServiceError::InvalidArgument("SCRFD keypoint output index is out of range".to_owned())
        })?;
        let count = tensor_rows(score, 1)?;
        if tensor_rows(bbox, 4)? != count || tensor_rows(kps, 10)? != count {
            return Err(ServiceError::Internal(format!(
                "SCRFD output count mismatch for stride {}",
                mapping.stride
            )));
        }

        let feat_w = spec.input_size[0] / mapping.stride;
        let feat_h = spec.input_size[1] / mapping.stride;
        for index in 0..count {
            let confidence = score.values[index];
            if confidence < spec.score_threshold {
                continue;
            }
            let anchor_index = index / NUM_ANCHORS;
            let y = anchor_index / feat_w;
            let x = anchor_index % feat_w;
            if y >= feat_h {
                continue;
            }
            let center_x = (x as f32 + 0.5) * mapping.stride as f32;
            let center_y = (y as f32 + 0.5) * mapping.stride as f32;

            let b = &bbox.values[index * 4..index * 4 + 4];
            let mut decoded_bbox = [
                center_x - b[0] * mapping.stride as f32,
                center_y - b[1] * mapping.stride as f32,
                center_x + b[2] * mapping.stride as f32,
                center_y + b[3] * mapping.stride as f32,
            ];
            let mut decoded_kps = [[0.0f32; 2]; 5];
            for point in 0..5 {
                decoded_kps[point] = [
                    center_x + kps.values[index * 10 + point * 2] * mapping.stride as f32,
                    center_y + kps.values[index * 10 + point * 2 + 1] * mapping.stride as f32,
                ];
            }

            if spec.normalized_boxes {
                let w = spec.input_size[0] as f32;
                let h = spec.input_size[1] as f32;
                decoded_bbox = [
                    decoded_bbox[0] * w,
                    decoded_bbox[1] * h,
                    decoded_bbox[2] * w,
                    decoded_bbox[3] * h,
                ];
                for point in &mut decoded_kps {
                    point[0] *= w;
                    point[1] *= h;
                }
            }

            let bbox = unletterbox_bbox(decoded_bbox, transform, src_w, src_h);
            let width = bbox[2] - bbox[0];
            let height = bbox[3] - bbox[1];
            let face_size = width.max(height);
            if face_size < spec.min_face || face_size > spec.max_face {
                continue;
            }
            faces.push(FaceDetection {
                bbox,
                landmarks: unletterbox_landmarks(decoded_kps, transform, src_w, src_h),
                score: confidence,
            });
        }
    }
    Ok(faces)
}

fn tensor_rows(output: &TensorOutput, width: usize) -> ServiceResult<usize> {
    if output.values.len() % width != 0 {
        return Err(ServiceError::Internal(format!(
            "tensor shape {:?} has {} values, not divisible by row width {width}",
            output.shape,
            output.values.len()
        )));
    }
    Ok(output.values.len() / width)
}

fn unletterbox_bbox(
    bbox: [f32; 4],
    transform: &LetterboxTransform,
    src_w: u32,
    src_h: u32,
) -> [f32; 4] {
    [
        ((bbox[0] - transform.pad_x) / transform.scale).clamp(0.0, src_w as f32),
        ((bbox[1] - transform.pad_y) / transform.scale).clamp(0.0, src_h as f32),
        ((bbox[2] - transform.pad_x) / transform.scale).clamp(0.0, src_w as f32),
        ((bbox[3] - transform.pad_y) / transform.scale).clamp(0.0, src_h as f32),
    ]
}

fn unletterbox_landmarks(
    landmarks: [[f32; 2]; 5],
    transform: &LetterboxTransform,
    src_w: u32,
    src_h: u32,
) -> [[f32; 2]; 5] {
    let mut out = [[0.0; 2]; 5];
    for (dst, src) in out.iter_mut().zip(landmarks) {
        dst[0] = ((src[0] - transform.pad_x) / transform.scale).clamp(0.0, src_w as f32);
        dst[1] = ((src[1] - transform.pad_y) / transform.scale).clamp(0.0, src_h as f32);
    }
    out
}

fn nms(mut faces: Vec<FaceDetection>, threshold: f32) -> Vec<FaceDetection> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept = Vec::new();
    while let Some(face) = faces.first().cloned() {
        faces.remove(0);
        faces.retain(|candidate| iou(face.bbox, candidate.bbox) <= threshold);
        kept.push(face);
    }
    kept
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    inter / (area_a + area_b - inter).max(1e-6)
}

fn rec_preprocess(image: &RgbImage, spec: &InsightFaceRecognitionSpec) -> Vec<f32> {
    let w = spec.input_size[0];
    let h = spec.input_size[1];
    let mut tensor = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = image.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let src_c = if spec.color_order == "rgb" { c } else { 2 - c };
                let normalized = (pixel[src_c] as f32 - spec.mean[c]) / spec.std[c];
                if spec.channels_last {
                    tensor[(y * w + x) * 3 + c] = normalized;
                } else {
                    tensor[c * h * w + y * w + x] = normalized;
                }
            }
        }
    }
    tensor
}

fn crop_face(image: &RgbImage, bbox: [f32; 4], size: [usize; 2]) -> RgbImage {
    let x1 = bbox[0].floor().max(0.0) as u32;
    let y1 = bbox[1].floor().max(0.0) as u32;
    let x2 = bbox[2].ceil().min(image.width() as f32) as u32;
    let y2 = bbox[3].ceil().min(image.height() as f32) as u32;
    let crop_w = x2.saturating_sub(x1).max(1);
    let crop_h = y2.saturating_sub(y1).max(1);
    let crop = image::imageops::crop_imm(image, x1, y1, crop_w, crop_h).to_image();
    image::imageops::resize(
        &crop,
        size[0] as u32,
        size[1] as u32,
        FilterType::CatmullRom,
    )
}

fn align_face(image: &RgbImage, landmarks: &[[f32; 2]; 5], size: [usize; 2]) -> RgbImage {
    let affine =
        estimate_affine(landmarks, &ARCFACE_TEMPLATE).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    warp_affine(image, affine, size[0] as u32, size[1] as u32)
}

fn estimate_affine(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> Option<[[f32; 3]; 2]> {
    let mut ata = [[0.0f32; 6]; 6];
    let mut atb = [0.0f32; 6];
    for (s, d) in src.iter().zip(dst) {
        let rows = [
            ([s[0], s[1], 1.0, 0.0, 0.0, 0.0], d[0]),
            ([0.0, 0.0, 0.0, s[0], s[1], 1.0], d[1]),
        ];
        for (row, target) in rows {
            for i in 0..6 {
                atb[i] += row[i] * target;
                for j in 0..6 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let coeffs = solve_6x6(ata, atb)?;
    Some([
        [coeffs[0], coeffs[1], coeffs[2]],
        [coeffs[3], coeffs[4], coeffs[5]],
    ])
}

fn solve_6x6(mut a: [[f32; 6]; 6], mut b: [f32; 6]) -> Option<[f32; 6]> {
    for col in 0..6 {
        let mut pivot = col;
        for row in col + 1..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-6 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let div = a[col][col];
        for j in col..6 {
            a[col][j] /= div;
        }
        b[col] /= div;

        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..6 {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

fn warp_affine(image: &RgbImage, affine: [[f32; 3]; 2], out_w: u32, out_h: u32) -> RgbImage {
    let det = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0];
    if det.abs() < 1e-6 {
        return image::imageops::resize(image, out_w, out_h, FilterType::CatmullRom);
    }
    let inv_det = 1.0 / det;
    let inv = [
        [affine[1][1] * inv_det, -affine[0][1] * inv_det],
        [-affine[1][0] * inv_det, affine[0][0] * inv_det],
    ];
    let mut out = RgbImage::from_pixel(out_w, out_h, Rgb([0, 0, 0]));
    for y in 0..out_h {
        for x in 0..out_w {
            let dx = x as f32 - affine[0][2];
            let dy = y as f32 - affine[1][2];
            let sx = inv[0][0] * dx + inv[0][1] * dy;
            let sy = inv[1][0] * dx + inv[1][1] * dy;
            out.put_pixel(x, y, bilinear_sample(image, sx, sy));
        }
    }
    out
}

fn bilinear_sample(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    if x < 0.0 || y < 0.0 || x > (image.width() - 1) as f32 || y > (image.height() - 1) as f32 {
        return Rgb([0, 0, 0]);
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - wx) + p10[c] as f32 * wx;
        let bottom = p01[c] as f32 * (1.0 - wx) + p11[c] as f32 * wx;
        out[c] = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

fn l2_normalize(mut values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for value in &mut values {
            *value /= norm;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::{LetterboxTransform, iou, source_bbox_to_canvas, source_landmarks_to_canvas};

    #[test]
    fn maps_source_geometry_back_to_canvas_space() {
        let transform = LetterboxTransform {
            scale: 0.5,
            pad_x: 10.0,
            pad_y: 20.0,
        };
        let landmarks = source_landmarks_to_canvas(&[[100.0, 200.0]; 5], transform);
        assert_eq!(landmarks[0], [60.0, 120.0]);

        let bbox = source_bbox_to_canvas(&[100.0, 200.0, 140.0, 260.0], transform);
        assert_eq!(bbox, [60.0, 120.0, 80.0, 150.0]);
    }

    #[test]
    fn computes_iou_for_overlapping_boxes() {
        let value = iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]);
        assert!((value - 25.0 / 175.0).abs() < 1e-6);
    }
}
