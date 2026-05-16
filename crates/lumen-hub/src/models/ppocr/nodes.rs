use std::collections::HashMap;

use async_trait::async_trait;
use image::{GrayImage, Luma};
use imageproc::{
    contours as imageproc_contours, drawing::draw_polygon_mut, geometry as imageproc_geometry,
    point::Point,
};
use lumnn::core::{
    context::MLContext,
    node::MLNode,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
};

// ---------------------------------------------------------------------------
// DBPostProcessNode – converts detection probability map → bounding boxes
// ---------------------------------------------------------------------------

/// DBNet post-processing node.
///
/// Takes the detection model's probability map (sigmoid output) and produces
/// axis-aligned or rotated bounding boxes by:
///
/// 1. Thresholding the probability map into a binary image.
/// 2. Finding contours via border-following.
/// 3. Fitting a minimum-area rotated rectangle to each contour.
/// 4. Scoring each box against the probability map.
/// 5. Expanding (unclipping) each box.
/// 6. Clipping and scaling boxes back to original image coordinates.
///
/// # Inputs
///
/// | Name       | Dtype     | Shape        | Description                              |
/// |------------|-----------|--------------|------------------------------------------|
/// | `prob_map` | Float32   | `[1, 1, H, W]` | Detection model sigmoid output        |
/// | `ratio_h`  | Float32   | `[1]`        | Height scale factor (resize_h / orig_h) |
/// | `ratio_w`  | Float32   | `[1]`        | Width scale factor (resize_w / orig_w)  |
/// | `src_h`    | Int64     | `[1]`        | Original image height in pixels         |
/// | `src_w`    | Int64     | `[1]`        | Original image width in pixels          |
///
/// # Outputs
///
/// | Name     | Dtype   | Shape    | Description                                      |
/// |----------|---------|----------|--------------------------------------------------|
/// | `boxes`  | Float32 | `[N, 8]` | Bounding boxes: 4 corners × (x, y) in orig coords |
pub struct DBPostProcessNode {
    name: String,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
}

impl DBPostProcessNode {
    pub fn new(name: impl Into<String>, thresh: f32, box_thresh: f32, unclip_ratio: f32) -> Self {
        let input_descriptors = HashMap::from([
            (
                "prob_map".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 1, 1, 1])
                    .with_dynamic_axis(2)
                    .with_dynamic_axis(3),
            ),
            (
                "ratio_h".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1]),
            ),
            (
                "ratio_w".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1]),
            ),
            (
                "src_h".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1]),
            ),
            (
                "src_w".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1]),
            ),
        ]);

        // Dynamic output: N varies per image
        let output_descriptors = HashMap::from([(
            "boxes".to_owned(),
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 8]).with_dynamic_batch(),
        )]);

        Self {
            name: name.into(),
            input_descriptors,
            output_descriptors,
            thresh,
            box_thresh,
            unclip_ratio,
        }
    }
}

#[async_trait]
impl MLNode for DBPostProcessNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
        &self.input_descriptors
    }

    fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
        &self.output_descriptors
    }

    async fn execute(
        &self,
        mut inputs: HashMap<String, MLPacket>,
        _context: &MLContext,
    ) -> Result<HashMap<String, MLPacket>, String> {
        // --- Extract inputs ---
        let prob_packet = inputs
            .remove("prob_map")
            .ok_or("missing required input `prob_map`")?;
        let ratio_h_packet = inputs
            .remove("ratio_h")
            .ok_or("missing required input `ratio_h`")?;
        let ratio_w_packet = inputs
            .remove("ratio_w")
            .ok_or("missing required input `ratio_w`")?;
        let src_h_packet = inputs
            .remove("src_h")
            .ok_or("missing required input `src_h`")?;
        let src_w_packet = inputs
            .remove("src_w")
            .ok_or("missing required input `src_w`")?;

        // Validate no unexpected inputs
        if !inputs.is_empty() {
            let mut names = inputs.into_keys().collect::<Vec<_>>();
            names.sort();
            return Err(format!("unexpected inputs: {}", names.join(", ")));
        }

        // --- Extract scalar values ---
        let ratio_h = extract_scalar_f32(ratio_h_packet, "ratio_h").await?;
        let ratio_w = extract_scalar_f32(ratio_w_packet, "ratio_w").await?;
        let src_h = extract_scalar_i64(src_h_packet, "src_h").await? as u32;
        let src_w = extract_scalar_i64(src_w_packet, "src_w").await? as u32;

        // --- Extract probability map ---
        let (ctx, prob_desc, prob_payload) = prob_packet.into_parts()?;
        let prob_values = match prob_payload.to_host_tensor()? {
            HostTensor::Float32(v) => v,
            other => {
                return Err(format!(
                    "DBPostProcessNode expected Float32 prob_map, got {other:?}"
                ));
            }
        };

        let h = prob_desc.shape[2] as u32;
        let w = prob_desc.shape[3] as u32;

        // --- Post-process ---
        let boxes = db_postprocess(
            &prob_values,
            h,
            w,
            self.thresh,
            self.box_thresh,
            self.unclip_ratio,
        )
        .map_err(|e| format!("DB post-process failed: {e}"))?;

        // --- Scale boxes to original image coordinates ---
        let scaled_boxes: Vec<f32> = boxes
            .chunks(8)
            .flat_map(|box_coords| {
                box_coords.chunks(2).flat_map(|point| {
                    let x = (point[0] / ratio_w).clamp(0.0, src_w as f32);
                    let y = (point[1] / ratio_h).clamp(0.0, src_h as f32);
                    [x, y]
                })
            })
            .collect();

        let num_boxes = scaled_boxes.len() / 8;
        let out_shape = if num_boxes == 0 {
            vec![0, 8]
        } else {
            vec![num_boxes, 8]
        };

        let output_desc = MLPacketDescriptor::new(MLPacketDataType::Float32, out_shape);
        let output_packet =
            MLPacket::from_host_tensor(ctx, output_desc, HostTensor::Float32(scaled_boxes))?;

        Ok(HashMap::from([("boxes".to_owned(), output_packet)]))
    }
}

async fn extract_scalar_f32(packet: MLPacket, name: &str) -> Result<f32, String> {
    let values = match packet.to_host_tensor().await {
        Ok(HostTensor::Float32(v)) => v,
        Ok(other) => return Err(format!("expected Float32 for `{name}`, got {other:?}")),
        Err(e) => return Err(format!("failed to read `{name}`: {e}")),
    };
    if values.len() != 1 {
        return Err(format!(
            "expected scalar for `{name}`, got {} elements",
            values.len()
        ));
    }
    Ok(values[0])
}

async fn extract_scalar_i64(packet: MLPacket, name: &str) -> Result<i64, String> {
    let values = match packet.to_host_tensor().await {
        Ok(HostTensor::Int64(v)) => v,
        Ok(other) => return Err(format!("expected Int64 for `{name}`, got {other:?}")),
        Err(e) => return Err(format!("failed to read `{name}`: {e}")),
    };
    if values.len() != 1 {
        return Err(format!(
            "expected scalar for `{name}`, got {} elements",
            values.len()
        ));
    }
    Ok(values[0])
}

// ---------------------------------------------------------------------------
// DB post-processing core (pure functions, no ONNX dependency)
// ---------------------------------------------------------------------------

fn db_postprocess(
    prob: &[f32],
    h: u32,
    w: u32,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
) -> Result<Vec<f32>, String> {
    // 1. Threshold → binary segmentation
    let binary: Vec<u8> = prob
        .iter()
        .map(|&p| if p > thresh { 255u8 } else { 0u8 })
        .collect();

    // 2. Find contours
    let contours = find_contours(&binary, h as usize, w as usize);

    let mut boxes: Vec<f32> = Vec::new();

    for contour in &contours {
        if contour.len() < 4 {
            continue;
        }

        // 3. Get minimum area rotated rectangle
        let (rect_points, short_side) = min_area_rect(contour);
        if short_side < 3.0 {
            continue;
        }

        // 4. Score box against probability map
        let score = box_score(prob, h as usize, w as usize, &rect_points);
        if score < box_thresh {
            continue;
        }

        // 5. Unclip (expand) the box
        let expanded = unclip_polygon(&rect_points, unclip_ratio);

        // 6. Re-fit min area rect on expanded polygon
        let expanded_contour: Vec<(f32, f32)> = expanded.iter().map(|p| (p[0], p[1])).collect();
        let (final_points, final_short_side) = min_area_rect(&expanded_contour);
        if final_short_side < 5.0 {
            continue;
        }

        // 7. Flatten 4 points × (x, y) → [x0, y0, x1, y1, x2, y2, x3, y3]
        for pt in &final_points {
            boxes.push(pt.0);
            boxes.push(pt.1);
        }
    }

    Ok(boxes)
}

fn find_contours(binary: &[u8], h: usize, w: usize) -> Vec<Vec<(f32, f32)>> {
    let Some(image) = GrayImage::from_vec(w as u32, h as u32, binary.to_vec()) else {
        return Vec::new();
    };

    imageproc_contours::find_contours::<i32>(&image)
        .into_iter()
        .filter(|contour| contour.border_type == imageproc_contours::BorderType::Outer)
        .map(|contour| {
            contour
                .points
                .into_iter()
                .map(|p| (p.x as f32, p.y as f32))
                .collect()
        })
        .collect()
}

fn min_area_rect(contour: &[(f32, f32)]) -> (Vec<(f32, f32)>, f32) {
    if contour.len() < 3 {
        return (contour.to_vec(), 0.0);
    }

    let points = to_imageproc_points_f32(contour);
    let rect = imageproc_geometry::min_area_rect(&points);
    let ordered: Vec<(f32, f32)> = rect.iter().map(|p| (p.x, p.y)).collect();
    let short_side = dist(ordered[0], ordered[1]).min(dist(ordered[1], ordered[2]));

    (ordered, short_side)
}

/// Computes the mean probability-map score inside a polygon.
fn box_score(prob: &[f32], h: usize, w: usize, box_points: &[(f32, f32)]) -> f32 {
    let xmin = box_points
        .iter()
        .map(|p| p.0)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as usize;
    let xmax = box_points
        .iter()
        .map(|p| p.0)
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(w as f32 - 1.0) as usize;
    let ymin = box_points
        .iter()
        .map(|p| p.1)
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as usize;
    let ymax = box_points
        .iter()
        .map(|p| p.1)
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(h as f32 - 1.0) as usize;

    let mask_w = xmax - xmin + 1;
    let mask_h = ymax - ymin + 1;
    let mut mask = GrayImage::new(mask_w as u32, mask_h as u32);

    // Offset points to local coordinates
    let local_pts: Vec<(i32, i32)> = box_points
        .iter()
        .map(|p| ((p.0 - xmin as f32) as i32, (p.1 - ymin as f32) as i32))
        .collect();

    fill_polygon_scanline(&mut mask, &local_pts);

    let mut sum = 0.0f32;
    let mut count = 0usize;

    for my in 0..mask_h {
        for mx in 0..mask_w {
            if mask.get_pixel(mx as u32, my as u32).0[0] != 0 {
                let px = xmin + mx;
                let py = ymin + my;
                if py < h && px < w {
                    sum += prob[py * w + px];
                    count += 1;
                }
            }
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn fill_polygon_scanline(mask: &mut GrayImage, points: &[(i32, i32)]) {
    if points.len() < 3 {
        return;
    }

    let poly: Vec<Point<i32>> = points.iter().map(|&(x, y)| Point::new(x, y)).collect();
    draw_polygon_mut(mask, &poly, Luma([255u8]));
}

fn to_imageproc_points_f32(points: &[(f32, f32)]) -> Vec<Point<f32>> {
    points.iter().map(|&(x, y)| Point::new(x, y)).collect()
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// Expands a polygon by `unclip_ratio` using a simple corner-offset method.
///
/// For each vertex, computes the bisector direction and offsets outward.
/// This is a simplified approximation of the pyclipper unclip operation.
fn unclip_polygon(points: &[(f32, f32)], unclip_ratio: f32) -> Vec<[f32; 2]> {
    if points.len() < 3 || unclip_ratio <= 0.0 {
        return points.iter().map(|p| [p.0, p.1]).collect();
    }

    // Compute area and perimeter
    let n = points.len();
    let area = polygon_area(points);
    let perimeter = polygon_perimeter(points);
    if perimeter < 1e-6 {
        return points.iter().map(|p| [p.0, p.1]).collect();
    }

    let distance = area * unclip_ratio / perimeter;

    // Offset each vertex outward along the angle bisector
    let mut expanded: Vec<[f32; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = points[(i + n - 1) % n];
        let curr = points[i];
        let next = points[(i + 1) % n];

        // Vectors along edges pointing toward current vertex
        let v1x = curr.0 - prev.0;
        let v1y = curr.1 - prev.1;
        let v2x = next.0 - curr.0;
        let v2y = next.1 - curr.1;

        let l1 = (v1x * v1x + v1y * v1y).sqrt().max(1e-6);
        let l2 = (v2x * v2x + v2y * v2y).sqrt().max(1e-6);

        // Inward normals of the edges
        let n1x = -v1y / l1;
        let n1y = v1x / l1;
        let n2x = -v2y / l2;
        let n2y = v2x / l2;

        // Bisector direction (outward)
        let bx = -(n1x + n2x);
        let by = -(n1y + n2y);
        let blen = (bx * bx + by * by).sqrt().max(1e-6);

        // Scale bisector to give the correct offset distance
        // The offset along the bisector is distance / sin(half_angle)
        let dot = n1x * n2x + n1y * n2y;
        let half_angle = ((1.0 + dot) / 2.0).sqrt().max(1e-6);
        let offset = if half_angle > 1e-3 {
            distance / half_angle
        } else {
            distance
        };

        expanded.push([curr.0 + bx / blen * offset, curr.1 + by / blen * offset]);
    }

    expanded
}

fn polygon_area(points: &[(f32, f32)]) -> f32 {
    let n = points.len();
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].0 * points[j].1;
        area -= points[j].0 * points[i].1;
    }
    area.abs() * 0.5
}

fn polygon_perimeter(points: &[(f32, f32)]) -> f32 {
    let n = points.len();
    let mut p = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = points[j].0 - points[i].0;
        let dy = points[j].1 - points[i].1;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

// ---------------------------------------------------------------------------
// CtcDecodeNode – CTC greedy decoding
// ---------------------------------------------------------------------------

/// CTC greedy-decoding node for PP-OCR recognition output.
///
/// Takes the recognition model's CTC logits (softmax output) and produces
/// decoded character indices with per-character confidence scores.
///
/// # Inputs
///
/// | Name     | Dtype   | Shape              | Description                        |
/// |----------|---------|--------------------|------------------------------------|
/// | `logits` | Float32 | `[1, SeqLen, NumClasses]` | CTC output logits           |
///
/// # Outputs
///
/// | Name          | Dtype   | Shape | Description                              |
/// |---------------|---------|-------|------------------------------------------|
/// | `text_indices`| Int64   | `[M]` | Decoded character indices (0=blank)     |
/// | `confidence`  | Float32 | `[1]` | Mean confidence of decoded characters   |
pub struct CtcDecodeNode {
    name: String,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    blank_id: i64,
}

impl CtcDecodeNode {
    pub fn new(name: impl Into<String>, blank_id: i64) -> Self {
        let input_descriptors = HashMap::from([(
            "logits".to_owned(),
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 1, 1])
                .with_dynamic_axis(1)
                .with_dynamic_axis(2),
        )]);

        let output_descriptors = HashMap::from([
            (
                "text_indices".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1]).with_dynamic_batch(),
            ),
            (
                "confidence".to_owned(),
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1]),
            ),
        ]);

        Self {
            name: name.into(),
            input_descriptors,
            output_descriptors,
            blank_id,
        }
    }
}

#[async_trait]
impl MLNode for CtcDecodeNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
        &self.input_descriptors
    }

    fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
        &self.output_descriptors
    }

    async fn execute(
        &self,
        mut inputs: HashMap<String, MLPacket>,
        _context: &MLContext,
    ) -> Result<HashMap<String, MLPacket>, String> {
        let logit_packet = inputs
            .remove("logits")
            .ok_or("missing required input `logits`")?;

        if !inputs.is_empty() {
            let mut names = inputs.into_keys().collect::<Vec<_>>();
            names.sort();
            return Err(format!("unexpected inputs: {}", names.join(", ")));
        }

        let (ctx, logit_desc, logit_payload) = logit_packet.into_parts()?;
        let logits = match logit_payload.to_host_tensor()? {
            HostTensor::Float32(v) => v,
            other => {
                return Err(format!(
                    "CtcDecodeNode expected Float32 logits, got {other:?}"
                ));
            }
        };

        let seq_len = logit_desc.shape[1];
        let num_classes = logit_desc.shape[2];

        // Greedy CTC decode
        let (indices, confidences) =
            ctc_greedy_decode(&logits, seq_len, num_classes, self.blank_id);

        let confidence = if confidences.is_empty() {
            0.0f32
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };

        let text_indices_desc =
            MLPacketDescriptor::new(MLPacketDataType::Int64, vec![indices.len()]);
        let text_indices_packet =
            MLPacket::from_host_tensor(ctx.clone(), text_indices_desc, HostTensor::Int64(indices))?;

        let confidence_desc = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1]);
        let confidence_packet = MLPacket::from_host_tensor(
            ctx,
            confidence_desc,
            HostTensor::Float32(vec![confidence]),
        )?;

        Ok(HashMap::from([
            ("text_indices".to_owned(), text_indices_packet),
            ("confidence".to_owned(), confidence_packet),
        ]))
    }
}

/// Greedy CTC decoding: drop blanks (blank_id), merge consecutive duplicates.
fn ctc_greedy_decode(
    logits: &[f32],
    seq_len: usize,
    num_classes: usize,
    blank_id: i64,
) -> (Vec<i64>, Vec<f32>) {
    let mut indices: Vec<i64> = Vec::new();
    let mut confidences: Vec<f32> = Vec::new();
    let mut prev: i64 = -1;

    for t in 0..seq_len {
        let start = t * num_classes;
        let end = start + num_classes;
        let slice = &logits[start..end];

        // argmax
        let mut max_idx = 0usize;
        let mut max_val = f32::MIN;
        for (i, &v) in slice.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }

        let class_id = max_idx as i64;

        // Skip blank
        if class_id == blank_id {
            prev = -1;
            continue;
        }

        // Skip consecutive duplicates
        if class_id == prev {
            continue;
        }

        indices.push(class_id);
        confidences.push(max_val);
        prev = class_id;
    }

    (indices, confidences)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctc_greedy_decode() {
        // Simulate: blank(0), 'a'(1), 'a'(1), blank(0), 'b'(2), blank(0)
        // Shape: [1, 6, 3], seq_len=6, num_classes=3
        let logits = vec![
            0.9, 0.05, 0.05, // t=0: blank
            0.1, 0.8, 0.1, // t=1: 'a'
            0.1, 0.8, 0.1, // t=2: 'a' (dup)
            0.9, 0.05, 0.05, // t=3: blank
            0.1, 0.1, 0.8, // t=4: 'b'
            0.9, 0.05, 0.05, // t=5: blank
        ];

        let (indices, confidences) = ctc_greedy_decode(&logits, 6, 3, 0);
        assert_eq!(indices, vec![1, 2]); // 'a', 'b'
        assert_eq!(confidences.len(), 2);
        assert!((confidences[0] - 0.8).abs() < 1e-6);
        assert!((confidences[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_unclip_square() {
        let square = vec![(0.0f32, 0.0f32), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let expanded = unclip_polygon(&square, 1.5);
        // Should expand outward
        assert_eq!(expanded.len(), 4);
        // First vertex should move toward negative x,y
        assert!(expanded[0][0] < 0.0);
        assert!(expanded[0][1] < 0.0);
    }

    #[test]
    fn test_convex_hull_square() {
        let square = to_imageproc_points_f32(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        let hull = imageproc_geometry::convex_hull(square);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_polygon_area() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!((polygon_area(&square) - 100.0).abs() < 1e-6);
    }
}
