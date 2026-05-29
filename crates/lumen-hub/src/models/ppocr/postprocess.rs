//! Pure PP-OCR postprocessing: DBNet box extraction and CTC greedy decoding.
//!
//! These were previously expressed as `MLNode`s in the lumnn pipeline; with Burn
//! they are plain functions invoked directly by the task handler.

use image::{GrayImage, Luma};
use imageproc::{
    contours as imageproc_contours, drawing::draw_polygon_mut, geometry as imageproc_geometry,
    point::Point,
};

// ---------------------------------------------------------------------------
// DBNet detection postprocessing
// ---------------------------------------------------------------------------

/// Extracts text boxes from a DBNet probability map and scales them back to the
/// original image coordinate space.
///
/// `prob` is the `[H, W]` sigmoid map (`h`/`w` taken from the detection output).
/// `ratio_h`/`ratio_w` are `resize / original`, and `src_h`/`src_w` are the
/// original image dimensions. Returns 4-corner boxes `[x0,y0,..,x3,y3]`.
#[allow(clippy::too_many_arguments)]
pub fn detect_boxes(
    prob: &[f32],
    h: u32,
    w: u32,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
    ratio_h: f32,
    ratio_w: f32,
    src_h: u32,
    src_w: u32,
) -> Vec<[f32; 8]> {
    let boxes = match db_postprocess(prob, h, w, thresh, box_thresh, unclip_ratio) {
        Ok(boxes) => boxes,
        Err(_) => return Vec::new(),
    };

    boxes
        .chunks(8)
        .filter(|chunk| chunk.len() == 8)
        .map(|chunk| {
            let mut scaled = [0.0f32; 8];
            for point in 0..4 {
                let x = (chunk[point * 2] / ratio_w).clamp(0.0, src_w as f32);
                let y = (chunk[point * 2 + 1] / ratio_h).clamp(0.0, src_h as f32);
                scaled[point * 2] = x;
                scaled[point * 2 + 1] = y;
            }
            scaled
        })
        .collect()
}

fn db_postprocess(
    prob: &[f32],
    h: u32,
    w: u32,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
) -> Result<Vec<f32>, String> {
    let binary: Vec<u8> = prob
        .iter()
        .map(|&p| if p > thresh { 255u8 } else { 0u8 })
        .collect();

    let contours = find_contours(&binary, h as usize, w as usize);

    let mut boxes: Vec<f32> = Vec::new();

    for contour in &contours {
        if contour.len() < 4 {
            continue;
        }

        let (rect_points, short_side) = min_area_rect(contour);
        if short_side < 3.0 {
            continue;
        }

        let score = box_score(prob, h as usize, w as usize, &rect_points);
        if score < box_thresh {
            continue;
        }

        let expanded = unclip_polygon(&rect_points, unclip_ratio);
        let expanded_contour: Vec<(f32, f32)> = expanded.iter().map(|p| (p[0], p[1])).collect();
        let (final_points, final_short_side) = min_area_rect(&expanded_contour);
        if final_short_side < 5.0 {
            continue;
        }

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
fn unclip_polygon(points: &[(f32, f32)], unclip_ratio: f32) -> Vec<[f32; 2]> {
    if points.len() < 3 || unclip_ratio <= 0.0 {
        return points.iter().map(|p| [p.0, p.1]).collect();
    }

    let n = points.len();
    let area = polygon_area(points);
    let perimeter = polygon_perimeter(points);
    if perimeter < 1e-6 {
        return points.iter().map(|p| [p.0, p.1]).collect();
    }

    let distance = area * unclip_ratio / perimeter;

    let mut expanded: Vec<[f32; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = points[(i + n - 1) % n];
        let curr = points[i];
        let next = points[(i + 1) % n];

        let v1x = curr.0 - prev.0;
        let v1y = curr.1 - prev.1;
        let v2x = next.0 - curr.0;
        let v2y = next.1 - curr.1;

        let l1 = (v1x * v1x + v1y * v1y).sqrt().max(1e-6);
        let l2 = (v2x * v2x + v2y * v2y).sqrt().max(1e-6);

        let n1x = -v1y / l1;
        let n1y = v1x / l1;
        let n2x = -v2y / l2;
        let n2y = v2x / l2;

        let bx = -(n1x + n2x);
        let by = -(n1y + n2y);
        let blen = (bx * bx + by * by).sqrt().max(1e-6);

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
// CTC greedy decoding
// ---------------------------------------------------------------------------

/// Greedy CTC decode of `[seq_len, num_classes]` logits: drop blanks, merge
/// consecutive duplicates. Returns decoded indices and the mean confidence.
pub fn ctc_greedy_decode(
    logits: &[f32],
    seq_len: usize,
    num_classes: usize,
    blank_id: i64,
) -> (Vec<i64>, f32) {
    let mut indices: Vec<i64> = Vec::new();
    let mut confidences: Vec<f32> = Vec::new();
    let mut prev: i64 = -1;

    for t in 0..seq_len {
        let start = t * num_classes;
        let end = start + num_classes;
        let slice = &logits[start..end];

        let mut max_idx = 0usize;
        let mut max_val = f32::MIN;
        for (i, &v) in slice.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }

        let class_id = max_idx as i64;

        if class_id == blank_id {
            prev = -1;
            continue;
        }
        if class_id == prev {
            continue;
        }

        indices.push(class_id);
        confidences.push(max_val);
        prev = class_id;
    }

    let confidence = if confidences.is_empty() {
        0.0f32
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    (indices, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctc_greedy_decode() {
        let logits = vec![
            0.9, 0.05, 0.05, // t=0: blank
            0.1, 0.8, 0.1, // t=1: 'a'
            0.1, 0.8, 0.1, // t=2: 'a' (dup)
            0.9, 0.05, 0.05, // t=3: blank
            0.1, 0.1, 0.8, // t=4: 'b'
            0.9, 0.05, 0.05, // t=5: blank
        ];

        let (indices, conf) = ctc_greedy_decode(&logits, 6, 3, 0);
        assert_eq!(indices, vec![1, 2]);
        assert!((conf - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_unclip_square() {
        let square = vec![(0.0f32, 0.0f32), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let expanded = unclip_polygon(&square, 1.5);
        assert_eq!(expanded.len(), 4);
        assert!(expanded[0][0] < 0.0);
        assert!(expanded[0][1] < 0.0);
    }

    #[test]
    fn test_polygon_area() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!((polygon_area(&square) - 100.0).abs() < 1e-6);
    }
}
