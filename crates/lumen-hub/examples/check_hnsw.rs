use hnswlib_rs::HnswIndex;

fn inner_product_distance(a: &[f32], b: &[f32]) -> f64 {
    let dot = a.iter().zip(b).map(|(lhs, rhs)| lhs * rhs).sum::<f32>();
    f64::from(1.0 - dot)
}

fn main() {
    let _ = std::mem::size_of::<HnswIndex<'static>>();
    let _ = inner_product_distance(&[1.0, 0.0], &[1.0, 0.0]);
}
