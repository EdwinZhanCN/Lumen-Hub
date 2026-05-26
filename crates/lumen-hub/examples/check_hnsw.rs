use hnswlib_rs::legacy::load_hnswlib;
use hnswlib_rs::InnerProduct;

fn main() {
    let bytes = vec![0u8; 100];
    if let Ok((mut graph, vectors)) = load_hnswlib(InnerProduct::<f32>::new(), 768, &bytes) {
        // Test if set_ef exists
        graph.set_ef(100);
    }
}
