use hnswlib_rs::InnerProduct;
use hnswlib_rs::legacy::load_hnswlib;

fn main() {
    let bytes = Vec::new();
    if let Ok((graph, _vectors)) = load_hnswlib(InnerProduct::new(), 768, &bytes) {
        graph.set_ef_search(100);
    }
}
