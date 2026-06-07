//! Generate Burn `Model<B>` architectures + fp32 weights from prepared ONNX.
//!
//! Each `onnx/<repo>/<component>.prepared.onnx` (produced by tools/onnx_prep.py)
//! is run through burn-onnx's `ModelGen`, emitting `<component>.prepared.rs` and a
//! `.bpk` weights file under `OUT_DIR/<out_dir>/`. The quantize binary includes the
//! generated modules and reads the fp32 `.bpk`. Inputs that are absent are skipped
//! so the crate still builds before any ONNX is dropped in.

use std::path::Path;

use burn_onnx::ModelGen;

fn main() {
    // (prepared onnx, out_dir under OUT_DIR)
    let models = [
        (
            "onnx/pp-ocrv5-server/detection.prepared.onnx",
            "pp_ocrv5_server/detection/",
        ),
        (
            "onnx/pp-ocrv5-server/recognition.prepared.onnx",
            "pp_ocrv5_server/recognition/",
        ),
        (
            "onnx/pp-ocrv5-server/classification.prepared.onnx",
            "pp_ocrv5_server/classification/",
        ),
    ];

    for (input, out_dir) in models {
        println!("cargo:rerun-if-changed={input}");
        if !Path::new(input).exists() {
            println!("cargo:warning=lumen-convert: missing {input}, skipping codegen");
            continue;
        }
        ModelGen::new()
            .input(input)
            .out_dir(out_dir)
            .run_from_script();
    }
}
