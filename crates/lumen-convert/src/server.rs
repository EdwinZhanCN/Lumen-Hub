//! Burn architectures generated from the pp-ocrv5-server ONNX by `build.rs`
//! (burn-onnx `ModelGen`). The `.rs` + fp32 `.bpk` live in `OUT_DIR`; these modules
//! `include!` the generated code so the quantize binary can load and quantize them.
//!
//! Once validated these graduate into `lumen-hub/src/model_arch/` for runtime use.
#![allow(
    clippy::all,
    clippy::pedantic,
    dead_code,
    non_snake_case,
    unused_imports
)]

pub mod detection {
    include!(concat!(
        env!("OUT_DIR"),
        "/pp_ocrv5_server/detection/detection.rs"
    ));
}

pub mod recognition {
    include!(concat!(
        env!("OUT_DIR"),
        "/pp_ocrv5_server/recognition/recognition.rs"
    ));
}

// Orientation classifier: its ONNX input is dynamic, which burn-onnx can't codegen,
// so onnx_prep pins a static [1,3,80,160] crop (the model is fully-conv → 2 classes).
pub mod classification {
    include!(concat!(
        env!("OUT_DIR"),
        "/pp_ocrv5_server/classification/classification.rs"
    ));
}
