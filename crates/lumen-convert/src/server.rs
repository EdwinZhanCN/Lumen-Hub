//! Burn architectures generated from PP-OCR ONNX graphs by `build.rs`
//! (burn-onnx `ModelGen`). The `.rs` + fp32 `.bpk` live in `OUT_DIR`; these modules
//! `include!` the generated code so helper binaries can stage/export them.
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

// Aesthetic scoring heads (tiny MLP on SigLIP2 vision pooled features),
// generated from the head ONNX. Graduated copies live in lumen-hub model_arch.
pub mod aesthetic_head {
    pub mod siglip2_base_patch16_224 {
        include!(concat!(
            env!("OUT_DIR"),
            "/aesthetic_head/siglip2_base_patch16_224/aesthetic.rs"
        ));
    }

    pub mod siglip2_so400m_patch14_384 {
        include!(concat!(
            env!("OUT_DIR"),
            "/aesthetic_head/siglip2_so400m_patch14_384/aesthetic.rs"
        ));
    }
}

pub mod pp_ocrv6_small {
    pub mod detection {
        include!(concat!(
            env!("OUT_DIR"),
            "/pp_ocrv6_small/detection/detection.rs"
        ));
    }

    pub mod recognition {
        include!(concat!(
            env!("OUT_DIR"),
            "/pp_ocrv6_small/recognition/recognition.rs"
        ));
    }

    // PP-OCRv6-small uses the text-line orientation classifier too. As with the
    // server pack above, onnx_prep pins a static [1,3,48,192] crop so burn-onnx
    // can fold the dynamic shape subgraph away.
    pub mod classification {
        include!(concat!(
            env!("OUT_DIR"),
            "/pp_ocrv6_small/classification/classification.rs"
        ));
    }
}
