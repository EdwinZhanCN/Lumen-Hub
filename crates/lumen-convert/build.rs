//! Generate Burn `Model<B>` architectures + fp32 weights from prepared ONNX.
//!
//! Each `onnx/<repo>/<component>.prepared.onnx` (produced by tools/onnx_prep.py)
//! is run through burn-onnx's `ModelGen`, emitting `<component>.prepared.rs` and a
//! `.bpk` weights file under `OUT_DIR/<out_dir>/`. The quantize binary includes the
//! generated modules and reads the fp32 `.bpk`. Inputs that are absent are skipped
//! so the crate still builds before any ONNX is dropped in.

use std::{env, fs, path::Path, path::PathBuf};

use burn_onnx::ModelGen;

/// Codegen kind, selecting the compile-only stub written when the prepared ONNX
/// is absent (the `onnx/` tree is gitignored, so CI/check builds have no assets).
#[derive(Clone, Copy)]
enum Kind {
    /// PP-OCR component: `forward(Tensor<4>) -> Tensor<rank>` keyed by component name.
    Ocr,
    /// Aesthetic head MLP: `forward(Tensor<2>) -> Tensor<1>`, generated file `aesthetic.rs`.
    Head,
}

fn main() {
    // (prepared onnx, out_dir under OUT_DIR, codegen kind)
    let models = [
        (
            "onnx/pp-ocrv5-server/detection.prepared.onnx",
            "pp_ocrv5_server/detection/",
            Kind::Ocr,
        ),
        (
            "onnx/pp-ocrv5-server/recognition.prepared.onnx",
            "pp_ocrv5_server/recognition/",
            Kind::Ocr,
        ),
        (
            "onnx/pp-ocrv5-server/classification.prepared.onnx",
            "pp_ocrv5_server/classification/",
            Kind::Ocr,
        ),
        (
            "onnx/pp-ocrv6-small/detection.prepared.onnx",
            "pp_ocrv6_small/detection/",
            Kind::Ocr,
        ),
        (
            "onnx/pp-ocrv6-small/recognition.prepared.onnx",
            "pp_ocrv6_small/recognition/",
            Kind::Ocr,
        ),
        (
            "onnx/pp-ocrv6-small/classification.prepared.onnx",
            "pp_ocrv6_small/classification/",
            Kind::Ocr,
        ),
        (
            "onnx/aesthetic-head-siglip2-base-patch16-224-ava/aesthetic.prepared.onnx",
            "aesthetic_head/siglip2_base_patch16_224/",
            Kind::Head,
        ),
        (
            "onnx/aesthetic-head-siglip2-so400m-patch14-384-ava/aesthetic.prepared.onnx",
            "aesthetic_head/siglip2_so400m_patch14_384/",
            Kind::Head,
        ),
    ];

    for (input, out_dir, kind) in models {
        println!("cargo:rerun-if-changed={input}");
        if !Path::new(input).exists() {
            println!("cargo:warning=lumen-convert: missing {input}, generating compile-only stub");
            match kind {
                Kind::Ocr => write_stub_module(out_dir).expect("write missing-model stub"),
                Kind::Head => write_head_stub(out_dir).expect("write missing-head stub"),
            }
            continue;
        }
        ModelGen::new()
            .input(input)
            .out_dir(out_dir)
            .run_from_script();
    }
}

/// Compile-only stub for an aesthetic head when its prepared ONNX is absent.
/// The generated file is named after the input stem (`aesthetic.rs`), independent
/// of the out_dir's last segment.
fn write_head_stub(out_dir: &str) -> std::io::Result<()> {
    let stub_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set"))
        .join(out_dir)
        .join("aesthetic.rs");
    let parent = stub_path.parent().expect("stub parent");
    fs::create_dir_all(parent)?;
    fs::write(stub_path, head_stub_source())
}

fn head_stub_source() -> String {
    r#"// Auto-generated stub because the prepared ONNX for the aesthetic head is absent.
// Keeps `lumen-convert` compiling in CI/check builds that do not ship model assets.
use burn::prelude::*;
use burn::tensor::Bytes;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    phantom: core::marker::PhantomData<B>,
}

extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::new(&Default::default())
    }
}

impl<B: Backend> Model<B> {
    pub fn from_file<P: AsRef<std::path::Path>>(_file: P, device: &B::Device) -> Self {
        Self::new(device)
    }

    pub fn from_bytes(_bytes: Bytes, device: &B::Device) -> Self {
        Self::new(device)
    }

    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        Self {
            phantom: core::marker::PhantomData,
        }
    }

    pub fn forward(&self, _x: Tensor<B, 2>) -> Tensor<B, 1> {
        panic!(
            "lumen-convert aesthetic-head stub was used at runtime; add the prepared ONNX and rebuild"
        )
    }
}
"#
    .to_string()
}

fn write_stub_module(out_dir: &str) -> std::io::Result<()> {
    let component = out_dir
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .expect("component in out_dir");
    let stub_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set"))
        .join(out_dir)
        .join(format!("{component}.rs"));
    let parent = stub_path.parent().expect("stub parent");
    fs::create_dir_all(parent)?;
    fs::write(stub_path, stub_source(component))
}

fn stub_source(component: &str) -> String {
    let output_rank = match component {
        "detection" => 4,
        "recognition" => 3,
        "classification" => 2,
        other => panic!("unsupported stub component `{other}`"),
    };
    format!(
        r#"// Auto-generated stub because the prepared ONNX for `{component}` is absent.
// This keeps `lumen-convert` compiling in CI/check builds that do not ship model assets.
use burn::prelude::*;
use burn::tensor::Bytes;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {{
    phantom: core::marker::PhantomData<B>,
}}

extern crate std;

impl<B: Backend> Default for Model<B> {{
    fn default() -> Self {{
        Self::new(&Default::default())
    }}
}}

impl<B: Backend> Model<B> {{
    pub fn from_file<P: AsRef<std::path::Path>>(_file: P, device: &B::Device) -> Self {{
        Self::new(device)
    }}

    pub fn from_bytes(_bytes: Bytes, device: &B::Device) -> Self {{
        Self::new(device)
    }}

    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {{
        Self {{
            phantom: core::marker::PhantomData,
        }}
    }}

    pub fn forward(&self, _x: Tensor<B, 4>) -> Tensor<B, {output_rank}> {{
        panic!(
            "lumen-convert stub model `{component}` was used at runtime; add the prepared ONNX and rebuild"
        )
    }}
}}
"#
    )
}
