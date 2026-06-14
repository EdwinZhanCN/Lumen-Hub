//! Generate Burn `Model<B>` architectures + fp32 weights from prepared ONNX.
//!
//! Each `onnx/<repo>/<component>.prepared.onnx` (produced by tools/onnx_prep.py)
//! is run through burn-onnx's `ModelGen`, emitting `<component>.prepared.rs` and a
//! `.bpk` weights file under `OUT_DIR/<out_dir>/`. The quantize binary includes the
//! generated modules and reads the fp32 `.bpk`. Inputs that are absent are skipped
//! so the crate still builds before any ONNX is dropped in.

use std::{env, fs, path::Path, path::PathBuf};

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
        (
            "onnx/pp-ocrv6-small/detection.prepared.onnx",
            "pp_ocrv6_small/detection/",
        ),
        (
            "onnx/pp-ocrv6-small/recognition.prepared.onnx",
            "pp_ocrv6_small/recognition/",
        ),
        (
            "onnx/pp-ocrv6-small/classification.prepared.onnx",
            "pp_ocrv6_small/classification/",
        ),
    ];

    for (input, out_dir) in models {
        println!("cargo:rerun-if-changed={input}");
        if !Path::new(input).exists() {
            println!("cargo:warning=lumen-convert: missing {input}, generating compile-only stub");
            write_stub_module(out_dir).expect("write missing-model stub");
            continue;
        }
        ModelGen::new()
            .input(input)
            .out_dir(out_dir)
            .run_from_script();
    }
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
