use std::{fs, path::Path};

#[test]
fn core_does_not_reference_backend_specific_providers() {
    let core_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/core");
    let mut violations = Vec::new();
    collect_token_references(&core_dir, BACKEND_TOKENS, &mut violations);

    assert!(
        violations.is_empty(),
        "core must remain runtime-agnostic; backend references found:\n{}",
        violations.join("\n")
    );
}

#[test]
fn lumnn_does_not_reference_lumen_product_layers() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations = Vec::new();
    collect_token_references(&src_dir, PRODUCT_LAYER_TOKENS, &mut violations);

    assert!(
        violations.is_empty(),
        "lumnn must remain product-agnostic; Lumen hub references found:\n{}",
        violations.join("\n")
    );
}

fn collect_token_references(dir: &Path, tokens: &[&str], violations: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("core directory should be readable") {
        let entry = entry.expect("core directory entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            collect_token_references(&path, tokens, violations);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }

        let contents = fs::read_to_string(&path).expect("core source file should be readable");
        for token in tokens {
            if contents.contains(token) {
                violations.push(format!("{} contains `{token}`", path.display()));
            }
        }
    }
}

const BACKEND_TOKENS: &[&str] = &[
    "crate::ort",
    "ort::",
    "ONNX Runtime",
    "OpenVINO",
    "TensorRT",
    "CoreML",
    "DirectML",
    "CUDA",
    "Metal",
    "candle",
    "Candle",
];

const PRODUCT_LAYER_TOKENS: &[&str] = &[
    "daemon",
    "service",
    "tonic",
    "mdns",
    "clip",
    "siglip",
    "lumen_hub",
    "lumen-hub",
];
