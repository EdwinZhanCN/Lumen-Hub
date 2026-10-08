//! Runs the committed qa-tiny fixture (fixtures/iree/qa-tiny) on the CPU driver.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use lumen_iree::{Driver, Model, ParamsMode, Runtime, TensorData, TensorRef};

#[cfg(target_arch = "x86_64")]
const CPU_TARGET: &str = "cpu-x86_64";
#[cfg(target_arch = "aarch64")]
const CPU_TARGET: &str = "cpu-aarch64";

fn fixture_dir() -> PathBuf {
    std::env::var_os("LUMEN_IREE_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/iree/qa-tiny")
        })
}

fn input() -> Vec<f32> {
    (0..3072).map(|i| (i % 251) as f32 / 251.0).collect()
}

fn expected(precision: &str) -> Vec<f32> {
    let text = std::fs::read_to_string(fixture_dir().join("expected.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["outputs"][precision]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn load(runtime: &Arc<Runtime>, precision: &str, mode: ParamsMode) -> Model {
    let dir = fixture_dir().join("iree");
    Model::load(
        runtime,
        &dir.join(format!("net.{precision}.{CPU_TARGET}.vmfb")),
        &dir.join(format!("net.{precision}.irpa")),
        mode,
    )
    .unwrap()
}

#[test]
fn matches_onnxruntime_reference_for_both_precisions_and_param_modes() {
    let runtime = Runtime::new(Driver::LocalTask).unwrap();
    let x = input();
    for precision in ["fp32", "w8a32"] {
        for mode in [ParamsMode::Mmap, ParamsMode::Read] {
            let model = load(&runtime, precision, mode);
            assert!(model.has_function("main"));
            assert!(!model.has_function("missing"));
            let out = model
                .invoke(
                    "main",
                    &[TensorRef::F32 {
                        dims: &[1, 3072],
                        data: &x,
                    }],
                )
                .unwrap();
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].dims, vec![1, 16]);
            let TensorData::F32(y) = &out[0].data else {
                panic!("f32 output expected")
            };
            let c = cosine(y, &expected(precision));
            assert!(c > 0.99999, "{precision} {mode:?}: cosine {c}");
            model.trim().unwrap();
        }
    }
    let w8 = load(&runtime, "w8a32", ParamsMode::Mmap)
        .invoke(
            "main",
            &[TensorRef::F32 {
                dims: &[1, 3072],
                data: &x,
            }],
        )
        .unwrap()
        .remove(0)
        .into_f32()
        .unwrap();
    assert!(cosine(&w8, &expected("fp32")) > 0.999);
}

#[test]
fn reports_errors_instead_of_crashing() {
    let runtime = Runtime::new(Driver::LocalTask).unwrap();
    let model = load(&runtime, "fp32", ParamsMode::Mmap);
    let x = input();
    let wrong_shape = model.invoke(
        "main",
        &[TensorRef::F32 {
            dims: &[1, 1536],
            data: &x[..1536],
        }],
    );
    assert!(wrong_shape.unwrap_err().message().contains("mismatch"));
    let bad_len = model.invoke(
        "main",
        &[TensorRef::F32 {
            dims: &[1, 3072],
            data: &x[..10],
        }],
    );
    assert!(bad_len.is_err());
    let ids = vec![0i64; 3072];
    assert!(
        model
            .invoke(
                "main",
                &[TensorRef::I64 {
                    dims: &[1, 3072],
                    data: &ids
                }]
            )
            .is_err()
    );
    assert!(model.invoke("missing", &[]).is_err());
    assert!(model.invoke("main", &[]).is_err());
    let dir = fixture_dir().join("iree");
    assert!(
        Model::load(
            &runtime,
            &dir.join("absent.vmfb"),
            &dir.join("net.fp32.irpa"),
            ParamsMode::Mmap
        )
        .is_err()
    );
    // Wrong ISA: the other CPU target must be rejected, not executed.
    let other = if CPU_TARGET == "cpu-x86_64" {
        "cpu-aarch64"
    } else {
        "cpu-x86_64"
    };
    assert!(
        Model::load(
            &runtime,
            &dir.join(format!("net.fp32.{other}.vmfb")),
            &dir.join("net.fp32.irpa"),
            ParamsMode::Mmap
        )
        .is_err()
    );
}

#[test]
fn shared_model_is_safe_across_threads() {
    let runtime = Runtime::new(Driver::LocalTask).unwrap();
    let model = Arc::new(load(&runtime, "fp32", ParamsMode::Mmap));
    drop(runtime); // the model keeps the runtime alive
    let reference = expected("fp32");
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let model = Arc::clone(&model);
            let reference = reference.clone();
            thread::spawn(move || {
                let x = input();
                for _ in 0..25 {
                    let y = model
                        .invoke(
                            "main",
                            &[TensorRef::F32 {
                                dims: &[1, 3072],
                                data: &x,
                            }],
                        )
                        .unwrap()
                        .remove(0)
                        .into_f32()
                        .unwrap();
                    assert!(cosine(&y, &reference) > 0.99999);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
