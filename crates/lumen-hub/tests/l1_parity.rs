//! L1 parity suite: quantized (fp16q8/int8) outputs vs fp32 on the same
//! backend, using real production weights. Env-gated like l1_models; the
//! bodies are the proven int8 comparison suites, consolidated.
//!
//! Run: LUMEN_MODELS_DIR=/path/to/lumen-models cargo test --release --test l1_parity

mod common;

mod insightface_int8 {
    use crate::common;
    // int8 runtime spot-check for antelopev2: compares fp32 vs int8 through the full
    // InsightFace service path. Embedding fidelity (ArcFace recognition — the
    // borderline component) is measured on a single clear face; detection parity on a
    // multi-face image.
    //
    // Run on CPU:   cargo test --test e2e_insightface_int8 -- --nocapture
    // Run on Metal: cargo test --features metal --test e2e_insightface_int8 -- --nocapture

    use std::collections::BTreeMap;
    use std::sync::Arc;

    use lumen_hub::backend::default_device;
    use lumen_hub::models::insightface::InsightFaceService;
    use lumen_hub::service::{InferenceService, TaskRequest};
    use lumen_schema::{FaceV1, ModelConfig, Runtime, ServiceConfig};

    const MODEL: &str = "antelopev2";

    fn config(model: &str, precision: &str) -> ServiceConfig {
        ServiceConfig {
            enabled: true,
            package: "insightface".to_owned(),
            models: BTreeMap::from([(
                "default".to_owned(),
                ModelConfig {
                    model: model.to_owned(),
                    runtime: Runtime::Burn,
                    dataset: None,
                    precision: Some(precision.to_owned()),
                },
            )]),
        }
    }

    async fn run(cache_dir: &str, model: &str, precision: &str, rel: &str, mime: &str) -> FaceV1 {
        let device = Arc::new(default_device());
        let service = InsightFaceService::from_config(
            "insightface",
            &config(model, precision),
            cache_dir,
            device,
        )
        .unwrap_or_else(|e| panic!("service builds ({precision}): {e:?}"));
        let task = service
            .tasks()
            .task_names()
            .into_iter()
            .next()
            .expect("task");
        let img = common::sample_bytes(rel);
        let result = service
            .tasks()
            .handle(&task, TaskRequest::new(img, mime))
            .await
            .unwrap_or_else(|e| panic!("face task ({precision}, {rel}): {e:?}"));
        serde_json::from_slice(&result.payload).expect("face_v1 JSON")
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-12)
    }

    async fn body(cache_dir: String, model: String) {
        // --- ArcFace embedding fidelity on a single clear face ---
        let f32_face = run(
            &cache_dir,
            &model,
            "fp32",
            "warmup/face/face.jpg",
            "image/jpeg",
        )
        .await;
        let i8_face = run(
            &cache_dir,
            &model,
            "fp16q8",
            "warmup/face/face.jpg",
            "image/jpeg",
        )
        .await;
        assert!(
            f32_face.count > 0 && i8_face.count > 0,
            "expected a face in face.jpg"
        );
        let e32 = f32_face.faces[0]
            .embedding
            .as_ref()
            .expect("fp32 embedding");
        let e8 = i8_face.faces[0].embedding.as_ref().expect("int8 embedding");
        let emb_cos = cosine(e32, e8);
        eprintln!("\n=== antelopev2 int8 runtime spot-check ===");
        eprintln!(
            "face.jpg: embedding cosine fp32-vs-int8 = {emb_cos:.5} (dim {})",
            e32.len()
        );

        // --- detection parity on a multi-face image ---
        let f32_multi = run(
            &cache_dir,
            &model,
            "fp32",
            "tests/test_sample/face_test_1.png",
            "image/png",
        )
        .await;
        let i8_multi = run(
            &cache_dir,
            &model,
            "fp16q8",
            "tests/test_sample/face_test_1.png",
            "image/png",
        )
        .await;
        eprintln!(
            "face_test_1.png: detected faces fp32={} int8={}",
            f32_multi.count, i8_multi.count
        );

        assert!(
            emb_cos > 0.95,
            "int8 ArcFace embedding drifted too far: cosine {emb_cos:.5}"
        );
        assert!(i8_multi.count > 0, "int8 detection found no faces");
    }

    #[test]
    fn insightface_int8_matches_fp32() {
        let Some((cache_dir, model_name)) =
            common::require_model(MODEL, &["detection", "recognition"])
        else {
            return;
        };
        if common::require_model_precision(MODEL, &["detection", "recognition"], "fp16q8").is_none()
        {
            return;
        }
        const STACK: usize = 256 * 1024 * 1024;
        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(move || {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .thread_stack_size(STACK)
                    .build()
                    .expect("tokio runtime")
                    .block_on(body(cache_dir, model_name));
            })
            .expect("spawn test thread")
            .join()
            .expect("test thread panicked");
    }
}

mod int8_spotcheck {
    use crate::common;
    // int8 runtime spot-check for siglip + bioclip: fp32 vs int8 through the full
    // service path on real images/text. siglip covers vision + text + cross-modal
    // (text is the large-const-quantized risk); bioclip covers vision via species
    // classification on a cat.
    //
    // Run on Metal: cargo test --features metal --test e2e_int8_spotcheck -- --nocapture
    // (int8 inference is slow on the CPU backend — use a GPU backend.)

    use std::collections::BTreeMap;
    use std::sync::Arc;

    use lumen_hub::backend::default_device;
    use lumen_hub::models::bioclip::BioclipService;
    use lumen_hub::models::siglip::SiglipService;
    use lumen_hub::service::{InferenceService, TaskRequest};
    use lumen_schema::{EmbeddingV1, LabelsV1, ModelConfig, Runtime, ServiceConfig};

    fn svc_config(model: &str, precision: &str, dataset: Option<&str>) -> ServiceConfig {
        ServiceConfig {
            enabled: true,
            package: "spotcheck".to_owned(),
            models: BTreeMap::from([(
                "default".to_owned(),
                ModelConfig {
                    model: model.to_owned(),
                    runtime: Runtime::Burn,
                    dataset: dataset.map(str::to_owned),
                    precision: Some(precision.to_owned()),
                },
            )]),
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-12)
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    async fn emb(svc: &SiglipService, task: &str, req: TaskRequest) -> Vec<f32> {
        let r = svc.tasks().handle(task, req).await.expect("siglip task");
        serde_json::from_slice::<EmbeddingV1>(&r.payload)
            .expect("embedding_v1")
            .vector
    }

    fn img_req(rel: &str) -> TaskRequest {
        TaskRequest::new(common::sample_bytes(rel), "image/jpeg")
    }
    fn text_req(s: &str) -> TaskRequest {
        TaskRequest::new(s.as_bytes().to_vec(), "text/plain")
    }

    async fn body(cache: String) {
        println!("\n=== int8 runtime spot-check (fp32 vs int8) ===");

        // ---- SigLIP base: vision + text + cross-modal ----
        {
            let m = "siglip2-base-patch16-224";
            let dev = Arc::new(default_device());
            let f32s =
                SiglipService::from_config("s", &svc_config(m, "fp32", None), &cache, dev.clone())
                    .expect("siglip fp32");
            let i8s = SiglipService::from_config("s", &svc_config(m, "fp16q8", None), &cache, dev)
                .expect("siglip int8");

            let img32 = emb(
                &f32s,
                "semantic_image_embed",
                img_req("warmup/semantic/bus.jpg"),
            )
            .await;
            let img8 = emb(
                &i8s,
                "semantic_image_embed",
                img_req("warmup/semantic/bus.jpg"),
            )
            .await;
            let txt32 = emb(&f32s, "semantic_text_embed", text_req("a photo of a bus")).await;
            let txt8 = emb(&i8s, "semantic_text_embed", text_req("a photo of a bus")).await;
            let kit8 = emb(&i8s, "semantic_text_embed", text_req("a photo of a kitten")).await;

            let img_cos = cosine(&img32, &img8);
            let txt_cos = cosine(&txt32, &txt8);
            let (sim_bus, sim_kit) = (dot(&img8, &txt8), dot(&img8, &kit8));
            println!("siglip image  cosine fp32-vs-int8 = {img_cos:.5}");
            println!("siglip text   cosine fp32-vs-int8 = {txt_cos:.5}");
            println!(
                "siglip int8 cross-modal: bus={sim_bus:.4} kitten={sim_kit:.4} (bus should win)"
            );

            assert!(img_cos > 0.99, "siglip image int8 drifted: {img_cos:.5}");
            assert!(txt_cos > 0.99, "siglip TEXT int8 drifted: {txt_cos:.5}");
            assert!(sim_bus > sim_kit, "siglip int8 lost cross-modal alignment");
        }

        // ---- BioCLIP: species classification on a cat ----
        {
            let m = "bioclip-2";
            let ds = "TreeOfLife200MCore";
            if !common::has_burn_weights(&cache, m, &["vision"], "fp32")
                || !common::has_burn_weights(&cache, m, &["vision"], "fp16q8")
            {
                println!("bioclip: SKIP (missing fp32/int8 vision weights)");
                return;
            }
            let have_ds = std::path::Path::new(&cache)
                .join(m)
                .join("datasets")
                .join(format!("{ds}.npy"))
                .is_file();
            if !have_ds {
                println!("bioclip: SKIP (no {ds} catalog)");
                return;
            }
            let dev = Arc::new(default_device());
            let classify = |svc: BioclipService| async move {
                let req = TaskRequest::new(
                    common::sample_bytes("warmup/bio/abyssinian.jpg"),
                    "image/jpeg",
                )
                .with_meta("top_k", "5");
                let r = svc
                    .tasks()
                    .handle("bioclip_classify", req)
                    .await
                    .expect("bioclip task");
                serde_json::from_slice::<LabelsV1>(&r.payload).expect("labels_v1")
            };
            let f = classify(
                BioclipService::from_config(
                    "b",
                    &svc_config(m, "fp32", Some(ds)),
                    &cache,
                    dev.clone(),
                )
                .expect("bioclip fp32"),
            )
            .await;
            let i = classify(
                BioclipService::from_config("b", &svc_config(m, "fp16q8", Some(ds)), &cache, dev)
                    .expect("bioclip int8"),
            )
            .await;
            println!(
                "bioclip top-1  fp32={:?} ({:.3})  int8={:?} ({:.3})",
                f.labels[0].label, f.labels[0].score, i.labels[0].label, i.labels[0].score
            );
            assert_eq!(
                f.labels[0].label, i.labels[0].label,
                "bioclip int8 changed the top species"
            );
        }
    }

    #[test]
    fn int8_spotcheck_siglip_bioclip() {
        let Some((cache, _)) =
            common::require_model("siglip2-base-patch16-224", &["text", "vision"])
        else {
            return;
        };
        if common::require_model_precision(
            "siglip2-base-patch16-224",
            &["text", "vision"],
            "fp16q8",
        )
        .is_none()
        {
            return;
        }
        const STACK: usize = 256 * 1024 * 1024;
        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(move || {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .thread_stack_size(STACK)
                    .build()
                    .expect("tokio runtime")
                    .block_on(body(cache));
            })
            .expect("spawn test thread")
            .join()
            .expect("test thread panicked");
    }
}
