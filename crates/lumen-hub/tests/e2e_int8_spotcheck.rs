//! int8 runtime spot-check for siglip + bioclip: fp32 vs int8 through the full
//! service path on real images/text. siglip covers vision + text + cross-modal
//! (text is the large-const-quantized risk); bioclip covers vision via species
//! classification on a cat.
//!
//! Run on Metal: cargo test --features metal --test e2e_int8_spotcheck -- --nocapture
//! (int8 inference is slow on the CPU backend — use a GPU backend.)

mod common;

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
        let i8s = SiglipService::from_config("s", &svc_config(m, "int8", None), &cache, dev)
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
        println!("siglip int8 cross-modal: bus={sim_bus:.4} kitten={sim_kit:.4} (bus should win)");

        assert!(img_cos > 0.99, "siglip image int8 drifted: {img_cos:.5}");
        assert!(txt_cos > 0.99, "siglip TEXT int8 drifted: {txt_cos:.5}");
        assert!(sim_bus > sim_kit, "siglip int8 lost cross-modal alignment");
    }

    // ---- BioCLIP: species classification on a cat ----
    {
        let m = "bioclip-2";
        let ds = "TreeOfLife200MCore";
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
            BioclipService::from_config("b", &svc_config(m, "fp32", Some(ds)), &cache, dev.clone())
                .expect("bioclip fp32"),
        )
        .await;
        let i = classify(
            BioclipService::from_config("b", &svc_config(m, "int8", Some(ds)), &cache, dev)
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
    let Some((cache, _)) = common::require_model("siglip2-base-patch16-224", &["text", "vision"])
    else {
        return;
    };
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
