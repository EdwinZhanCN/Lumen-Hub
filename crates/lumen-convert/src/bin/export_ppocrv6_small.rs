use std::{fs, path::Path};

use burn::module::Module;
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use lumen_convert::server::pp_ocrv6_small::{classification, detection, recognition};
use lumen_hub::backend::{Backend, Device, default_device};

const DEFAULT_ROOT: &str = "lumen-models";
const MODEL: &str = "pp-ocrv6-small";

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ROOT.to_owned());
    let burn_dir = format!("{root}/{MODEL}/burn");
    fs::create_dir_all(&burn_dir).expect("create model burn dir");

    stage_generated_fp32(
        "pp_ocrv6_small/detection/detection.bpk",
        &burn_dir,
        "detection",
    );
    export_fp16::<detection::Model<Backend>, _>(&burn_dir, "detection", |path, device| {
        detection::Model::<Backend>::from_file(path, device)
    });

    stage_generated_fp32(
        "pp_ocrv6_small/recognition/recognition.bpk",
        &burn_dir,
        "recognition",
    );
    export_fp16::<recognition::Model<Backend>, _>(&burn_dir, "recognition", |path, device| {
        recognition::Model::<Backend>::from_file(path, device)
    });

    stage_generated_fp32(
        "pp_ocrv6_small/classification/classification.bpk",
        &burn_dir,
        "classification",
    );
    export_fp16::<classification::Model<Backend>, _>(
        &burn_dir,
        "classification",
        |path, device| classification::Model::<Backend>::from_file(path, device),
    );
}

fn stage_generated_fp32(gen_rel: &str, burn_dir: &str, component: &str) {
    let src = format!("{}/{}", env!("OUT_DIR"), gen_rel);
    let dst = format!("{burn_dir}/{component}.fp32.bpk");
    fs::copy(&src, &dst).unwrap_or_else(|e| panic!("stage {src} -> {dst}: {e}"));
}

fn export_fp16<M, L>(burn_dir: &str, component: &str, load: L)
where
    M: Module<Backend> + ModuleSnapshot<Backend>,
    L: Fn(&str, &Device) -> M,
{
    let fp32_path = format!("{burn_dir}/{component}.fp32.bpk");
    let fp16_path = format!("{burn_dir}/{component}.fp16.bpk");
    let fp16q8_path = format!("{burn_dir}/{component}.fp16q8.bpk");

    if !Path::new(&fp32_path).is_file() {
        panic!("missing staged fp32 burnpack for {component}: {fp32_path}");
    }

    let device = default_device();
    let model = load(&fp32_path, &device);
    let mut store = BurnpackStore::from_file(&fp16_path)
        .overwrite(true)
        .with_to_adapter(HalfPrecisionAdapter::new());
    model
        .save_into(&mut store)
        .unwrap_or_else(|e| panic!("save fp16 burnpack for {component}: {e}"));
    fs::copy(&fp16_path, &fp16q8_path)
        .unwrap_or_else(|e| panic!("copy {fp16_path} -> {fp16q8_path}: {e}"));
}
