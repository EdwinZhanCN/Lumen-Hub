//! Stage the generated aesthetic-head fp32 burnpack and emit fp16 + fp16q8.
//!
//! The head is a tiny MLP on the SigLIP2 vision pooled features. It is the final
//! scalar output projection, so it is deliberately kept out of the Q8 path: the
//! `fp16q8` artifact is a byte-identical copy of `fp16` (mirroring HF Optimum's
//! same-bytes-different-name convention), and the runtime loads it as fp16
//! without quantizing — see `model_arch::load_aesthetic_head`.
//!
//! Output: `<root>/<siglip-model>/burn/aesthetic.{fp32,fp16,fp16q8}.bpk`.
//!
//! Run:
//!   cargo run --release -p lumen-convert --bin export_aesthetic_head -- [models_dir]

use std::{fs, path::Path};

use burn::module::Module;
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use lumen_convert::server::aesthetic_head::{siglip2_base_patch16_224, siglip2_so400m_patch14_384};
use lumen_hub::backend::{Backend, Device, default_device};

const DEFAULT_ROOT: &str = "lumen-models";

/// (siglip model repo name, generated bpk path under OUT_DIR).
const HEADS: &[(&str, &str)] = &[
    (
        "siglip2-base-patch16-224",
        "aesthetic_head/siglip2_base_patch16_224/aesthetic.bpk",
    ),
    (
        "siglip2-so400m-patch14-384",
        "aesthetic_head/siglip2_so400m_patch14_384/aesthetic.bpk",
    ),
];

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ROOT.to_owned());

    for &(model, gen_rel) in HEADS {
        let burn_dir = format!("{root}/{model}/burn");
        fs::create_dir_all(&burn_dir).expect("create model burn dir");
        stage_generated_fp32(gen_rel, &burn_dir);
        match model {
            "siglip2-base-patch16-224" => {
                export_fp16::<siglip2_base_patch16_224::Model<Backend>, _>(
                    &burn_dir,
                    |path, dev| siglip2_base_patch16_224::Model::<Backend>::from_file(path, dev),
                )
            }
            "siglip2-so400m-patch14-384" => {
                export_fp16::<siglip2_so400m_patch14_384::Model<Backend>, _>(
                    &burn_dir,
                    |path, dev| siglip2_so400m_patch14_384::Model::<Backend>::from_file(path, dev),
                )
            }
            other => panic!("no aesthetic head architecture registered for `{other}`"),
        }
        println!("  {model}: aesthetic.{{fp32,fp16,fp16q8}}.bpk written to {burn_dir}");
    }
}

fn stage_generated_fp32(gen_rel: &str, burn_dir: &str) {
    let src = format!("{}/{}", env!("OUT_DIR"), gen_rel);
    let dst = format!("{burn_dir}/aesthetic.fp32.bpk");
    fs::copy(&src, &dst).unwrap_or_else(|e| panic!("stage {src} -> {dst}: {e}"));
}

fn export_fp16<M, L>(burn_dir: &str, load: L)
where
    M: Module<Backend> + ModuleSnapshot<Backend>,
    L: Fn(&str, &Device) -> M,
{
    let fp32_path = format!("{burn_dir}/aesthetic.fp32.bpk");
    let fp16_path = format!("{burn_dir}/aesthetic.fp16.bpk");
    let fp16q8_path = format!("{burn_dir}/aesthetic.fp16q8.bpk");

    if !Path::new(&fp32_path).is_file() {
        panic!("missing staged fp32 burnpack for aesthetic head: {fp32_path}");
    }

    let device = default_device();
    let model = load(&fp32_path, &device);
    let mut store = BurnpackStore::from_file(&fp16_path)
        .overwrite(true)
        .with_to_adapter(HalfPrecisionAdapter::new());
    model
        .save_into(&mut store)
        .unwrap_or_else(|e| panic!("save fp16 burnpack for aesthetic head: {e}"));
    // fp16q8 deliberately reuses the fp16 bytes; the head is never Q8-quantized.
    fs::copy(&fp16_path, &fp16q8_path)
        .unwrap_or_else(|e| panic!("copy {fp16_path} -> {fp16q8_path}: {e}"));
}
