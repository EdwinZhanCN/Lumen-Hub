//! Real-image end-to-end check for pp-ocrv5-server recognition int8.
//!
//! Decodes a text-line crop with both fp32 and int8 recognition and compares the
//! strings — the meaningful metric for a CTC recognizer (vs the misleading cosine
//! on raw logits). The default image is the project's "BORDER" OCR warmup crop.
//!
//! Run:
//!   cargo run --release -p lumen-convert --bin validate_ocr
//!   cargo run --release -p lumen-convert --bin validate_ocr -- <image.png> <expected> <model_dir>

use burn::tensor::{Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};
use image::imageops::FilterType;

use lumen_convert::server::{detection, recognition};
use lumen_hub::backend::{Backend, Device, default_device};

const DEFAULT_REPO: &str = "lumen-models/pp-ocrv5-server";
// PP-OCR recognition preprocessing (matches lumen-hub ppocr task).
const REC_H: u32 = 48;
const REC_W: u32 = 320;
const SCALE: f32 = 1.0 / 255.0;
const MEAN: f32 = 0.5;
const STD: f32 = 0.5;
const BLANK_ID: i64 = 0;

/// Resize keeping aspect ratio to height 48, pad width to 320, normalize → NCHW.
fn rec_preprocess(img: &image::RgbImage) -> Vec<f32> {
    let ratio = REC_H as f32 / img.height() as f32;
    let rw = ((img.width() as f32 * ratio).ceil() as u32).clamp(1, REC_W);
    let resized = image::imageops::resize(img, rw, REC_H, FilterType::CatmullRom);
    let (rh, tw, rw) = (REC_H as usize, REC_W as usize, rw as usize);
    let pad = (0.0 * SCALE - MEAN) / STD;
    let mut t = vec![pad; 3 * rh * tw];
    for y in 0..rh {
        for x in 0..rw {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                t[c * rh * tw + y * tw + x] = (px[c] as f32 * SCALE - MEAN) / STD;
            }
        }
    }
    t
}

fn ctc_decode(logits: &[f32], seq: usize, classes: usize) -> (Vec<i64>, f32) {
    let (mut idxs, mut confs, mut prev) = (Vec::new(), Vec::<f32>::new(), -1i64);
    for t in 0..seq {
        let s = &logits[t * classes..t * classes + classes];
        let (mut mi, mut mv) = (0usize, f32::MIN);
        for (i, &v) in s.iter().enumerate() {
            if v > mv {
                mv = v;
                mi = i;
            }
        }
        let id = mi as i64;
        if id == BLANK_ID {
            prev = -1;
            continue;
        }
        if id == prev {
            continue;
        }
        idxs.push(id);
        confs.push(mv);
        prev = id;
    }
    let conf = if confs.is_empty() {
        0.0
    } else {
        confs.iter().sum::<f32>() / confs.len() as f32
    };
    (idxs, conf)
}

fn to_text(idxs: &[i64], vocab: &[String]) -> String {
    idxs.iter()
        .filter_map(|&i| {
            if i <= BLANK_ID {
                return None;
            }
            vocab.get((i - 1) as usize).map(String::as_str)
        })
        .collect()
}

fn decode(
    model: &recognition::Model<Backend>,
    input: &[f32],
    device: &Device,
    vocab: &[String],
) -> (String, f32) {
    let x = Tensor::<Backend, 4>::from_data(
        TensorData::new(input.to_vec(), [1, 3, REC_H as usize, REC_W as usize]),
        device,
    );
    let out = model.forward(x);
    let [_, seq, classes] = out.dims();
    let logits = out.into_data().convert::<f32>().into_vec::<f32>().unwrap();
    let (idxs, conf) = ctc_decode(&logits, seq, classes);
    (to_text(&idxs, vocab), conf)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let img_path = args
        .next()
        .unwrap_or_else(|| "crates/lumen-hub/warmup/ocr/border.png".into());
    let expected = args.next().unwrap_or_else(|| "BORDER".into());
    let repo = args.next().unwrap_or_else(|| DEFAULT_REPO.into());
    let device = default_device();

    let vocab: Vec<String> = std::fs::read_to_string(format!("{repo}/ppocrv5_dict.txt"))
        .expect("dict")
        .lines()
        .map(str::to_string)
        .collect();
    let img = image::open(&img_path).expect("open image").to_rgb8();
    let input = rec_preprocess(&img);
    println!(
        "image {} ({}x{}), expected {expected:?}\n",
        img_path,
        img.width(),
        img.height()
    );

    let fp32 = recognition::Model::<Backend>::from_file(
        format!("{repo}/burn/recognition.fp32.bpk"),
        &device,
    );
    let mut int8 = recognition::Model::<Backend>::new(&device);
    int8.load_from(&mut BurnpackStore::from_file(format!(
        "{repo}/burn/recognition.int8.bpk"
    )))
    .expect("load int8");

    let (t32, c32) = decode(&fp32, &input, &device, &vocab);
    let (t8, c8) = decode(&int8, &input, &device, &vocab);

    println!("fp32: {t32:?}  (conf {c32:.3})");
    println!("int8: {t8:?}  (conf {c8:.3})");
    let ok32 = t32.eq_ignore_ascii_case(&expected);
    let ok8 = t8.eq_ignore_ascii_case(&expected);
    println!(
        "\nfp32 == expected: {ok32} | int8 == expected: {ok8} | int8 == fp32: {}",
        t8 == t32
    );

    // Detection on the same real image: int8-vs-fp32 heatmap cosine. On a real
    // text image (real signal) this should be high — unlike the 0.756 we saw on
    // synthetic noise, where the heatmap was ≈0 and cosine was meaningless.
    let (dh, dw) = (160usize, 640usize); // 149->160, 640 (both mult of 32)
    let det_mean = [0.485f32, 0.456, 0.406];
    let det_std = [0.229f32, 0.224, 0.225];
    let resized = image::imageops::resize(&img, dw as u32, dh as u32, FilterType::CatmullRom);
    let mut din = vec![0f32; 3 * dh * dw];
    for y in 0..dh {
        for x in 0..dw {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                din[c * dh * dw + y * dw + x] = (px[c] as f32 * SCALE - det_mean[c]) / det_std[c];
            }
        }
    }
    let det_embed = |m: &detection::Model<Backend>| -> Vec<f32> {
        let x =
            Tensor::<Backend, 4>::from_data(TensorData::new(din.clone(), [1, 3, dh, dw]), &device);
        m.forward(x)
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()
    };
    let det32 =
        detection::Model::<Backend>::from_file(format!("{repo}/burn/detection.fp32.bpk"), &device);
    let mut det8 = detection::Model::<Backend>::new(&device);
    det8.load_from(&mut BurnpackStore::from_file(format!(
        "{repo}/burn/detection.int8.bpk"
    )))
    .expect("load det int8");
    let (h32, h8) = (det_embed(&det32), det_embed(&det8));
    let maxp = h32.iter().cloned().fold(0f32, f32::max);
    println!(
        "\ndetection on real image: heatmap max(fp32)={maxp:.3} | int8-vs-fp32 cosine={:.5}",
        lumen_convert::cosine(&h32, &h8)
    );
}
