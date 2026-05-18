use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct InsightFacePackSpec {
    pub detection: InsightFaceDetectionSpec,
    pub recognition: InsightFaceRecognitionSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InsightFaceDetectionSpec {
    #[serde(rename = "type")]
    pub detector_type: String,
    pub input_size: [usize; 2],
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub letterbox: bool,
    pub normalized_boxes: bool,
    pub strides: Vec<usize>,
    pub outputs: Vec<ScrfdOutputSpec>,
    pub score_threshold: f32,
    pub nms_threshold: f32,
    pub min_face: f32,
    pub max_face: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScrfdOutputSpec {
    pub stride: usize,
    pub score: usize,
    pub bbox: usize,
    pub kps: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InsightFaceRecognitionSpec {
    pub input_size: [usize; 2],
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub channels_last: bool,
    pub color_order: String,
    pub align_landmarks: bool,
    pub embedding_dim: usize,
}

pub fn pack_spec(name: &str) -> Option<InsightFacePackSpec> {
    match name {
        "antelopev2" | "buffalo_l" | "buffalo_m" | "buffalo_s" | "buffalo_sc" => {
            Some(shared_scrfd_arcface_spec())
        }
        _ => None,
    }
}

fn shared_scrfd_arcface_spec() -> InsightFacePackSpec {
    InsightFacePackSpec {
        detection: InsightFaceDetectionSpec {
            detector_type: "scrfd".to_owned(),
            input_size: [640, 640],
            mean: [127.5, 127.5, 127.5],
            std: [128.0, 128.0, 128.0],
            letterbox: true,
            normalized_boxes: false,
            strides: vec![8, 16, 32],
            outputs: vec![
                ScrfdOutputSpec {
                    stride: 8,
                    score: 0,
                    bbox: 3,
                    kps: 6,
                },
                ScrfdOutputSpec {
                    stride: 16,
                    score: 1,
                    bbox: 4,
                    kps: 7,
                },
                ScrfdOutputSpec {
                    stride: 32,
                    score: 2,
                    bbox: 5,
                    kps: 8,
                },
            ],
            score_threshold: 0.4,
            nms_threshold: 0.4,
            min_face: 32.0,
            max_face: 1000.0,
        },
        recognition: InsightFaceRecognitionSpec {
            input_size: [112, 112],
            mean: [127.5, 127.5, 127.5],
            std: [127.5, 127.5, 127.5],
            channels_last: false,
            color_order: "rgb".to_owned(),
            align_landmarks: true,
            embedding_dim: 512,
        },
    }
}
