//! CLIP/BioCLIP image preprocessing (resize → optional center-crop → rescale →
//! normalize → NCHW), driven entirely by `model_info.json`
//! `task_metadata.tasks.<task>.preprocess`.

use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use serde::{Deserialize, Deserializer, de};

use crate::service::{ServiceError, ServiceResult};

#[derive(Debug, Clone)]
pub struct ClipImagePreprocessConfig {
    resize_shortest_edge: u32,
    crop_width: u32,
    crop_height: u32,
    do_resize: bool,
    do_center_crop: bool,
    do_rescale: bool,
    do_normalize: bool,
    rescale_factor: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    filter: FilterType,
    color_space: ClipImageColorSpace,
    layout: ClipTensorLayout,
}

impl ClipImagePreprocessConfig {
    pub fn from_json_str(input: &str) -> Result<Self, String> {
        serde_json::from_str(input)
            .map_err(|err| format!("failed to parse image preprocess metadata: {err}"))
    }

    pub fn output_shape(&self) -> Vec<usize> {
        debug_assert!(matches!(self.layout, ClipTensorLayout::Nchw));
        vec![1, 3, self.crop_height as usize, self.crop_width as usize]
    }

    pub fn preprocess_image_bytes(&self, bytes: &[u8]) -> ServiceResult<Vec<f32>> {
        debug_assert!(matches!(self.color_space, ClipImageColorSpace::Rgb));
        let image = image::load_from_memory(bytes).map_err(|err| {
            ServiceError::InvalidArgument(format!("failed to decode image: {err}"))
        })?;
        let mut rgb = image.to_rgb8();

        if self.do_resize {
            rgb = resize_shortest_edge(&rgb, self.resize_shortest_edge, self.filter);
        }

        if self.do_center_crop {
            rgb = center_crop(&rgb, self.crop_width, self.crop_height, self.filter);
        } else if rgb.width() != self.crop_width || rgb.height() != self.crop_height {
            rgb = imageops::resize(&rgb, self.crop_width, self.crop_height, self.filter);
        }

        Ok(rgb_to_nchw_normalized(self, &rgb))
    }

    fn from_raw(raw: RawImagePreprocessConfig) -> Result<Self, String> {
        if raw.resize_shortest_edge == 0 {
            return Err("`resize_shortest_edge` must be greater than zero".to_owned());
        }
        if raw.crop_size.width == 0 || raw.crop_size.height == 0 {
            return Err(
                "`crop_size.width` and `crop_size.height` must be greater than zero".to_owned(),
            );
        }
        if !raw.rescale_factor.is_finite() {
            return Err("`rescale_factor` must be finite".to_owned());
        }

        Ok(Self {
            resize_shortest_edge: raw.resize_shortest_edge,
            crop_width: raw.crop_size.width,
            crop_height: raw.crop_size.height,
            do_resize: raw.do_resize,
            do_center_crop: raw.do_center_crop,
            do_rescale: raw.do_rescale,
            do_normalize: raw.do_normalize,
            rescale_factor: raw.rescale_factor,
            image_mean: vec3(raw.image_mean, "image_mean")?,
            image_std: nonzero_vec3(raw.image_std, "image_std")?,
            filter: raw.resample.into_filter_type(),
            color_space: raw.color_space,
            layout: raw.layout,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawImagePreprocessConfig {
    resize_shortest_edge: u32,
    crop_size: CropSize,
    do_resize: bool,
    do_center_crop: bool,
    do_rescale: bool,
    do_normalize: bool,
    rescale_factor: f32,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    resample: ResizeFilter,
    color_space: ClipImageColorSpace,
    layout: ClipTensorLayout,
}

#[derive(Debug, Deserialize)]
struct CropSize {
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Copy)]
enum ResizeFilter {
    Nearest,
    Lanczos3,
    Bilinear,
    Bicubic,
}

impl ResizeFilter {
    fn into_filter_type(self) -> FilterType {
        match self {
            ResizeFilter::Nearest => FilterType::Nearest,
            ResizeFilter::Lanczos3 => FilterType::Lanczos3,
            ResizeFilter::Bilinear => FilterType::Triangle,
            ResizeFilter::Bicubic => FilterType::CatmullRom,
        }
    }
}

impl<'de> Deserialize<'de> for ResizeFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let normalized = match value {
            serde_json::Value::String(value) => value.trim().to_ascii_lowercase(),
            serde_json::Value::Number(value) => value.to_string(),
            other => {
                return Err(de::Error::custom(format!(
                    "`resample` must be a string or integer, got {other}"
                )));
            }
        };

        match normalized.as_str() {
            "nearest" | "0" => Ok(Self::Nearest),
            "lanczos" | "lanczos3" | "1" => Ok(Self::Lanczos3),
            "bilinear" | "triangle" | "2" => Ok(Self::Bilinear),
            "bicubic" | "catmull_rom" | "catmullrom" | "3" => Ok(Self::Bicubic),
            other => Err(de::Error::custom(format!(
                "unsupported `resample` value `{other}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ClipImageColorSpace {
    Rgb,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ClipTensorLayout {
    Nchw,
}

impl<'de> Deserialize<'de> for ClipImagePreprocessConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawImagePreprocessConfig::deserialize(deserializer)?;
        Self::from_raw(raw).map_err(de::Error::custom)
    }
}

fn vec3(values: Vec<f32>, field_name: &str) -> Result<[f32; 3], String> {
    if values.len() != 3 {
        return Err(format!(
            "`{field_name}` must contain exactly 3 values, got {}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("`{field_name}` values must be finite"));
    }
    Ok([values[0], values[1], values[2]])
}

fn nonzero_vec3(values: Vec<f32>, field_name: &str) -> Result<[f32; 3], String> {
    let values = vec3(values, field_name)?;
    if values.contains(&0.0) {
        return Err(format!("`{field_name}` values must be non-zero"));
    }
    Ok(values)
}

fn resize_shortest_edge(image: &RgbImage, shortest_edge: u32, filter: FilterType) -> RgbImage {
    let (width, height) = image.dimensions();
    let shortest = width.min(height);
    if shortest == shortest_edge {
        return image.clone();
    }

    let scale = shortest_edge as f32 / shortest as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(1);
    let resized_height = ((height as f32 * scale).round() as u32).max(1);
    imageops::resize(image, resized_width, resized_height, filter)
}

fn center_crop(
    image: &RgbImage,
    crop_width: u32,
    crop_height: u32,
    filter: FilterType,
) -> RgbImage {
    let image = if image.width() < crop_width || image.height() < crop_height {
        imageops::resize(image, crop_width, crop_height, filter)
    } else {
        image.clone()
    };

    let x = image.width().saturating_sub(crop_width) / 2;
    let y = image.height().saturating_sub(crop_height) / 2;
    imageops::crop_imm(&image, x, y, crop_width, crop_height).to_image()
}

fn rgb_to_nchw_normalized(config: &ClipImagePreprocessConfig, image: &RgbImage) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let plane = width * height;
    let mut values = vec![0.0; 3 * plane];

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                let mut value = pixel[channel] as f32;
                if config.do_rescale {
                    value *= config.rescale_factor;
                }
                if config.do_normalize {
                    value = (value - config.image_mean[channel]) / config.image_std[channel];
                }
                values[channel * plane + y * width + x] = value;
            }
        }
    }

    values
}
