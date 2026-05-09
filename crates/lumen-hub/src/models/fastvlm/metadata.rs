/// Hardcoded FastVLM model constants.
///
/// All values are specific to the FastVLM-0.5B model and do not vary at runtime.

#[derive(Debug, Clone)]
pub struct FastVlmMetadata {
    pub hidden_size: usize,
    pub eos_token_id: u32,
    pub image_token_id: u32,
    pub model_max_length: usize,
    pub add_special_tokens_after_template: bool,
    pub vision_input_name: &'static str,
    pub vision_output_name: &'static str,
    pub embed_input_name: &'static str,
    pub embed_output_name: &'static str,
    pub decoder_output_name: &'static str,
    pub vision_preprocess: FastVlmPreprocessMetadata,
    pub kv_cache: FastVlmKvCacheMetadata,
    pub generation_defaults: FastVlmGenerationDefaults,
}

#[derive(Debug, Clone, Copy)]
pub struct FastVlmGenerationDefaults {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FastVlmKvCacheMetadata {
    pub num_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub initial_past_sequence_length: usize,
}

#[derive(Debug, Clone)]
pub struct FastVlmPreprocessMetadata {
    pub resize_longest_edge: u32,
    pub pad_to: FastVlmPadSize,
    pub pad_color_rgb: [u8; 3],
    pub do_resize: bool,
    pub do_pad: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub resample: FastVlmResizeFilter,
    pub color_space: &'static str,
    pub layout: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct FastVlmPadSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum FastVlmResizeFilter {
    Nearest,
    Lanczos3,
    Bilinear,
    Bicubic,
}

pub const METADATA: FastVlmMetadata = FastVlmMetadata {
    hidden_size: 896,
    eos_token_id: 151645,
    image_token_id: 151646,
    model_max_length: 8192,
    add_special_tokens_after_template: false,
    vision_input_name: "pixel_values",
    vision_output_name: "image_features",
    embed_input_name: "input_ids",
    embed_output_name: "inputs_embeds",
    decoder_output_name: "logits",
    vision_preprocess: FastVlmPreprocessMetadata {
        resize_longest_edge: 448,
        pad_to: FastVlmPadSize {
            width: 448,
            height: 448,
        },
        pad_color_rgb: [128, 128, 128],
        do_resize: true,
        do_pad: true,
        do_rescale: true,
        do_normalize: true,
        rescale_factor: 0.00392156862745098,
        image_mean: [0.0, 0.0, 0.0],
        image_std: [1.0, 1.0, 1.0],
        resample: FastVlmResizeFilter::Bicubic,
        color_space: "rgb",
        layout: "nchw",
    },
    kv_cache: FastVlmKvCacheMetadata {
        num_layers: 24,
        num_key_value_heads: 2,
        head_dim: 64,
        initial_past_sequence_length: 0,
    },
    generation_defaults: FastVlmGenerationDefaults {
        max_new_tokens: 128,
        temperature: 0.7,
        top_p: 0.9,
    },
};
