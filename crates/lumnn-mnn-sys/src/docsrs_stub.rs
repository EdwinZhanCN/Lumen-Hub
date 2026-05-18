use ndarray::{ArrayD, ArrayViewD};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnnError {
    InvalidParameter(String),
    OutOfMemory,
    RuntimeError(String),
    Unsupported,
    ModelLoadFailed(String),
    NullPointer,
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for MnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MnnError {}

pub type Result<T> = std::result::Result<T, MnnError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum PrecisionMode {
    #[default]
    Normal = 0,
    Low = 1,
    High = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum DataFormat {
    #[default]
    NCHW = 0,
    NHWC = 1,
    Auto = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    CPU,
    Metal,
    OpenCL,
    OpenGL,
    Vulkan,
    CUDA,
    CoreML,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub thread_count: i32,
    pub precision_mode: PrecisionMode,
    pub use_cache: bool,
    pub data_format: DataFormat,
    pub backend: Backend,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            use_cache: false,
            data_format: DataFormat::NCHW,
            backend: Backend::CPU,
        }
    }
}

impl InferenceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }
}

pub struct SharedRuntime;

impl SharedRuntime {
    pub fn new(_config: &InferenceConfig) -> Result<Self> {
        Ok(Self)
    }
}

pub struct InferenceEngine {
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl InferenceEngine {
    pub fn from_buffer(_model_buffer: &[u8], _config: Option<InferenceConfig>) -> Result<Self> {
        Ok(Self::empty())
    }

    pub fn from_file(
        _model_path: impl AsRef<std::path::Path>,
        _config: Option<InferenceConfig>,
    ) -> Result<Self> {
        Ok(Self::empty())
    }

    pub fn from_buffer_with_runtime(
        _model_buffer: &[u8],
        _runtime: &SharedRuntime,
    ) -> Result<Self> {
        Ok(Self::empty())
    }

    fn empty() -> Self {
        Self {
            input_names: Vec::new(),
            output_names: Vec::new(),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        }
    }

    pub fn input_shape(&self) -> &[usize] {
        self.input_shapes.first().map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn output_shape(&self) -> &[usize] {
        self.output_shapes.first().map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn run(&self, _input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        Err(MnnError::Unsupported)
    }

    pub fn run_raw(&self, _input: &[f32], _output: &mut [f32]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn has_dynamic_shape(&self) -> bool {
        false
    }

    pub fn input_count(&self) -> usize {
        self.input_names.len()
    }

    pub fn output_count(&self) -> usize {
        self.output_names.len()
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    pub fn input_shape_at(&self, index: usize) -> Option<&[usize]> {
        self.input_shapes.get(index).map(Vec::as_slice)
    }

    pub fn output_shape_at(&self, index: usize) -> Option<&[usize]> {
        self.output_shapes.get(index).map(Vec::as_slice)
    }

    pub fn input_shape_by_name(&self, _name: &str) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }

    pub fn output_shape_by_name(&self, _name: &str) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }

    pub fn resize_input_by_name(&self, _name: &str, _shape: &[usize]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn input_dtype_at(&self, _index: usize) -> i32 {
        0
    }

    pub fn output_dtype_at(&self, _index: usize) -> i32 {
        0
    }

    pub fn backend_type(&self) -> i32 {
        0
    }

    pub fn input_name(&self) -> Option<String> {
        self.input_names.first().cloned()
    }

    pub fn output_name(&self) -> Option<String> {
        self.output_names.first().cloned()
    }

    pub fn run_multi(
        &self,
        _inputs: &[(&str, &[f32])],
        _outputs: &mut [(&str, &mut [f32])],
    ) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn copy_input_f32(&self, _name: &str, _data: &[f32]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn copy_input_i32(&self, _name: &str, _data: &[i32]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn copy_input_i64(&self, _name: &str, _data: &[i64]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn run_only(&self) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn copy_output_f32(&self, _name: &str, _data: &mut [f32]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn copy_output_i32(&self, _name: &str, _data: &mut [i32]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn copy_output_i64(&self, _name: &str, _data: &mut [i64]) -> Result<()> {
        Err(MnnError::Unsupported)
    }

    pub fn run_dynamic(&self, _input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        Err(MnnError::Unsupported)
    }

    pub fn run_dynamic_raw(
        &self,
        _input: &[f32],
        _input_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        Err(MnnError::Unsupported)
    }
}

pub struct SessionPool;

impl SessionPool {
    pub fn new(
        _engine: &InferenceEngine,
        _pool_size: usize,
        _config: Option<InferenceConfig>,
    ) -> Result<Self> {
        Ok(Self)
    }

    pub fn run(&self, _input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        Err(MnnError::Unsupported)
    }

    pub fn available(&self) -> usize {
        0
    }
}

pub fn get_version() -> String {
    "docsrs-stub".to_owned()
}
