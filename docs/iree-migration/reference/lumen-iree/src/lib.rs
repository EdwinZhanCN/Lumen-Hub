//! Safe, inference-only wrapper over the Lumen IREE shim.
//!
//! * [`Runtime`]: one IREE instance + one HAL device (thread-safe, shared via `Arc`).
//! * [`Model`]: one compiled module (`.vmfb`) + its parameter archive (`.irpa`).
//!   Invocations are serialized by an internal mutex; a `Model` keeps its
//!   `Runtime` alive.
//! * [`TensorRef`] in, [`Tensor`] out: dense row-major `f32`, `i64` or `i32`.

use std::ffi::{CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};

use lumen_iree_sys as sys;

/// Parameter scope used by every Lumen artifact (tools/iree constants.PARAM_SCOPE).
pub const PARAM_SCOPE: &str = "model";
/// Maximum tensor rank accepted by the shim.
pub const MAX_RANK: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Converts a shim error pointer into a `Result`, releasing the error.
fn check(error: *mut sys::lumen_iree_error_t) -> Result<()> {
    if error.is_null() {
        return Ok(());
    }
    // SAFETY: a non-null error was returned by the shim, is valid until freed,
    // and is freed exactly once here.
    let message = unsafe { CStr::from_ptr(sys::lumen_iree_error_message(error)) }
        .to_string_lossy()
        .into_owned();
    unsafe { sys::lumen_iree_error_free(error) };
    Err(Error::new(message))
}

fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| Error::new(format!("string contains NUL: {value:?}")))
}

fn c_path(path: &Path) -> Result<CString> {
    let text = path
        .to_str()
        .ok_or_else(|| Error::new(format!("path is not UTF-8: {}", path.display())))?;
    c_string(text)
}

/// IREE HAL driver compiled into lumen-iree-sys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driver {
    /// Multi-threaded CPU (`local-task`); always available.
    LocalTask,
    /// NVIDIA CUDA (feature `driver-cuda`).
    Cuda,
    /// Apple Metal (feature `driver-metal`).
    Metal,
}

impl Driver {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalTask => "local-task",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

/// One IREE instance and one device.
pub struct Runtime {
    raw: NonNull<sys::lumen_iree_runtime_t>,
    driver: Driver,
}

// SAFETY: IREE instances and HAL devices are internally synchronized.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn new(driver: Driver) -> Result<Arc<Self>> {
        let name = c_string(driver.as_str())?;
        let mut raw = ptr::null_mut();
        // SAFETY: valid NUL-terminated string and out-pointer.
        check(unsafe { sys::lumen_iree_runtime_create(name.as_ptr(), &mut raw) })?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::new("shim returned a null runtime"))?;
        Ok(Arc::new(Self { raw, driver }))
    }

    pub fn driver(&self) -> Driver {
        self.driver
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: every Model holds an Arc<Runtime>, so no model outlives this.
        unsafe { sys::lumen_iree_runtime_release(self.raw.as_ptr()) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsMode {
    /// Memory-map the .irpa (zero-copy on the CPU device). Used by lumen-hub.
    Mmap,
    /// Read parameters into device buffers at load time.
    Read,
}

struct RawModel(NonNull<sys::lumen_iree_model_t>);

// SAFETY: the session is only accessed while holding Model::inner.
unsafe impl Send for RawModel {}

/// A loaded module with its parameters.
pub struct Model {
    inner: Mutex<RawModel>,
    _runtime: Arc<Runtime>,
}

impl Model {
    pub fn load(
        runtime: &Arc<Runtime>,
        vmfb: &Path,
        irpa: &Path,
        params: ParamsMode,
    ) -> Result<Self> {
        let vmfb_c = c_path(vmfb)?;
        let irpa_c = c_path(irpa)?;
        let scope = c_string(PARAM_SCOPE)?;
        let mode = match params {
            ParamsMode::Mmap => sys::LUMEN_IREE_PARAMS_MMAP,
            ParamsMode::Read => sys::LUMEN_IREE_PARAMS_READ,
        };
        let mut raw = ptr::null_mut();
        // SAFETY: valid runtime handle, strings and out-pointer.
        check(unsafe {
            sys::lumen_iree_model_load(
                runtime.raw.as_ptr(),
                vmfb_c.as_ptr(),
                irpa_c.as_ptr(),
                scope.as_ptr(),
                mode,
                &mut raw,
            )
        })
        .map_err(|error| Error::new(format!("loading {}: {error}", vmfb.display())))?;
        let raw = NonNull::new(raw).ok_or_else(|| Error::new("shim returned a null model"))?;
        Ok(Self {
            inner: Mutex::new(RawModel(raw)),
            _runtime: Arc::clone(runtime),
        })
    }

    pub fn has_function(&self, function: &str) -> bool {
        let Ok(name) = c_string(function) else {
            return false;
        };
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: valid model handle (lock held) and string.
        unsafe { sys::lumen_iree_model_has_function(guard.0.as_ptr(), name.as_ptr()) == 1 }
    }

    /// Runs `function` synchronously and returns every result.
    pub fn invoke(&self, function: &str, inputs: &[TensorRef<'_>]) -> Result<Vec<Tensor>> {
        let name = c_string(function)?;
        let raw_inputs = inputs
            .iter()
            .map(|tensor| tensor.to_raw())
            .collect::<Result<Vec<_>>>()?;
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut outputs = ptr::null_mut();
        // SAFETY: valid model handle (lock held); raw_inputs borrow `inputs`,
        // which outlive the call.
        check(unsafe {
            sys::lumen_iree_model_invoke(
                guard.0.as_ptr(),
                name.as_ptr(),
                raw_inputs.as_ptr(),
                raw_inputs.len(),
                &mut outputs,
            )
        })
        .map_err(|error| Error::new(format!("invoking {function}: {error}")))?;
        let outputs = Outputs(outputs);
        // SAFETY: valid outputs handle owned by `outputs`.
        let count = unsafe { sys::lumen_iree_outputs_count(outputs.0) };
        (0..count).map(|index| outputs.read(index)).collect()
    }

    /// Releases cached device memory held by this model's session.
    pub fn trim(&self) -> Result<()> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: valid model handle (lock held).
        check(unsafe { sys::lumen_iree_model_trim(guard.0.as_ptr()) })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let raw = self
            .inner
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: last use of the handle; the runtime is still alive (Arc).
        unsafe { sys::lumen_iree_model_release(raw.0.as_ptr()) }
    }
}

struct Outputs(*mut sys::lumen_iree_outputs_t);

impl Outputs {
    fn read(&self, index: usize) -> Result<Tensor> {
        let (mut element_type, mut rank, mut bytes) = (0i32, 0usize, 0usize);
        // SAFETY: valid outputs handle; dims == NULL queries the rank only.
        check(unsafe {
            sys::lumen_iree_outputs_describe(
                self.0,
                index,
                &mut element_type,
                &mut rank,
                ptr::null_mut(),
                0,
                &mut bytes,
            )
        })?;
        let mut dims = vec![0i64; rank];
        // SAFETY: dims has `rank` entries.
        check(unsafe {
            sys::lumen_iree_outputs_describe(
                self.0,
                index,
                &mut element_type,
                &mut rank,
                dims.as_mut_ptr(),
                dims.len(),
                &mut bytes,
            )
        })?;
        let data = match element_type {
            sys::LUMEN_IREE_ELEMENT_F32 => TensorData::F32(self.copy(index, bytes)?),
            sys::LUMEN_IREE_ELEMENT_I64 => TensorData::I64(self.copy(index, bytes)?),
            sys::LUMEN_IREE_ELEMENT_I32 => TensorData::I32(self.copy(index, bytes)?),
            other => {
                return Err(Error::new(format!(
                    "unsupported output element type {other}"
                )));
            }
        };
        Ok(Tensor { dims, data })
    }

    fn copy<T: Copy + Default>(&self, index: usize, bytes: usize) -> Result<Vec<T>> {
        let size = std::mem::size_of::<T>();
        if !bytes.is_multiple_of(size) {
            return Err(Error::new(format!(
                "output byte length {bytes} is not a multiple of {size}"
            )));
        }
        let mut values = vec![T::default(); bytes / size];
        // SAFETY: `values` owns exactly `bytes` bytes of plain-old-data.
        check(unsafe {
            sys::lumen_iree_outputs_read(self.0, index, values.as_mut_ptr().cast(), bytes)
        })?;
        Ok(values)
    }
}

impl Drop for Outputs {
    fn drop(&mut self) {
        // SAFETY: owned handle released once (NULL-safe in the shim).
        unsafe { sys::lumen_iree_outputs_release(self.0) }
    }
}

/// Borrowed input tensor.
#[derive(Debug, Clone, Copy)]
pub enum TensorRef<'a> {
    F32 { dims: &'a [i64], data: &'a [f32] },
    I64 { dims: &'a [i64], data: &'a [i64] },
    I32 { dims: &'a [i64], data: &'a [i32] },
}

impl TensorRef<'_> {
    fn to_raw(self) -> Result<sys::lumen_iree_tensor_t> {
        let (element_type, dims, data, len, byte_length) = match self {
            TensorRef::F32 { dims, data } => (
                sys::LUMEN_IREE_ELEMENT_F32,
                dims,
                data.as_ptr().cast(),
                data.len(),
                size_of_val(data),
            ),
            TensorRef::I64 { dims, data } => (
                sys::LUMEN_IREE_ELEMENT_I64,
                dims,
                data.as_ptr().cast(),
                data.len(),
                size_of_val(data),
            ),
            TensorRef::I32 { dims, data } => (
                sys::LUMEN_IREE_ELEMENT_I32,
                dims,
                data.as_ptr().cast(),
                data.len(),
                size_of_val(data),
            ),
        };
        if dims.len() > MAX_RANK {
            return Err(Error::new(format!(
                "rank {} exceeds {MAX_RANK}",
                dims.len()
            )));
        }
        let mut count: usize = 1;
        for &dim in dims {
            let dim = usize::try_from(dim)
                .map_err(|_| Error::new(format!("negative dimension in {dims:?}")))?;
            count = count
                .checked_mul(dim)
                .ok_or_else(|| Error::new(format!("shape {dims:?} overflows")))?;
        }
        if count != len {
            return Err(Error::new(format!(
                "shape {dims:?} needs {count} elements, got {len}"
            )));
        }
        Ok(sys::lumen_iree_tensor_t {
            element_type,
            rank: dims.len(),
            dims: dims.as_ptr(),
            data,
            byte_length,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

/// Owned output tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub dims: Vec<i64>,
    pub data: TensorData,
}

impl Tensor {
    /// Returns the `f32` payload or an error naming the actual element type.
    pub fn into_f32(self) -> Result<Vec<f32>> {
        match self.data {
            TensorData::F32(values) => Ok(values),
            other => Err(Error::new(format!("expected an f32 output, got {other:?}"))),
        }
    }
}
