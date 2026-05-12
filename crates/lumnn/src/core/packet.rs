use crate::core::context::MLContext;
use half::f16;
use std::{any::Any, sync::Arc};
use tokio::sync::Mutex;

/// Elementary data types for tensor elements.
///
/// Each variant maps to a known Rust type and has a fixed
/// [`byte_size`](MLPacketDataType::byte_size).
///
/// # Example
///
/// ```
/// use lumnn::core::packet::MLPacketDataType;
/// assert_eq!(MLPacketDataType::Float32.byte_size(), 4);
/// assert_eq!(MLPacketDataType::Int8.byte_size(),   1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLPacketDataType {
    Float32,
    Float16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Int8,
    Uint8,
}

impl MLPacketDataType {
    /// Size in bytes of a single element of this type.
    pub fn byte_size(&self) -> usize {
        match self {
            MLPacketDataType::Float32 | MLPacketDataType::Int32 | MLPacketDataType::Uint32 => 4,
            MLPacketDataType::Float16 => 2,
            MLPacketDataType::Int64 | MLPacketDataType::Uint64 => 8,
            MLPacketDataType::Int8 | MLPacketDataType::Uint8 => 1,
        }
    }
}

/// Describes the data type, shape, and axis variability of an [`MLPacket`].
///
/// A descriptor serves two roles:
///
/// * **Contract** — a node declares what it accepts/produces via
///   [`input_descriptors`](crate::core::node::MLNode::input_descriptors).
///   Dynamic axes (batch, sequence length) are declared here.
/// * **Metadata** — a concrete packet carries a fully-resolved descriptor
///   so downstream consumers can validate compatibility.
///
/// Use [`validate_compatibility`](MLPacketDescriptor::validate_compatibility)
/// to check that a runtime descriptor satisfies a contract.
///
/// # Dynamic axes
///
/// When a dimension can vary at runtime (e.g., batch size or KV-cache length),
/// mark the descriptor with [`with_dynamic_batch`](MLPacketDescriptor::with_dynamic_batch)
/// or [`with_dynamic_axis`](MLPacketDescriptor::with_dynamic_axis).
/// The contract carries dynamic markers; the runtime descriptor must supply
/// concrete values.
///
/// # Examples
///
/// ```
/// use lumnn::core::packet::{MLPacketDescriptor, MLPacketDataType};
///
/// // Fixed 224×224 RGB image with batch=1
/// let fixed = MLPacketDescriptor::new(
///     MLPacketDataType::Float32,
///     vec![1, 3, 224, 224],
/// );
/// assert_eq!(fixed.element_count(), 150528);
///
/// // Variable batch size
/// let dyn_batch = MLPacketDescriptor::new(
///     MLPacketDataType::Float32,
///     vec![1, 3, 224, 224],
/// ).with_dynamic_batch();
///
/// // Validate a concrete batch of 4
/// let concrete = MLPacketDescriptor::new(
///     MLPacketDataType::Float32,
///     vec![4, 3, 224, 224],
/// );
/// dyn_batch.validate_compatibility(&concrete, "images").unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MLPacketDescriptor {
    /// Data type of each element.
    pub dtype: MLPacketDataType,
    /// Shape with concrete dims (contract may use placeholder for dynamic axes).
    pub shape: Vec<usize>,
    /// Whether axis 0 (batch) may vary.
    pub dynamic_batch: bool,
    /// Per-axis dynamic markers; length equals `shape.len()`.
    pub dynamic_axes: Vec<bool>,
}

impl MLPacketDescriptor {
    /// Creates a descriptor with all axes fixed.
    pub fn new(dtype: MLPacketDataType, shape: Vec<usize>) -> Self {
        let dynamic_axes = vec![false; shape.len()];
        Self {
            dtype,
            shape,
            dynamic_batch: false,
            dynamic_axes,
        }
    }

    /// Marks axis 0 (batch) as dynamic.
    pub fn with_dynamic_batch(mut self) -> Self {
        self.dynamic_batch = true;
        if !self.dynamic_axes.is_empty() {
            self.dynamic_axes[0] = true;
        }
        self
    }

    /// Marks a specific axis as dynamic.
    ///
    /// If `axis` is 0, this also sets `dynamic_batch = true`.
    pub fn with_dynamic_axis(mut self, axis: usize) -> Self {
        if axis >= self.dynamic_axes.len() {
            self.dynamic_axes.resize(self.shape.len(), false);
        }
        if axis < self.dynamic_axes.len() {
            self.dynamic_axes[axis] = true;
        }
        if axis == 0 {
            self.dynamic_batch = true;
        }
        self
    }

    /// Total number of elements: `shape.iter().product()`.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte length: `element_count × dtype.byte_size()`.
    pub fn byte_length(&self) -> usize {
        self.element_count() * self.dtype.byte_size()
    }

    /// Validates that `actual` satisfies `self` (the contract).
    ///
    /// The contract (`self`) may declare dynamic axes; the runtime descriptor
    /// (`actual`) must be fully concrete. Validation checks dtype, rank, and
    /// shape on fixed axes. For dynamic axes, only a zero batch is rejected.
    ///
    /// # Errors
    ///
    /// Returns `Err` on dtype mismatch, rank mismatch, shape mismatch on fixed
    /// axes, zero-sized dynamic batch, or if `actual` still has dynamic markers.
    pub fn validate_compatibility(&self, actual: &Self, packet_name: &str) -> Result<(), String> {
        if actual.dtype != self.dtype {
            return Err(format!(
                "input `{packet_name}` dtype mismatch: expected {:?}, got {:?}",
                self.dtype, actual.dtype
            ));
        }

        if actual.dynamic_batch {
            return Err(format!(
                "input `{packet_name}` actual descriptor must use a concrete batch size"
            ));
        }
        if actual.dynamic_axes.iter().any(|is_dynamic| *is_dynamic) {
            return Err(format!(
                "input `{packet_name}` actual descriptor must use concrete dimensions"
            ));
        }

        if self.shape.len() != actual.shape.len() {
            return Err(format!(
                "input `{packet_name}` rank mismatch: expected {}, got {}",
                self.shape.len(),
                actual.shape.len()
            ));
        }

        if self.dynamic_batch && self.shape.is_empty() {
            return Err(format!(
                "input `{packet_name}` dynamic batch requires a non-scalar tensor"
            ));
        }

        for (axis, (&expected_dim, &actual_dim)) in self.shape.iter().zip(&actual.shape).enumerate()
        {
            let dynamic_axis = self.dynamic_axes.get(axis).copied().unwrap_or(false)
                || (self.dynamic_batch && axis == 0);

            if dynamic_axis {
                if axis == 0 && actual_dim == 0 {
                    return Err(format!(
                        "input `{packet_name}` dynamic batch size at axis {axis} must be greater than zero"
                    ));
                }
                continue;
            }

            if expected_dim != actual_dim {
                return Err(format!(
                    "input `{packet_name}` shape mismatch: expected {:?}, got {:?}",
                    self.shape, actual.shape
                ));
            }
        }

        Ok(())
    }
}

/// Where a packet's payload currently resides.
///
/// Nodes use this to decide whether to take a fast path (data already on the
/// right device) or a slow path (transfer needed).
///
/// # Example
///
/// ```
/// use lumnn::core::packet::RuntimeType;
///
/// let cpu = RuntimeType::Cpu;
/// let cuda = RuntimeType::backend("cuda", "cuda:0");
///
/// match cpu {
///     RuntimeType::Cpu => println!("host memory"),
///     RuntimeType::Backend { backend, .. } => println!("{backend} device"),
///     RuntimeType::Unknown => println!("opaque"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeType {
    /// Payload is in host (CPU) memory.
    Cpu,
    /// Payload lives on an accelerator backend.
    Backend {
        backend: &'static str,
        device: &'static str,
    },
    /// Payload origin is unknown; materialize via [`to_host_tensor`](PacketPayload::to_host_tensor).
    Unknown,
}

impl RuntimeType {
    /// Convenience constructor for [`RuntimeType::Backend`].
    pub fn backend(backend: &'static str, device: &'static str) -> Self {
        Self::Backend { backend, device }
    }
}

/// A tensor in host (CPU) memory.
///
/// `HostTensor` is the canonical in-memory representation that all backends
/// can materialize to. Variants hold owned `Vec` data of the corresponding
/// Rust type.
///
/// # Example
///
/// ```
/// use lumnn::core::packet::{HostTensor, MLPacketDataType};
///
/// let t = HostTensor::Float32(vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(t.dtype(), MLPacketDataType::Float32);
/// assert_eq!(t.element_count(), 4);
/// ```
#[derive(Debug, Clone)]
pub enum HostTensor {
    Float32(Vec<f32>),
    Float16(Vec<f16>),
    Int32(Vec<i32>),
    Uint32(Vec<u32>),
    Int64(Vec<i64>),
    Uint64(Vec<u64>),
    Int8(Vec<i8>),
    Uint8(Vec<u8>),
}

impl HostTensor {
    /// Returns the data type of this tensor.
    pub fn dtype(&self) -> MLPacketDataType {
        match self {
            HostTensor::Float32(_) => MLPacketDataType::Float32,
            HostTensor::Float16(_) => MLPacketDataType::Float16,
            HostTensor::Int32(_) => MLPacketDataType::Int32,
            HostTensor::Uint32(_) => MLPacketDataType::Uint32,
            HostTensor::Int64(_) => MLPacketDataType::Int64,
            HostTensor::Uint64(_) => MLPacketDataType::Uint64,
            HostTensor::Int8(_) => MLPacketDataType::Int8,
            HostTensor::Uint8(_) => MLPacketDataType::Uint8,
        }
    }

    /// Returns the number of elements.
    pub fn element_count(&self) -> usize {
        match self {
            HostTensor::Float32(values) => values.len(),
            HostTensor::Float16(values) => values.len(),
            HostTensor::Int32(values) => values.len(),
            HostTensor::Uint32(values) => values.len(),
            HostTensor::Int64(values) => values.len(),
            HostTensor::Uint64(values) => values.len(),
            HostTensor::Int8(values) => values.len(),
            HostTensor::Uint8(values) => values.len(),
        }
    }

    /// Checks that dtype and element count match the given descriptor.
    ///
    /// # Errors
    ///
    /// Returns `Err` on dtype or element-count mismatch.
    pub fn validate_against(
        &self,
        descriptor: &MLPacketDescriptor,
        packet_name: &str,
    ) -> Result<(), String> {
        if self.dtype() != descriptor.dtype {
            return Err(format!(
                "input `{packet_name}` payload dtype mismatch: expected {:?}, got {:?}",
                descriptor.dtype,
                self.dtype()
            ));
        }

        if self.element_count() != descriptor.element_count() {
            return Err(format!(
                "input `{packet_name}` element count mismatch: expected {}, got {}",
                descriptor.element_count(),
                self.element_count()
            ));
        }

        Ok(())
    }

    /// Downcasts a `Box<dyn Any + Send + Sync>` into a `HostTensor` of the
    /// given dtype.
    ///
    /// Supports both `Vec<T>` and `Box<[T]>` as the underlying allocation.
    pub fn from_any(
        data: Box<dyn Any + Send + Sync>,
        dtype: MLPacketDataType,
    ) -> Result<Self, String> {
        match dtype {
            MLPacketDataType::Float32 => downcast_boxed_vec::<f32>(data).map(HostTensor::Float32),
            MLPacketDataType::Float16 => downcast_boxed_vec::<f16>(data).map(HostTensor::Float16),
            MLPacketDataType::Int32 => downcast_boxed_vec::<i32>(data).map(HostTensor::Int32),
            MLPacketDataType::Uint32 => downcast_boxed_vec::<u32>(data).map(HostTensor::Uint32),
            MLPacketDataType::Int64 => downcast_boxed_vec::<i64>(data).map(HostTensor::Int64),
            MLPacketDataType::Uint64 => downcast_boxed_vec::<u64>(data).map(HostTensor::Uint64),
            MLPacketDataType::Int8 => downcast_boxed_vec::<i8>(data).map(HostTensor::Int8),
            MLPacketDataType::Uint8 => downcast_boxed_vec::<u8>(data).map(HostTensor::Uint8),
        }
    }
}

/// Abstraction over backend-specific tensor payloads.
///
/// Implementations live in backend crates and wrap device-resident tensors
/// (accelerator buffers, runtime tensors, etc.). The trait lets [`MLPacket`] store a
/// heterogeneous payload without knowing the concrete backend type.
///
/// The default implementation is [`HostTensor`], which represents CPU-resident
/// data.
///
/// # Required methods
///
/// * [`runtime`](PacketPayload::runtime) — where the payload lives; consumers
///   use this to pick a fast path.
/// * [`to_host_tensor`](PacketPayload::to_host_tensor) — materialize to host
///   memory. For [`HostTensor`] this is a cheap clone; for device payloads it
///   may trigger a GPU→CPU transfer.
pub trait PacketPayload: Send + Sync + Any {
    /// Where this payload resides (CPU, GPU, etc.).
    fn runtime(&self) -> RuntimeType;
    /// Erased reference for downcasting.
    fn as_any(&self) -> &(dyn Any + Send + Sync);
    /// Erased owned value for downcasting.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
    /// Materialize this payload into a [`HostTensor`].
    ///
    /// For host-resident data this is cheap; for device data it may transfer.
    fn to_host_tensor(&self) -> Result<HostTensor, String>;
}

impl PacketPayload for HostTensor {
    fn runtime(&self) -> RuntimeType {
        RuntimeType::Cpu
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
    fn to_host_tensor(&self) -> Result<HostTensor, String> {
        Ok(self.clone())
    }
}

/// The unit of data flowing through a computational graph.
///
/// An `MLPacket` carries:
/// * a public [`descriptor`](MLPacketDescriptor) (dtype + shape),
/// * a reference-counted [`MLContext`] (ownership anchor),
/// * a [`PacketPayload`] behind a [`tokio::sync::Mutex`] for async-safe access.
///
/// Packets are created via [`MLContext`] factory methods
/// ([`packet_from_f32`](MLContext::packet_from_f32),
/// [`packet_from_host_tensor`](MLContext::packet_from_host_tensor)),
/// or directly through the constructors on this type.
///
/// # Lifecycle
///
/// 1. **Create** — context creates the packet with a payload.
/// 2. **Flow** — nodes consume input packets and produce output packets.
/// 3. **Inspect** — call [`runtime`](MLPacket::runtime) to dispatch,
///    [`to_host_tensor`](MLPacket::to_host_tensor) to read data.
/// 4. **Destroy** — call [`destroy`](MLPacket::destroy) to eagerly drop the
///    payload (useful for GPU memory management).
/// 5. **Consume** — call [`into_parts`](MLPacket::into_parts) to take ownership
///    of the payload for zero-copy processing.
///
/// # Example
///
/// ```
/// use lumnn::core::{
///     context::{MLContext, MLContextOptions},
///     packet::{HostTensor, MLPacketDescriptor, MLPacketDataType},
/// };
///
/// let ctx = MLContext::new(MLContextOptions::default()).unwrap();
/// let desc = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 3]);
///
/// let packet = ctx.packet_from_host_tensor(desc, HostTensor::Float32(vec![1.0, 2.0, 3.0]))
///     .expect("creation should succeed");
///
/// // Later, in an async context:
/// // let tensor = packet.to_host_tensor().await?;
/// ```
pub struct MLPacket {
    pub descriptor: MLPacketDescriptor,
    context: Arc<MLContext>,
    data: Mutex<Option<Box<dyn PacketPayload>>>,
}

impl MLPacket {
    /// Creates a packet from an untyped `Box<dyn Any>`.
    ///
    /// The data is downcast based on `descriptor.dtype` and wrapped in a
    /// [`HostTensor`]. Supports both `Vec<T>` and `Box<[T]>` allocations.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the downcast fails or the data doesn't match the
    /// descriptor.
    pub fn new(
        context: Arc<MLContext>,
        descriptor: MLPacketDescriptor,
        data: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, String> {
        let host_tensor = HostTensor::from_any(data, descriptor.dtype)?;
        host_tensor.validate_against(&descriptor, "packet")?;

        Ok(Self::from_payload(
            context,
            descriptor,
            Box::new(host_tensor),
        ))
    }

    /// Creates a packet from a [`HostTensor`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor's dtype or element count disagrees with
    /// the descriptor.
    pub fn from_host_tensor(
        context: Arc<MLContext>,
        descriptor: MLPacketDescriptor,
        tensor: HostTensor,
    ) -> Result<Self, String> {
        tensor.validate_against(&descriptor, "packet")?;
        Ok(Self::from_payload(context, descriptor, Box::new(tensor)))
    }

    /// Creates a packet from an already-constructed [`PacketPayload`].
    ///
    /// This is the low-level entry point for backend implementations that
    /// produce custom payload types (e.g., ORT values directly on GPU).
    pub fn from_payload(
        context: Arc<MLContext>,
        descriptor: MLPacketDescriptor,
        payload: Box<dyn PacketPayload>,
    ) -> Self {
        Self {
            descriptor,
            context,
            data: Mutex::new(Some(payload)),
        }
    }

    /// Eagerly drops the payload.
    ///
    /// Useful for releasing GPU memory before the packet itself goes out of
    /// scope. After destruction, [`runtime`](MLPacket::runtime) and
    /// [`to_host_tensor`](MLPacket::to_host_tensor) return errors.
    pub async fn destroy(&self) {
        let mut data_guard = self.data.lock().await;
        *data_guard = None;
    }

    /// Returns the [`RuntimeType`] of the current payload.
    ///
    /// Nodes use this to decide between fast paths (payload already on the
    /// right device) and slow paths (transfer needed).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the payload has been [`destroy`](MLPacket::destroy)ed.
    pub async fn runtime(&self) -> Result<RuntimeType, String> {
        let data_guard = self.data.lock().await;
        let payload = data_guard
            .as_ref()
            .ok_or_else(|| "MLPacket payload has already been destroyed".to_string())?;
        Ok(payload.runtime())
    }

    /// Materializes the payload into a [`HostTensor`].
    ///
    /// For CPU-resident payloads this is a cheap clone. For device-resident
    /// payloads this triggers a device→host transfer, which may be expensive.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the payload has been [`destroy`](MLPacket::destroy)ed
    /// or if the backend transfer fails.
    pub async fn to_host_tensor(&self) -> Result<HostTensor, String> {
        let data_guard = self.data.lock().await;
        let payload = data_guard
            .as_ref()
            .ok_or_else(|| "MLPacket payload has already been destroyed".to_string())?;
        payload.to_host_tensor()
    }

    /// Consumes the packet and returns its parts.
    ///
    /// This is the zero-copy extraction path: nodes that can consume the
    /// payload directly (without cloning or transferring) should use this
    /// instead of [`to_host_tensor`](MLPacket::to_host_tensor).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the payload has been [`destroy`](MLPacket::destroy)ed.
    pub fn into_parts(
        self,
    ) -> Result<(Arc<MLContext>, MLPacketDescriptor, Box<dyn PacketPayload>), String> {
        let Self {
            descriptor,
            context,
            data,
        } = self;

        let payload = data
            .into_inner()
            .ok_or_else(|| "MLPacket payload has already been destroyed".to_string())?;

        Ok((context, descriptor, payload))
    }
}

/// Attempts to downcast a boxed Any to a Vec<T>.
fn downcast_boxed_vec<T: Send + Sync + 'static>(
    data: Box<dyn Any + Send + Sync>,
) -> Result<Vec<T>, String> {
    match data.downcast::<Vec<T>>() {
        Ok(values) => Ok(*values),
        Err(data) => match data.downcast::<Box<[T]>>() {
            Ok(values) => Ok(Vec::from(*values)),
            Err(_) => Err("payload type does not match descriptor dtype".to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context::{MLContext, MLContextOptions};
    use std::sync::Arc;

    fn test_context() -> Arc<MLContext> {
        MLContext::new(MLContextOptions::default()).expect("context creation should succeed")
    }

    #[test]
    fn test_descriptor_memory_size() {
        let desc = MLPacketDescriptor::new(
            MLPacketDataType::Float32,
            vec![1, 3, 224, 224], // Mock Image Input
        );
        assert_eq!(desc.element_count(), 150528);
        assert_eq!(desc.byte_length(), 602112); // 150528 * 4 bytes
    }

    #[test]
    fn test_dynamic_batch_descriptor_accepts_concrete_batch_size() {
        let expected = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 3, 224, 224])
            .with_dynamic_batch();
        let actual = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![4, 3, 224, 224]);

        expected
            .validate_compatibility(&actual, "pixel_values")
            .expect("dynamic batch descriptor should accept concrete batch size");
    }

    #[test]
    fn test_dynamic_batch_descriptor_rejects_non_batch_shape_change() {
        let expected = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 3, 224, 224])
            .with_dynamic_batch();
        let actual = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![4, 3, 384, 384]);

        let err = expected
            .validate_compatibility(&actual, "pixel_values")
            .expect_err("dynamic batch descriptor should reject non-batch shape changes");

        assert!(err.contains("shape mismatch"));
    }

    #[test]
    fn test_actual_descriptor_cannot_be_dynamic_batch() {
        let expected = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 3, 224, 224]);
        let actual = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 3, 224, 224])
            .with_dynamic_batch();

        let err = expected
            .validate_compatibility(&actual, "pixel_values")
            .expect_err("actual descriptor should be concrete");

        assert!(err.contains("concrete batch size"));
    }

    #[test]
    fn test_dynamic_axis_descriptor_accepts_concrete_dimension() {
        let expected =
            MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, 1]).with_dynamic_axis(1);
        let actual = MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, 64]);

        expected
            .validate_compatibility(&actual, "input_ids")
            .expect("dynamic sequence descriptor should accept concrete sequence length");
    }

    #[test]
    fn test_dynamic_non_batch_axis_accepts_zero_length_cache() {
        let expected = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2, 1, 64])
            .with_dynamic_axis(2);
        let actual = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2, 0, 64]);

        expected
            .validate_compatibility(&actual, "past_key_values.0.key")
            .expect("dynamic non-batch axes should accept zero-length KV cache");
    }

    #[test]
    fn test_actual_descriptor_cannot_have_dynamic_axes() {
        let expected = MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, 1]);
        let actual =
            MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, 1]).with_dynamic_axis(1);

        let err = expected
            .validate_compatibility(&actual, "input_ids")
            .expect_err("actual descriptor should be concrete");

        assert!(err.contains("concrete"));
    }

    #[test]
    fn test_packet_any_downcast() {
        // 1. Prepare data (simulating an image tensor read from external source)
        let raw_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // 2. Pack: Put into Packet
        let desc = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 4]);
        let packet = MLPacket::new(test_context(), desc, Box::new(raw_data))
            .expect("packet creation should succeed");

        // 3. Unpack: Simulate the next node taking data (using async block for testing)
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let data_guard = packet.data.lock().await;

            // 确保里面有数据
            let payload = data_guard.as_ref().expect("Packet data should not be None");
            let any_data = payload.as_any();

            // 核心测试：尝试把它强转回 Vec<f32>
            let extracted = any_data
                .downcast_ref::<HostTensor>()
                .expect("Downcast to HostTensor failed!");

            match extracted {
                HostTensor::Float32(values) => {
                    assert_eq!(values[0], 1.0);
                    assert_eq!(values.len(), 4);
                }
                other => panic!("unexpected host tensor variant: {other:?}"),
            }
        });
    }

    #[test]
    fn test_packet_runtime_and_to_host_tensor() {
        let packet = MLPacket::from_host_tensor(
            test_context(),
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 4]),
            HostTensor::Float32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .expect("packet creation should succeed");

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            assert_eq!(
                packet.runtime().await.expect("runtime should exist"),
                RuntimeType::Cpu
            );

            let host_tensor = packet
                .to_host_tensor()
                .await
                .expect("host tensor materialization should succeed");

            match host_tensor {
                HostTensor::Float32(values) => assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]),
                other => panic!("unexpected host tensor variant: {other:?}"),
            }
        });
    }

    #[test]
    fn test_packet_runtime_after_destroy_returns_error() {
        let packet = MLPacket::from_host_tensor(
            test_context(),
            MLPacketDescriptor::new(MLPacketDataType::Uint8, vec![4]),
            HostTensor::Uint8(vec![1, 2, 3, 4]),
        )
        .expect("packet creation should succeed");

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            packet.destroy().await;

            let runtime_err = packet
                .runtime()
                .await
                .expect_err("destroyed packet should not expose runtime");
            assert!(runtime_err.contains("destroyed"));

            let host_err = packet
                .to_host_tensor()
                .await
                .expect_err("destroyed packet should not materialize host tensor");
            assert!(host_err.contains("destroyed"));
        });
    }
}
