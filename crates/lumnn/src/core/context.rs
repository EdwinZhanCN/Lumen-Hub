use crate::core::packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor};
use std::sync::Arc;

/// Options for constructing an [`MLContext`].
///
/// `MLContextOptions` carries the configuration knobs that affect how nodes
/// interpret the execution environment. It is deliberately minimal: the context
/// itself is a lightweight handle that owns no runtime resources; backend
/// selection and device placement are the responsibility of concrete
/// [`MLNode`](crate::core::node::MLNode) implementations.
///
/// # Default
///
/// The [`Default`] impl returns `MLContextOptions { accelerated: false }`,
/// equivalent to [`MLContextOptions::cpu`].
///
/// # Examples
///
/// ```
/// use lumnn::core::context::MLContextOptions;
///
/// // CPU-only context
/// let cpu_opts = MLContextOptions::cpu();
/// assert!(!cpu_opts.accelerated);
///
/// // Context that signals a preference for hardware acceleration
/// let accel_opts = MLContextOptions::accelerated();
/// assert!(accel_opts.accelerated);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MLContextOptions {
    /// Backend-neutral preference for accelerated execution.
    ///
    /// When `true`, nodes may use GPU, NPU, or other accelerator backends.
    /// When `false`, nodes should execute on the CPU.
    ///
    /// Concrete nodes decide whether and how to honor this preference.
    pub accelerated: bool,
}

impl MLContextOptions {
    /// Returns options configured for CPU-only execution.
    ///
    /// Nodes will use their CPU path when running against contexts created
    /// with these options.
    pub fn cpu() -> Self {
        Self { accelerated: false }
    }

    /// Returns options that signal a preference for accelerated execution.
    ///
    /// Nodes may route work to a GPU, NPU, or other accelerator.
    /// The exact backend is determined by each node based on its own
    /// capabilities and feature flags.
    pub fn accelerated() -> Self {
        Self { accelerated: true }
    }
}

/// Shared execution environment for machine-learning workloads.
///
/// `MLContext` is the central ownership domain for [`MLPacket`] values flowing
/// through an [`MLPipeline`](crate::core::pipeline::MLPipeline). It is a
/// lightweight handle — it owns no graph structure, backend runtime, or
/// device memory. Its core responsibilities are:
///
/// * **Packet factory** — every packet is created via the context, which becomes
///   the packet's shared owner. This guarantees that the context outlives any
///   packet it creates.
/// * **Execution hint** — the [`accelerated`](MLContext::accelerated) flag
///   tells nodes whether acceleration is preferred, so they can route to the
///   appropriate device path.
/// * **Ownership anchor** — the context is always held behind [`Arc`], so
///   multiple packets and pipelines can share the same environment without
///   lifetime tangles.
///
/// # Usage pattern
///
/// A typical workflow looks like:
///
/// 1. Create the context via [`MLContext::new`].
/// 2. Use the context to create input [`MLPacket`]s.
/// 3. Build a pipeline with the same `Arc<MLContext>`, add nodes,
///    and call [`MLPipeline::run`](crate::core::pipeline::MLPipeline::run).
///
/// ```
/// use std::sync::Arc;
/// use lumnn::core::context::{MLContext, MLContextOptions};
///
/// let ctx = MLContext::new(MLContextOptions::accelerated())
///     .expect("context creation should succeed");
///
/// let packet = ctx.packet_from_f32(
///     vec![1, 4],
///     vec![1.0, 2.0, 3.0, 4.0],
/// ).expect("packet creation should succeed");
///
/// assert_eq!(ctx.accelerated(), true);
/// ```
///
/// # Memory model
///
/// Packets created through a context hold an `Arc<MLContext>` internally.
/// When the call site drops its last reference to the context, all packets
/// created from it remain valid — the context becomes unreachable but the
/// packets themselves are self-contained. This is fine for inference; if you
/// need the context for anything after packet creation (e.g., creating more
/// packets), keep at least one `Arc<MLContext>` alive.
#[derive(Debug, Clone)]
pub struct MLContext {
    options: MLContextOptions,
}

impl MLContext {
    /// Creates a new `MLContext` with the given options, wrapped in an [`Arc`].
    ///
    /// The context is returned behind `Arc` so it can be shared across multiple
    /// packets ([`MLPacket`]), nodes ([`MLNode`](crate::core::node::MLNode)),
    /// and the pipeline itself.
    ///
    /// # Errors
    ///
    /// Currently this method always returns `Ok`. The `Result` return type is
    /// reserved for future initialization logic (e.g., validating that a
    /// requested backend is available).
    ///
    /// # Examples
    ///
    /// ```
    /// use lumnn::core::context::{MLContext, MLContextOptions};
    ///
    /// let ctx = MLContext::new(MLContextOptions::default())
    ///     .expect("context creation should succeed");
    /// ```
    pub fn new(options: MLContextOptions) -> Result<Arc<Self>, String> {
        Ok(Arc::new(Self { options }))
    }

    /// Returns whether this context was configured for accelerated execution.
    ///
    /// Nodes should inspect this flag and route to an accelerated backend
    /// (GPU, NPU, etc.) when `true`, or fall back to CPU when `false`.
    pub fn accelerated(&self) -> bool {
        self.options.accelerated
    }

    /// Creates an [`MLPacket`] from a [`HostTensor`], consuming the tensor and
    /// associating the packet with this context.
    ///
    /// The [`descriptor`](MLPacketDescriptor) and [`tensor`](HostTensor) must
    /// agree on data type and element count; otherwise an error is returned.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor's [`dtype`](HostTensor::dtype) or element
    /// count does not match the descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use lumnn::core::context::{MLContext, MLContextOptions};
    /// use lumnn::core::packet::{HostTensor, MLPacketDescriptor, MLPacketDataType};
    ///
    /// let ctx = MLContext::new(MLContextOptions::default()).unwrap();
    /// let descriptor = MLPacketDescriptor::new(
    ///     MLPacketDataType::Float32,
    ///     vec![1, 4],
    /// );
    /// let tensor = HostTensor::Float32(vec![1.0, 2.0, 3.0, 4.0]);
    ///
    /// let packet = ctx.packet_from_host_tensor(descriptor, tensor)
    ///     .expect("descriptor and tensor should be compatible");
    /// ```
    pub fn packet_from_host_tensor(
        self: &Arc<Self>,
        descriptor: MLPacketDescriptor,
        tensor: HostTensor,
    ) -> Result<MLPacket, String> {
        MLPacket::from_host_tensor(Arc::clone(self), descriptor, tensor)
    }

    /// Shorthand for creating an [`MLPacket`] backed by `Vec<f32>` host memory.
    ///
    /// Equivalent to calling [`packet_from_host_tensor`](MLContext::packet_from_host_tensor)
    /// with [`MLPacketDataType::Float32`] and [`HostTensor::Float32`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `data.len()` does not match `shape.iter().product()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lumnn::core::context::{MLContext, MLContextOptions};
    ///
    /// let ctx = MLContext::new(MLContextOptions::default()).unwrap();
    ///
    /// // A 2×2 f32 matrix
    /// let packet = ctx.packet_from_f32(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// ).expect("shape and data should agree");
    /// ```
    pub fn packet_from_f32(
        self: &Arc<Self>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<MLPacket, String> {
        self.packet_from_host_tensor(
            MLPacketDescriptor::new(MLPacketDataType::Float32, shape),
            HostTensor::Float32(data),
        )
    }
}
