use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc};

use crate::core::context::MLContext;
use crate::core::packet::{MLPacket, MLPacketDescriptor};

/// A vertex in a computational graph.
///
/// `MLNode` is the fundamental abstraction that turns machine-learning workloads
/// into composable, named units of computation. Every node declares:
///
/// * **What it consumes** — a named set of input [`MLPacketDescriptor`]s that
///   act as the node's input "ports" (graph edges entering this vertex).
/// * **What it produces** — a named set of output descriptors that act as the
///   node's output "ports" (graph edges leaving this vertex).
/// * **How it executes** — an async `execute()`(MLNode::execute) method that
///   receives concrete [`MLPacket`] values keyed by input name and returns
///   output packets keyed by output name.
///
/// # Graph model
///
/// Edges in a lumnn computational graph are **implicit**: they are not
/// represented by a separate struct. Instead, the output name produced by one
/// node must match the input name expected by the next downstream node. The
/// [`MLPipeline`](crate::core::pipeline::MLPipeline) wires nodes together by
/// feeding the output `HashMap` of node *i* as the input `HashMap` of node
/// *i+1*. This is analogous to how frameworks like PyTorch and TensorFlow
/// resolve tensor names to connect layers.
///
/// ```text
/// ┌──────────┐    "features"    ┌───────────┐    "logits"   ┌────────┐
/// │ Encoder  │ ───────────────→ │  Decoder  │ ────────────→ │  Head  │
/// │  Node    │                  │   Node    │               │  Node  │
/// └──────────┘                  └───────────┘               └────────┘
/// ```
///
/// Each node is **stateless** from the trait's perspective — `execute()` takes
/// `&self` (not `&mut self`), allowing the same `Arc<dyn MLNode>` to be shared
/// across multiple pipelines or reused within the same pipeline.
///
/// # Lifecycle
///
/// 1. **Registration** — the node is created and wrapped in an
///    [`MLNodeRef`] (`Arc<dyn MLNode>`). At this point callers can inspect
///    [`input_descriptors`](MLNode::input_descriptors) and
///    [`output_descriptors`](MLNode::output_descriptors) to verify graph
///    connectivity before execution.
/// 2. **Execution** — the pipeline calls `execute()` with a `HashMap` of
///    named inputs. The node consumes those packets, performs its computation
///    (which may involve GPU work, ONNX Runtime sessions, or pure CPU math),
///    and returns a `HashMap` of named outputs.
///
/// # Concurrency
///
/// The supertraits `Send + Sync` mean a single `Arc<dyn MLNode>` can be
/// invoked concurrently from multiple pipelines or multiple async tasks.
/// Implementations that hold mutable state internally (e.g., an ONNX Runtime
/// session) must use interior mutability ([`std::sync::Mutex`], [`tokio::sync::Mutex`],
/// etc.) to uphold the `&self` contract.
///
/// # Implementing the trait
///
/// Concrete implementations fall into two categories:
///
/// * **Model nodes** — wrap a serialized model (ONNX, Candle, etc.), parse its
///   I/O signatures to populate descriptors, and run inference on each
///   `execute()` call. Examples: [`OrtNode`](crate::ort::node::OrtNode),
///   [`CandleOnnxNode`](crate::candle::node::CandleOnnxNode).
/// * **Operator nodes** — perform a specific mathematical or data-movement
///   operation directly on host tensors, without a separate model file.
///   Example: [`NdArrayNode`](crate::ndarray::node::NdArrayNode).
///
/// For a minimal custom node, see the `MockNode` in the pipeline integration
/// tests.
///
/// # Examples
///
/// Implementing a simple doubling node:
///
/// ```
/// use async_trait::async_trait;
/// use std::collections::HashMap;
/// use lumnn::core::{
///     context::MLContext,
///     node::MLNode,
///     packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
/// };
///
/// struct DoubleNode {
///     input_descriptors: HashMap<String, MLPacketDescriptor>,
///     output_descriptors: HashMap<String, MLPacketDescriptor>,
/// }
///
/// impl DoubleNode {
///     fn new() -> Self {
///         let desc = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 4]);
///         Self {
///             input_descriptors: HashMap::from([("x".into(), desc.clone())]),
///             output_descriptors: HashMap::from([("y".into(), desc)]),
///         }
///     }
/// }
///
/// #[async_trait]
/// impl MLNode for DoubleNode {
///     fn name(&self) -> &str { "double" }
///
///     fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
///         &self.input_descriptors
///     }
///
///     fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
///         &self.output_descriptors
///     }
///
///     async fn execute(
///         &self,
///         mut inputs: HashMap<String, MLPacket>,
///         _context: &MLContext,
///     ) -> Result<HashMap<String, MLPacket>, String> {
///         let packet = inputs.remove("x")
///             .ok_or("missing input `x`")?;
///         let (context, desc, payload) = packet.into_parts()?;
///         let values = match payload.to_host_tensor()? {
///             HostTensor::Float32(v) => v,
///             other => return Err(format!("expected Float32, got {other:?}")),
///         };
///         let doubled: Vec<f32> = values.iter().map(|v| v * 2.0).collect();
///         let output = MLPacket::from_host_tensor(context, desc, HostTensor::Float32(doubled))?;
///         Ok(HashMap::from([("y".into(), output)]))
///     }
/// }
/// ```
#[async_trait]
pub trait MLNode: Send + Sync {
    /// Returns a human-readable name for this node.
    ///
    /// Useful for debugging, logging, and error messages that identify
    /// which vertex in the graph produced a failure.
    fn name(&self) -> &str;

    /// Returns the descriptors for the input packets required by this node.
    ///
    /// Each entry maps an input port name (the key) to the expected
    /// [`MLPacketDescriptor`] (data type, shape, and dynamic-axis policy).
    /// The pipeline uses these descriptors to validate that upstream nodes
    /// produce compatible packets before execution begins.
    ///
    /// # Naming convention
    ///
    /// Input names form the **edges** of the computational graph. For two
    /// nodes *A* → *B* to be connected, *A*'s output name must match *B*'s
    /// input name. Choose stable, semantic names (e.g., `"pixel_values"`,
    /// `"input_ids"`, `"attention_mask"`) rather than positional indices.
    fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor>;

    /// Returns the descriptors for the output packets produced by this node.
    ///
    /// Each entry maps an output port name to the descriptor that downstream
    /// nodes can expect. Implementations should return the same descriptors
    /// across calls — they represent the static contract of the node, not the
    /// shape of any particular execution.
    fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor>;

    /// Executes the node's computation on the given inputs.
    ///
    /// The `inputs` map is keyed by input port name; every key must match an
    /// entry in [`input_descriptors`](MLNode::input_descriptors). The returned
    /// map is keyed by output port name; every key must match an entry in
    /// [`output_descriptors`](MLNode::output_descriptors).
    ///
    /// # Contract
    ///
    /// * The node **takes ownership** of the input packets. It may destroy
    ///   them, extract their payloads, or pass them through unchanged.
    /// * The returned output packets must be created via the same
    ///   [`MLContext`] that was passed in (or a clone of it), so they share
    ///   the same ownership domain.
    /// * This method takes `&self` (not `&mut self`), enabling the same node
    ///   to be shared across pipelines. Implementations that need mutable
    ///   internal state (e.g., an ONNX session) must use interior mutability.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// * A required input is missing from `inputs`.
    /// * An input packet's actual descriptor is incompatible with the
    ///   declared [`input_descriptors`](MLNode::input_descriptors).
    /// * The underlying computation fails (model runtime error, shape
    ///   mismatch, device error, etc.).
    async fn execute(
        &self,
        inputs: HashMap<String, MLPacket>,
        context: &MLContext,
    ) -> Result<HashMap<String, MLPacket>, String>;
}

/// Shared reference to a computational-graph node.
///
/// `MLNodeRef` is the standard handle for passing nodes around the system:
/// it is used by [`MLPipeline`](crate::core::pipeline::MLPipeline) to store
/// its node sequence, and by [`MLPipelineBuilder::then_shared`](crate::core::pipeline::MLPipelineBuilder::then_shared)
/// to accept pre-constructed `Arc<dyn MLNode>` values.
///
/// Because `MLNode` requires `Send + Sync`, an `MLNodeRef` can be sent across
/// thread boundaries and shared across multiple pipelines concurrently.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use lumnn::core::node::MLNodeRef;
/// // OrtNode, CandleOnnxNode, NdArrayNode, and any custom MLNode
/// // implementation can all be stored in an MLNodeRef.
/// fn register(name: &str, node: MLNodeRef) {
///     println!("registered node `{name}`");
/// }
/// ```
pub type MLNodeRef = Arc<dyn MLNode>;
