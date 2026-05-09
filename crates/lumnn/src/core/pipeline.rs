use crate::core::{
    context::MLContext,
    node::{MLNode, MLNodeRef},
    packet::MLPacket,
};
use std::{collections::HashMap, sync::Arc};

/// A linear pipeline that runs [`MLNode`]s in sequence over a shared
/// [`MLContext`].
///
/// `MLPipeline` is the executor of the computational graph: it wires nodes
/// together by feeding the output packets of node *i* as the input to node
/// *i+1*. Nodes are matched by port name — the output name produced by one
/// node must match the input name expected by the next.
///
/// Use [`MLPipelineBuilder`] (via [`MLPipeline::builder`]) to construct
/// pipelines, or [`MLPipeline::new`] if you already have a `Vec<MLNodeRef>`.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::collections::HashMap;
/// use lumnn::core::{
///     context::{MLContext, MLContextOptions},
///     node::{MLNode, MLNodeRef},
///     packet::{HostTensor, MLPacket, MLPacketDescriptor, MLPacketDataType},
///     pipeline::MLPipeline,
/// };
///
/// # use async_trait::async_trait;
/// # struct IdentityNode {
/// #     input_desc: HashMap<String, MLPacketDescriptor>,
/// #     output_desc: HashMap<String, MLPacketDescriptor>,
/// # }
/// # #[async_trait]
/// # impl MLNode for IdentityNode {
/// #   fn name(&self) -> &str { "id" }
/// #   fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> { &self.input_desc }
/// #   fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> { &self.output_desc }
/// #   async fn execute(&self, mut inputs: HashMap<String, MLPacket>, _: &MLContext) -> Result<HashMap<String, MLPacket>, String> {
/// #       let pkt = inputs.remove("x").unwrap();
/// #       Ok(HashMap::from([("y".into(), pkt)]))
/// #   }
/// # }
/// #
/// let ctx = MLContext::new(MLContextOptions::default()).unwrap();
/// let desc = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![4]);
///
/// let node = IdentityNode {
///     input_desc: HashMap::from([("x".into(), desc.clone())]),
///     output_desc: HashMap::from([("y".into(), desc.clone())]),
/// };
///
/// let pipeline = MLPipeline::builder("my_pipeline", ctx.clone())
///     .then(node)
///     .build()
///     .expect("pipeline should build");
///
/// // Run with named inputs
/// # let rt = tokio::runtime::Runtime::new().unwrap();
/// # rt.block_on(async {
/// let input = ctx.packet_from_f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let outputs = pipeline.run(HashMap::from([("x".into(), input)])).await.unwrap();
/// assert!(outputs.contains_key("y"));
/// # });
/// ```
pub struct MLPipeline {
    name: String,
    context: Arc<MLContext>,
    nodes: Vec<MLNodeRef>,
}

impl MLPipeline {
    /// Creates a [`MLPipelineBuilder`] for this pipeline.
    ///
    /// Prefer this over [`MLPipeline::new`] unless you already have a
    /// pre-built `Vec<MLNodeRef>`.
    pub fn builder(name: impl Into<String>, context: Arc<MLContext>) -> MLPipelineBuilder {
        MLPipelineBuilder::new(name, context)
    }

    /// Creates a pipeline directly from a list of nodes.
    pub fn new(name: impl Into<String>, context: Arc<MLContext>, nodes: Vec<MLNodeRef>) -> Self {
        Self {
            name: name.into(),
            context,
            nodes,
        }
    }

    /// Human-readable name, set at construction time.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The [`MLContext`] this pipeline was created with.
    pub fn context(&self) -> &Arc<MLContext> {
        &self.context
    }

    /// Number of nodes in the pipeline.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the pipeline has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Runs the pipeline: feeds `inputs` through all nodes sequentially.
    ///
    /// For each node *i*, the output `HashMap` of *i* becomes the input of
    /// *i+1*. This means nodes are automatically wired when an upstream node's
    /// output name matches a downstream node's input name.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// * The pipeline has no nodes.
    /// * Any node's [`execute`](MLNode::execute) fails (missing input,
    ///   shape mismatch, backend error, etc.).
    pub async fn run(
        &self,
        inputs: HashMap<String, MLPacket>,
    ) -> Result<HashMap<String, MLPacket>, String> {
        if self.nodes.is_empty() {
            return Err(format!("pipeline `{}` has no nodes", self.name));
        }

        let mut packets = inputs;
        for node in &self.nodes {
            packets = node.execute(packets, self.context.as_ref()).await?;
        }

        Ok(packets)
    }
}

/// Builder for constructing an [`MLPipeline`] one node at a time.
///
/// Created via [`MLPipeline::builder`]. Add nodes with
/// [`then`](MLPipelineBuilder::then), [`then_shared`](MLPipelineBuilder::then_shared),
/// or [`then_boxed`](MLPipelineBuilder::then_boxed), then call
/// [`build`](MLPipelineBuilder::build) to produce the pipeline.
///
/// # Example
///
/// ```
/// use lumnn::core::{
///     context::{MLContext, MLContextOptions},
///     pipeline::MLPipeline,
/// };
///
/// let ctx = MLContext::new(MLContextOptions::default()).unwrap();
///
/// // An empty pipeline fails to build
/// let result = MLPipeline::builder("empty", ctx.clone()).build();
/// match result {
///     Err(e) => assert!(e.contains("has no nodes")),
///     Ok(_) => panic!("empty pipeline should fail to build"),
/// }
/// ```
pub struct MLPipelineBuilder {
    name: String,
    context: Arc<MLContext>,
    nodes: Vec<MLNodeRef>,
}

impl MLPipelineBuilder {
    pub fn new(name: impl Into<String>, context: Arc<MLContext>) -> Self {
        Self {
            name: name.into(),
            context,
            nodes: Vec::new(),
        }
    }

    /// Adds a node by value, wrapping it in `Arc`.
    ///
    /// Use this when you own the node and want the builder to take care of
    /// `Arc`-wrapping.
    pub fn then<N>(mut self, node: N) -> Self
    where
        N: MLNode + 'static,
    {
        self.nodes.push(Arc::new(node));
        self
    }

    /// Adds a pre-existing [`MLNodeRef`].
    ///
    /// Use this when the same node is shared across multiple pipelines, or
    /// when the node was constructed elsewhere.
    pub fn then_shared(mut self, node: MLNodeRef) -> Self {
        self.nodes.push(node);
        self
    }

    /// Adds a `Box<dyn MLNode>`, converting it to an `Arc`.
    pub fn then_boxed(mut self, node: Box<dyn MLNode>) -> Self {
        self.nodes.push(Arc::from(node));
        self
    }

    /// Consumes the builder and returns a validated [`MLPipeline`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if no nodes were added.
    pub fn build(self) -> Result<MLPipeline, String> {
        if self.nodes.is_empty() {
            return Err(format!("pipeline `{}` has no nodes", self.name));
        }

        Ok(MLPipeline {
            name: self.name,
            context: self.context,
            nodes: self.nodes,
        })
    }
}
