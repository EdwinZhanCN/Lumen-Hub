use crate::core::{
    context::MLContext,
    node::{MLNode, MLNodeRef},
    packet::MLPacket,
};
use std::{collections::HashMap, sync::Arc};

/// A linear pipeline that executes nodes sequentially in a shared context.
pub struct MLPipeline {
    name: String,
    context: Arc<MLContext>,
    nodes: Vec<MLNodeRef>,
}

impl MLPipeline {
    pub fn builder(name: impl Into<String>, context: Arc<MLContext>) -> MLPipelineBuilder {
        MLPipelineBuilder::new(name, context)
    }

    pub fn new(name: impl Into<String>, context: Arc<MLContext>, nodes: Vec<MLNodeRef>) -> Self {
        Self {
            name: name.into(),
            context,
            nodes,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn context(&self) -> &Arc<MLContext> {
        &self.context
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

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

    pub fn then<N>(mut self, node: N) -> Self
    where
        N: MLNode + 'static,
    {
        self.nodes.push(Arc::new(node));
        self
    }

    pub fn then_shared(mut self, node: MLNodeRef) -> Self {
        self.nodes.push(node);
        self
    }

    pub fn then_boxed(mut self, node: Box<dyn MLNode>) -> Self {
        self.nodes.push(Arc::from(node));
        self
    }

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
