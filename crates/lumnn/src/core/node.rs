use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc};

use crate::core::context::MLContext;
use crate::core::packet::{MLPacket, MLPacketDescriptor};

#[async_trait]
/// Represents a machine learning node that processes input packets into output packets asynchronously.
pub trait MLNode: Send + Sync {
    /// Returns the name of the node.
    fn name(&self) -> &str;

    /// Returns the descriptors for the input packets required by this node.
    fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor>;

    /// Returns the descriptors for the output packets produced by this node.
    fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor>;

    /// Executes the node's core logic with the provided inputs and context, returning the output packets.
    async fn execute(
        &self,
        inputs: HashMap<String, MLPacket>,
        context: &MLContext,
    ) -> Result<HashMap<String, MLPacket>, String>;
}

/// Shared reference to a machine learning node.
pub type MLNodeRef = Arc<dyn MLNode>;
