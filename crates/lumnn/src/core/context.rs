use crate::core::packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default)]
pub struct MLContextOptions {
    /// Backend-neutral preference for accelerated execution.
    ///
    /// Concrete nodes decide whether and how to honor this preference.
    pub accelerated: bool,
}

impl MLContextOptions {
    pub fn cpu() -> Self {
        Self { accelerated: false }
    }

    pub fn accelerated() -> Self {
        Self { accelerated: true }
    }
}

/// Shared execution environment for machine-learning workloads.
///
/// A context owns no graph structure or backend runtime by itself. It acts as
/// the factory / ownership domain for packets that flow between nodes.
#[derive(Debug, Clone)]
pub struct MLContext {
    options: MLContextOptions,
}

impl MLContext {
    pub fn new(options: MLContextOptions) -> Result<Arc<Self>, String> {
        Ok(Arc::new(Self { options }))
    }

    pub fn accelerated(&self) -> bool {
        self.options.accelerated
    }

    pub fn packet_from_host_tensor(
        self: &Arc<Self>,
        descriptor: MLPacketDescriptor,
        tensor: HostTensor,
    ) -> Result<MLPacket, String> {
        MLPacket::from_host_tensor(Arc::clone(self), descriptor, tensor)
    }

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
