//! Data-plane gate: the gRPC port binds before models are downloaded/loaded,
//! so `Inference` is served through this wrapper — `UNAVAILABLE` until the
//! real `HubGrpcService` is installed, then a transparent delegate.

use std::sync::{Arc, OnceLock};

use tonic::{Request, Response, Status};

use crate::daemon::{
    HubGrpcService,
    proto::home_native::v1::{self, inference_server::Inference},
};

/// Shared slot the startup task fills once the hub is built and warmed up.
pub type HubSlot = Arc<OnceLock<HubGrpcService>>;

pub struct LazyInference {
    slot: HubSlot,
}

impl LazyInference {
    pub fn new(slot: HubSlot) -> Self {
        Self { slot }
    }

    fn ready(&self) -> Result<&HubGrpcService, Status> {
        self.slot
            .get()
            .ok_or_else(|| Status::unavailable("lumen hub is starting; inference is not ready yet"))
    }
}

#[tonic::async_trait]
impl Inference for LazyInference {
    type InferStream = <HubGrpcService as Inference>::InferStream;

    async fn infer(
        &self,
        request: Request<tonic::Streaming<v1::InferRequest>>,
    ) -> Result<Response<Self::InferStream>, Status> {
        self.ready()?.infer(request).await
    }

    async fn get_capabilities(
        &self,
        request: Request<()>,
    ) -> Result<Response<v1::Capability>, Status> {
        self.ready()?.get_capabilities(request).await
    }

    type StreamCapabilitiesStream = <HubGrpcService as Inference>::StreamCapabilitiesStream;

    async fn stream_capabilities(
        &self,
        request: Request<()>,
    ) -> Result<Response<Self::StreamCapabilitiesStream>, Status> {
        self.ready()?.stream_capabilities(request).await
    }

    async fn health(&self, request: Request<()>) -> Result<Response<()>, Status> {
        // Legacy in-band probe: reachable while starting, mirroring the old
        // "port open = process alive" semantics. Real readiness lives in
        // grpc.health.v1 and lumen.control.v1.
        match self.slot.get() {
            Some(inner) => inner.health(request).await,
            None => Ok(Response::new(())),
        }
    }
}
