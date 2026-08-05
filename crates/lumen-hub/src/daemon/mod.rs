//! Daemon module.
//!
//! Placeholder module for background services (e.g. discovery, scheduling,
//! health checks, and runtime coordination).

pub mod batcher;
pub mod control;
pub mod error;
pub mod grpc;
pub mod lazy;
pub mod mdns;
pub mod proto;
pub mod server;

pub use batcher::{BatchFn, Batcher, BatcherConfig};
pub use control::ControlGrpcService;
pub use error::{DaemonError, DaemonResult};
pub use grpc::HubGrpcService;
pub use lazy::{HubSlot, LazyInference};
pub use mdns::{AdvertisedMetadata, DEFAULT_MDNS_SERVICE_TYPE, MdnsAdvertisement};
pub use server::{
    ControlPlaneServer, ReadyHandle, advertised_metadata, bind_addr, bind_addr_with_port,
    control_plane, hub_batcher_config, serve_grpc, serve_grpc_with_shutdown,
};
