//! Daemon module.
//!
//! Placeholder module for background services (e.g. discovery, scheduling,
//! health checks, and runtime coordination).

pub mod batcher;
pub mod error;
pub mod grpc;
pub mod mdns;
pub mod proto;
pub mod server;

pub use error::{DaemonError, DaemonResult};
pub use grpc::HubGrpcService;
pub use mdns::{DEFAULT_MDNS_SERVICE_TYPE, MdnsAdvertisement};
pub use server::{bind_addr, bind_addr_with_port, serve_grpc, serve_grpc_with_shutdown};
