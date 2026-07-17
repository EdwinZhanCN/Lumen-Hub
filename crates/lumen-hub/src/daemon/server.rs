use std::{
    future::Future,
    net::SocketAddr,
    sync::{Arc, OnceLock},
    time::Duration,
};

#[cfg(test)]
use lumen_schema::Mdns;
use lumen_schema::ServerConfig;
use tonic::transport::Server;
use tonic_health::{ServingStatus, server::HealthReporter};

use crate::{
    daemon::{
        AdvertisedCapabilities, BatcherConfig, DaemonError, DaemonResult, HubGrpcService,
        MdnsAdvertisement,
        control::ControlGrpcService,
        lazy::{HubSlot, LazyInference},
        proto::{
            home_native::v1::inference_server::InferenceServer,
            lumen::control::v1::control_server::ControlServer,
        },
    },
    service::ServiceHub,
    status::{LogBuffer, StatusBus},
};

pub fn bind_addr(config: &ServerConfig) -> DaemonResult<SocketAddr> {
    bind_addr_with_port(config, None)
}

pub fn bind_addr_with_port(
    config: &ServerConfig,
    port_override: Option<u16>,
) -> DaemonResult<SocketAddr> {
    let host = if config.host.is_empty() {
        "0.0.0.0"
    } else {
        config.host.as_str()
    };
    let port = port_override.unwrap_or(config.port);
    let raw_addr = format!("{host}:{port}");

    raw_addr
        .parse::<SocketAddr>()
        .map_err(|source| DaemonError::InvalidBindAddress {
            host: host.to_owned(),
            port,
            source,
        })
}

pub async fn serve_grpc(hub: Arc<ServiceHub>, config: &ServerConfig) -> DaemonResult<()> {
    let addr = bind_addr(config)?;
    serve_grpc_at_addr(hub, config, addr).await
}

pub async fn serve_grpc_with_shutdown<S>(
    hub: Arc<ServiceHub>,
    config: &ServerConfig,
    shutdown: S,
) -> DaemonResult<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let addr = bind_addr(config)?;
    serve_grpc_at_addr_with_shutdown(hub, config, addr, shutdown).await
}

async fn serve_grpc_at_addr(
    hub: Arc<ServiceHub>,
    config: &ServerConfig,
    addr: SocketAddr,
) -> DaemonResult<()> {
    let _mdns =
        MdnsAdvertisement::register(&config.mdns, addr.port(), &advertised_capabilities(&hub))?;
    tracing::info!(%addr, services = hub.len(), "starting Lumen gRPC server");

    Server::builder()
        .add_service(InferenceServer::new(HubGrpcService::new(
            hub,
            batcher_config(config),
        )))
        .serve(addr)
        .await?;

    Ok(())
}

async fn serve_grpc_at_addr_with_shutdown<S>(
    hub: Arc<ServiceHub>,
    config: &ServerConfig,
    addr: SocketAddr,
    shutdown: S,
) -> DaemonResult<()>
where
    S: Future<Output = ()> + Send + 'static,
{
    let _mdns =
        MdnsAdvertisement::register(&config.mdns, addr.port(), &advertised_capabilities(&hub))?;
    tracing::info!(%addr, services = hub.len(), "starting Lumen gRPC server");

    Server::builder()
        .add_service(InferenceServer::new(HubGrpcService::new(
            hub,
            batcher_config(config),
        )))
        .serve_with_shutdown(addr, shutdown)
        .await?;

    Ok(())
}

/// The supervised control-plane server: binds before models exist, serving
/// `lumen.control.v1.Control`, `grpc.health.v1.Health`, and a gated
/// `home_native.v1.Inference` that returns UNAVAILABLE until ready.
pub struct ControlPlaneServer {
    router: tonic::service::Routes,
}

/// Handed to the startup task; flips the server to ready (or failed) once the
/// hub is built and warmed up.
pub struct ReadyHandle {
    slot: HubSlot,
    health: HealthReporter,
}

impl ReadyHandle {
    /// Register the initial NOT_SERVING statuses so `Check("")` answers
    /// meaningfully during startup instead of NOT_FOUND.
    pub async fn init(&self) {
        self.health
            .set_service_status("", ServingStatus::NotServing)
            .await;
        self.health
            .set_not_serving::<InferenceServer<LazyInference>>()
            .await;
    }

    pub async fn set_ready(&self, service: HubGrpcService) {
        if self.slot.set(service).is_err() {
            tracing::warn!("hub service was already installed; ignoring duplicate set_ready");
            return;
        }
        self.health
            .set_service_status("", ServingStatus::Serving)
            .await;
        self.health
            .set_serving::<InferenceServer<LazyInference>>()
            .await;
    }

    pub async fn set_failed(&self) {
        self.health
            .set_service_status("", ServingStatus::NotServing)
            .await;
        self.health
            .set_not_serving::<InferenceServer<LazyInference>>()
            .await;
    }
}

/// Builds the control-plane router. Serve it with
/// [`ControlPlaneServer::serve_with_shutdown`]; complete startup through the
/// returned [`ReadyHandle`].
pub fn control_plane(
    bus: Arc<StatusBus>,
    logs: Arc<LogBuffer>,
) -> (ControlPlaneServer, ReadyHandle) {
    let slot: HubSlot = Arc::new(OnceLock::new());
    let (health, health_service) = tonic_health::server::health_reporter();

    let router = tonic::service::Routes::new(health_service)
        .add_service(ControlServer::new(ControlGrpcService::new(bus, logs)))
        .add_service(InferenceServer::new(LazyInference::new(Arc::clone(&slot))));

    (ControlPlaneServer { router }, ReadyHandle { slot, health })
}

impl ControlPlaneServer {
    pub async fn serve_with_shutdown<S>(self, addr: SocketAddr, shutdown: S) -> DaemonResult<()>
    where
        S: Future<Output = ()> + Send + 'static,
    {
        Server::builder()
            .add_routes(self.router)
            .serve_with_shutdown(addr, shutdown)
            .await?;
        Ok(())
    }
}

/// Batcher settings derived from the server config; public so the startup task
/// can construct the ready `HubGrpcService`.
pub fn hub_batcher_config(config: &ServerConfig) -> BatcherConfig {
    batcher_config(config)
}

/// Collects the mDNS TXT hints (task names, runtime) from the registered
/// services so discovery clients can route before fetching capabilities.
pub fn advertised_capabilities(hub: &ServiceHub) -> AdvertisedCapabilities {
    let mut seen = std::collections::BTreeSet::new();
    let mut tasks = Vec::new();
    let mut runtime = None;
    for capability in hub.capabilities() {
        if runtime.is_none() && !capability.runtime.is_empty() {
            runtime = Some(capability.runtime.clone());
        }
        for task in &capability.tasks {
            if seen.insert(task.name.clone()) {
                tasks.push(task.name.clone());
            }
        }
    }
    AdvertisedCapabilities { tasks, runtime }
}

fn batcher_config(config: &ServerConfig) -> BatcherConfig {
    BatcherConfig {
        enabled: config.batching.enabled,
        max_batch_size: config.batching.max_batch_size,
        queue_latency: Duration::from_millis(config.batching.queue_latency_ms),
    }
}

#[cfg(test)]
mod tests {
    use lumen_schema::ServerConfig;

    use super::*;

    #[test]
    fn bind_addr_uses_config_host_and_port() {
        let config = server_config("127.0.0.1", 50_051);

        assert_eq!(bind_addr(&config).unwrap().to_string(), "127.0.0.1:50051");
    }

    #[test]
    fn bind_addr_supports_bracketed_ipv6() {
        let config = server_config("[::]", 50_051);

        assert_eq!(bind_addr(&config).unwrap().to_string(), "[::]:50051");
    }

    #[test]
    fn bind_addr_allows_port_override() {
        let config = server_config("0.0.0.0", 50_051);

        assert_eq!(
            bind_addr_with_port(&config, Some(50_052))
                .unwrap()
                .to_string(),
            "0.0.0.0:50052"
        );
    }

    #[test]
    fn bind_addr_rejects_non_socket_host() {
        let config = server_config("localhost", 50_051);

        assert!(matches!(
            bind_addr(&config),
            Err(DaemonError::InvalidBindAddress { .. })
        ));
    }

    fn server_config(host: &str, port: u16) -> ServerConfig {
        ServerConfig {
            port,
            host: host.to_owned(),
            mdns: Mdns::default(),
            batching: Default::default(),
        }
    }
}
