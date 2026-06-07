use std::{future::Future, net::SocketAddr, sync::Arc, time::Duration};

#[cfg(test)]
use lumen_schema::Mdns;
use lumen_schema::ServerConfig;
use tonic::transport::Server;

use crate::{
    daemon::{
        BatcherConfig, DaemonError, DaemonResult, HubGrpcService, MdnsAdvertisement,
        proto::home_native::v1::inference_server::InferenceServer,
    },
    service::ServiceHub,
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
    let _mdns = MdnsAdvertisement::register(&config.mdns, addr.port())?;
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
    let _mdns = MdnsAdvertisement::register(&config.mdns, addr.port())?;
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
