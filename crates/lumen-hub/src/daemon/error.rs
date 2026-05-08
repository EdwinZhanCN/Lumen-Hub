use std::net::AddrParseError;

/// Errors raised by daemon startup and network service advertisement.
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    #[error("invalid server bind address `{host}:{port}`: {source}")]
    InvalidBindAddress {
        host: String,
        port: u16,
        source: AddrParseError,
    },

    #[error("invalid advertised IP `{ip}`: {source}")]
    InvalidAdvertiseIp { ip: String, source: AddrParseError },

    #[error("mDNS advertisement failed: {0}")]
    Mdns(#[from] mdns_sd::Error),

    #[error("gRPC server failed: {0}")]
    Transport(#[from] tonic::transport::Error),
}

pub type DaemonResult<T> = Result<T, DaemonError>;
