use std::{collections::HashMap, env, net::IpAddr, sync::LazyLock, time::Duration};

use lumen_schema::Mdns;
use mdns_sd::{ServiceDaemon, ServiceInfo};
use rand::Rng;

use crate::daemon::{DaemonError, DaemonResult};

pub const DEFAULT_MDNS_SERVICE_TYPE: &str = "_lumen._tcp.local.";
const DEFAULT_SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

fn default_instance_name() -> String {
    static NAME: LazyLock<String> = LazyLock::new(|| {
        let hash: String = rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(5)
            .map(char::from)
            .collect();
        format!("lumen-hub-{}", hash.to_lowercase())
    });
    NAME.clone()
}

/// Display-only metadata announced in mDNS TXT records.
///
/// Connection identity comes from the DNS-SD instance name. Protocol
/// compatibility and task routing are negotiated in-band through the gRPC
/// capability stream, never through TXT records.
#[derive(Debug, Clone, Default)]
pub struct AdvertisedMetadata {
    pub runtime: Option<String>,
}

/// Keeps an mDNS registration alive and unregisters it on drop.
pub struct MdnsAdvertisement {
    daemon: ServiceDaemon,
    fullname: String,
}

impl MdnsAdvertisement {
    pub fn register(
        config: &Mdns,
        port: u16,
        metadata: &AdvertisedMetadata,
    ) -> DaemonResult<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }

        let hostname = mdns_hostname();
        let default_name = default_instance_name();
        let instance_name = config.service_name.as_deref().unwrap_or(&default_name);
        let properties = mdns_properties(metadata);

        // With an explicit ADVERTISE_IP we announce exactly that address.
        // Otherwise mdns-sd auto-detects and tracks all local interface
        // addresses, which keeps offline LANs working (no internet probe).
        let service_info = match advertise_ip_override()? {
            Some(ip) => {
                if ip.is_loopback() {
                    tracing::warn!(
                        %ip,
                        "ADVERTISE_IP is a loopback address; other devices will not be able to connect"
                    );
                }
                ServiceInfo::new(
                    DEFAULT_MDNS_SERVICE_TYPE,
                    instance_name,
                    &hostname,
                    ip,
                    port,
                    properties,
                )?
            }
            None => ServiceInfo::new(
                DEFAULT_MDNS_SERVICE_TYPE,
                instance_name,
                &hostname,
                "",
                port,
                properties,
            )?
            .enable_addr_auto(),
        };
        let fullname = service_info.get_fullname().to_owned();

        let daemon = ServiceDaemon::new()?;
        daemon.register(service_info)?;

        tracing::info!(
            service = %fullname,
            port,
            "mDNS service advertised"
        );

        Ok(Some(Self { daemon, fullname }))
    }

    pub fn fullname(&self) -> &str {
        &self.fullname
    }
}

impl Drop for MdnsAdvertisement {
    fn drop(&mut self) {
        match self.daemon.unregister(&self.fullname) {
            Ok(receiver) => {
                let _ = receiver.recv_timeout(Duration::from_secs(1));
            }
            Err(error) => {
                tracing::warn!(service = %self.fullname, %error, "failed to unregister mDNS service");
            }
        }

        match self.daemon.shutdown() {
            Ok(receiver) => {
                let _ = receiver.recv_timeout(Duration::from_secs(1));
            }
            Err(error) => {
                tracing::warn!(%error, "failed to shut down mDNS daemon");
            }
        }
    }
}

fn advertise_ip_override() -> DaemonResult<Option<IpAddr>> {
    match env::var("ADVERTISE_IP") {
        Ok(ip) => ip
            .parse::<IpAddr>()
            .map(Some)
            .map_err(|source| DaemonError::InvalidAdvertiseIp { ip, source }),
        Err(_) => Ok(None),
    }
}

fn mdns_hostname() -> String {
    let hostname = env::var("HOSTNAME")
        .ok()
        .filter(|hostname| !hostname.is_empty())
        .unwrap_or_else(|| "lumnn".to_owned());

    normalize_local_hostname(hostname)
}

fn normalize_local_hostname(hostname: String) -> String {
    if hostname.ends_with(".local.") {
        hostname
    } else if hostname.ends_with(".local") {
        format!("{hostname}.")
    } else {
        format!("{hostname}.local.")
    }
}

fn mdns_properties(metadata: &AdvertisedMetadata) -> HashMap<String, String> {
    let version =
        env::var("SERVICE_VERSION").unwrap_or_else(|_| DEFAULT_SERVICE_VERSION.to_owned());

    let mut properties = HashMap::from([("v".to_owned(), version)]);
    if let Some(runtime) = metadata
        .runtime
        .as_deref()
        .filter(|runtime| !runtime.is_empty())
    {
        properties.insert("runtime".to_owned(), runtime.to_owned());
    }
    properties
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdns_hostname_adds_local_suffix() {
        assert_eq!(normalize_local_hostname("lumnn".to_owned()), "lumnn.local.");
        assert_eq!(
            normalize_local_hostname("lumnn.local".to_owned()),
            "lumnn.local."
        );
        assert_eq!(
            normalize_local_hostname("lumnn.local.".to_owned()),
            "lumnn.local."
        );
    }

    #[test]
    fn default_service_type_is_local_tcp() {
        assert_eq!(DEFAULT_MDNS_SERVICE_TYPE, "_lumen._tcp.local.");
    }

    #[test]
    fn mdns_properties_are_display_only() {
        let metadata = AdvertisedMetadata {
            runtime: Some("burn".to_owned()),
        };
        let properties = mdns_properties(&metadata);

        assert_eq!(properties.get("runtime").map(String::as_str), Some("burn"));
        assert!(properties.contains_key("v"));
        assert_eq!(properties.len(), 2);
    }

    #[test]
    fn mdns_properties_omit_empty_runtime() {
        let properties = mdns_properties(&AdvertisedMetadata::default());

        assert!(!properties.contains_key("runtime"));
        assert!(properties.contains_key("v"));
        assert_eq!(properties.len(), 1);
    }
}
