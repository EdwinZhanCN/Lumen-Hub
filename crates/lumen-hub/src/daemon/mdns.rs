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

/// What the hub announces about itself in mDNS TXT records.
///
/// The TXT keys are a contract with lumen-sdk (`pkg/discovery`): the SDK reads
/// `v` (version), `runtime`, `tasks` (CSV), and `proto` (data-plane protocol
/// version) as routing hints before the gRPC capability stream is available.
/// The `proto` key lets the SDK exclude nodes speaking an unsupported
/// data-plane major before dialing them.
#[derive(Debug, Clone, Default)]
pub struct AdvertisedCapabilities {
    pub tasks: Vec<String>,
    pub runtime: Option<String>,
    pub protocol_version: Option<String>,
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
        capabilities: &AdvertisedCapabilities,
    ) -> DaemonResult<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }

        let hostname = mdns_hostname();
        let default_name = default_instance_name();
        let instance_name = config.service_name.as_deref().unwrap_or(&default_name);
        let properties = mdns_properties(capabilities);

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
            tasks = %capabilities.tasks.join(","),
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

fn mdns_properties(capabilities: &AdvertisedCapabilities) -> HashMap<String, String> {
    let version =
        env::var("SERVICE_VERSION").unwrap_or_else(|_| DEFAULT_SERVICE_VERSION.to_owned());

    let mut properties = HashMap::from([
        (
            "uuid".to_owned(),
            env::var("SERVICE_UUID").unwrap_or_else(|_| format!("lumnn-{}", std::process::id())),
        ),
        (
            "status".to_owned(),
            env::var("SERVICE_STATUS").unwrap_or_else(|_| "ready".to_owned()),
        ),
        // `version` is legacy; `v` is what lumen-sdk reads.
        ("version".to_owned(), version.clone()),
        ("v".to_owned(), version),
    ]);
    if let Some(runtime) = capabilities
        .runtime
        .as_deref()
        .filter(|runtime| !runtime.is_empty())
    {
        properties.insert("runtime".to_owned(), runtime.to_owned());
    }
    if !capabilities.tasks.is_empty() {
        properties.insert("tasks".to_owned(), capabilities.tasks.join(","));
    }
    if let Some(protocol_version) = capabilities
        .protocol_version
        .as_deref()
        .filter(|protocol_version| !protocol_version.is_empty())
    {
        // `proto` is the data-plane protocol version (e.g. "1.0"); lumen-sdk
        // uses its major to filter nodes before connecting.
        properties.insert("proto".to_owned(), protocol_version.to_owned());
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
    fn mdns_properties_advertise_sdk_contract_keys() {
        let capabilities = AdvertisedCapabilities {
            tasks: vec!["semantic_image_embed".to_owned(), "ocr".to_owned()],
            runtime: Some("burn".to_owned()),
            protocol_version: Some("1.0".to_owned()),
        };
        let properties = mdns_properties(&capabilities);

        assert_eq!(
            properties.get("tasks").map(String::as_str),
            Some("semantic_image_embed,ocr")
        );
        assert_eq!(properties.get("runtime").map(String::as_str), Some("burn"));
        assert_eq!(properties.get("proto").map(String::as_str), Some("1.0"));
        // `v` mirrors the legacy `version` key; lumen-sdk reads `v`.
        assert_eq!(properties.get("v"), properties.get("version"));
        assert!(properties.contains_key("uuid"));
        assert!(properties.contains_key("status"));
    }

    #[test]
    fn mdns_properties_omit_empty_task_and_runtime_keys() {
        let properties = mdns_properties(&AdvertisedCapabilities::default());

        assert!(!properties.contains_key("tasks"));
        assert!(!properties.contains_key("runtime"));
        assert!(!properties.contains_key("proto"));
        assert!(properties.contains_key("v"));
    }
}
