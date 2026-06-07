use std::{
    collections::HashMap,
    env,
    net::{IpAddr, UdpSocket},
    sync::LazyLock,
    time::Duration,
};

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

/// Keeps an mDNS registration alive and unregisters it on drop.
pub struct MdnsAdvertisement {
    daemon: ServiceDaemon,
    fullname: String,
}

impl MdnsAdvertisement {
    pub fn register(config: &Mdns, port: u16) -> DaemonResult<Option<Self>> {
        if !config.enabled {
            return Ok(None);
        }

        let ip = advertise_ip()?;
        if ip.is_loopback() {
            tracing::warn!(
                %ip,
                "mDNS is advertising a loopback IP; set ADVERTISE_IP to a LAN IP if other devices need to connect"
            );
        }

        let hostname = mdns_hostname();
        let default_name = default_instance_name();
        let instance_name = config.service_name.as_deref().unwrap_or(&default_name);
        let service_info = ServiceInfo::new(
            DEFAULT_MDNS_SERVICE_TYPE,
            instance_name,
            &hostname,
            ip,
            port,
            mdns_properties(),
        )?;
        let fullname = service_info.get_fullname().to_owned();

        let daemon = ServiceDaemon::new()?;
        daemon.register(service_info)?;

        tracing::info!(
            service = %fullname,
            %ip,
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

fn advertise_ip() -> DaemonResult<IpAddr> {
    if let Ok(ip) = env::var("ADVERTISE_IP") {
        return ip
            .parse::<IpAddr>()
            .map_err(|source| DaemonError::InvalidAdvertiseIp { ip, source });
    }

    Ok(detect_lan_ip().unwrap_or(IpAddr::from([127, 0, 0, 1])))
}

fn detect_lan_ip() -> Option<IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    Some(socket.local_addr().ok()?.ip())
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

fn mdns_properties() -> HashMap<String, String> {
    HashMap::from([
        (
            "uuid".to_owned(),
            env::var("SERVICE_UUID").unwrap_or_else(|_| format!("lumnn-{}", std::process::id())),
        ),
        (
            "status".to_owned(),
            env::var("SERVICE_STATUS").unwrap_or_else(|_| "ready".to_owned()),
        ),
        (
            "version".to_owned(),
            env::var("SERVICE_VERSION").unwrap_or_else(|_| DEFAULT_SERVICE_VERSION.to_owned()),
        ),
    ])
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
}
