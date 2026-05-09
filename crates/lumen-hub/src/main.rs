use std::{
    env,
    ffi::OsString,
    fmt, fs,
    path::PathBuf,
    process,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(feature = "clip")]
use lumen_hub::models::clip::ClipService;
#[cfg(feature = "siglip")]
use lumen_hub::models::siglip::SiglipService;
use lumen_hub::{
    daemon::{DaemonError, bind_addr, serve_grpc_with_shutdown},
    service::ServiceHub,
};
use lumen_schema::{ConfigValidationError, LumenConfig, Mode, ServerConfig};
#[cfg(any(feature = "clip", feature = "siglip"))]
use lumnn::core::context::{MLContext, MLContextOptions};
use thiserror::Error;
use tracing::{
    Event, Level, Metadata, Subscriber,
    field::{Field, Visit},
    level_filters::LevelFilter,
    span::{Attributes, Id, Record},
    subscriber::{Interest, SetGlobalDefaultError, set_global_default},
};

#[tokio::main]
async fn main() {
    match run(env::args_os()).await {
        Ok(()) => {}
        Err(StartupError::Help) => {
            print_usage();
        }
        Err(error) => {
            eprintln!("error: {error}");
            process::exit(1);
        }
    }
}

async fn run<I>(args: I) -> StartupResult<()>
where
    I: IntoIterator<Item = OsString>,
{
    let cli = parse_args(args)?;
    init_logging(cli.log_level)?;
    let config = load_config(&cli.config_path)?;

    if config.deployment.mode != Mode::Hub {
        return Err(StartupError::InvalidDeploymentMode {
            mode: config.deployment.mode,
        });
    }

    let hub = build_service_hub_from_config(&config)?;
    let server_config = server_config_with_override(&config.server, cli.port_override);
    let addr = bind_addr(&server_config)?;

    println!(
        "Lumen Hub service listening on {addr} with {} service(s)",
        hub.len()
    );

    serve_grpc_with_shutdown(Arc::new(hub), &server_config, shutdown_signal()).await?;
    Ok(())
}

fn parse_args<I, S>(args: I) -> StartupResult<CliArgs>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter().map(Into::into);
    let _program = args.next();
    let mut config_path = None;
    let mut port_override = None;
    let mut log_level = LogLevel::Info;

    while let Some(arg) = args.next() {
        let arg = arg.into_string().map_err(|_| {
            StartupError::InvalidArgument("arguments must be valid UTF-8".to_owned())
        })?;

        match arg.as_str() {
            "-h" | "--help" => return Err(StartupError::Help),
            "--config" => {
                let value = next_arg_value(&mut args, "--config")?;
                config_path = Some(PathBuf::from(value));
            }
            "--port" => {
                let value = next_arg_value(&mut args, "--port")?;
                port_override = Some(parse_port(&value)?);
            }
            "--log-level" => {
                let value = next_arg_value(&mut args, "--log-level")?;
                log_level = value.parse()?;
            }
            _ if arg.starts_with("--config=") => {
                config_path = Some(PathBuf::from(arg.trim_start_matches("--config=")));
            }
            _ if arg.starts_with("--port=") => {
                port_override = Some(parse_port(arg.trim_start_matches("--port="))?);
            }
            _ if arg.starts_with("--log-level=") => {
                log_level = arg.trim_start_matches("--log-level=").parse()?;
            }
            _ => {
                return Err(StartupError::InvalidArgument(format!(
                    "unknown argument `{arg}`"
                )));
            }
        }
    }

    let config_path = config_path.ok_or_else(|| {
        StartupError::InvalidArgument("missing required argument `--config <path>`".to_owned())
    })?;

    Ok(CliArgs {
        config_path,
        port_override,
        log_level,
    })
}

fn next_arg_value<I>(args: &mut I, flag: &str) -> StartupResult<String>
where
    I: Iterator<Item = OsString>,
{
    args.next()
        .ok_or_else(|| StartupError::InvalidArgument(format!("missing value for `{flag}`")))?
        .into_string()
        .map_err(|_| StartupError::InvalidArgument(format!("value for `{flag}` must be UTF-8")))
}

fn parse_port(value: &str) -> StartupResult<u16> {
    let port = value
        .parse::<u16>()
        .map_err(|_| StartupError::InvalidArgument(format!("invalid port `{value}`")))?;

    if port < 1024 {
        return Err(StartupError::InvalidArgument(format!(
            "port `{port}` must be >= 1024"
        )));
    }

    Ok(port)
}

fn load_config(path: &PathBuf) -> StartupResult<LumenConfig> {
    let contents = fs::read_to_string(path).map_err(|source| StartupError::ReadConfig {
        path: path.clone(),
        source,
    })?;
    Ok(LumenConfig::from_json_str(&contents)?)
}

fn build_service_hub_from_config(config: &LumenConfig) -> StartupResult<ServiceHub> {
    #[cfg(any(feature = "clip", feature = "siglip"))]
    let context = MLContext::new(MLContextOptions::default()).map_err(StartupError::ContextInit)?;

    #[cfg(any(feature = "clip", feature = "siglip"))]
    let cache_dir = expand_tilde(&config.metadata.cache_dir);

    let requested_services = config
        .deployment_service_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();

    #[cfg_attr(not(any(feature = "clip", feature = "siglip")), allow(unused_mut))]
    let mut hub = ServiceHub::new();

    for service_name in &requested_services {
        let svc_config = config.services.get(service_name).ok_or_else(|| {
            StartupError::InvalidArgument(format!(
                "deployment references unknown service `{service_name}`"
            ))
        })?;

        match svc_config.package.as_str() {
            #[cfg(feature = "clip")]
            "clip" | "lumen_clip" => {
                let service = ClipService::from_config(
                    service_name,
                    svc_config,
                    &cache_dir,
                    Arc::clone(&context),
                )
                .map_err(|e| StartupError::ServiceConstruction {
                    service: service_name.to_owned(),
                    message: e.to_string(),
                })?;
                hub.register(service)
                    .map_err(|e| StartupError::ServiceConstruction {
                        service: service_name.to_owned(),
                        message: e.to_string(),
                    })?;
            }
            #[cfg(not(feature = "clip"))]
            "clip" | "lumen_clip" => {
                return Err(StartupError::PackageDisabled {
                    package: svc_config.package.clone(),
                    feature: "clip",
                });
            }
            #[cfg(feature = "siglip")]
            "siglip" | "lumen_siglip" => {
                let service = SiglipService::from_config(
                    service_name,
                    svc_config,
                    &cache_dir,
                    Arc::clone(&context),
                )
                .map_err(|e| StartupError::ServiceConstruction {
                    service: service_name.to_owned(),
                    message: e.to_string(),
                })?;
                hub.register(service)
                    .map_err(|e| StartupError::ServiceConstruction {
                        service: service_name.to_owned(),
                        message: e.to_string(),
                    })?;
            }
            #[cfg(not(feature = "siglip"))]
            "siglip" | "lumen_siglip" => {
                return Err(StartupError::PackageDisabled {
                    package: svc_config.package.clone(),
                    feature: "siglip",
                });
            }
            other => {
                return Err(StartupError::UnknownPackage {
                    package: other.to_owned(),
                });
            }
        }
    }

    Ok(hub)
}

#[cfg(any(feature = "clip", feature = "siglip", test))]
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix('~')
        && let Ok(home) = env::var("HOME")
    {
        let mut expanded = home;
        expanded.push_str(rest);
        return expanded;
    }
    path.to_owned()
}

fn init_logging(level: LogLevel) -> StartupResult<()> {
    set_global_default(SimpleSubscriber::new(level))?;
    Ok(())
}

fn server_config_with_override(config: &ServerConfig, port_override: Option<u16>) -> ServerConfig {
    let mut config = config.clone();
    if let Some(port) = port_override {
        config.port = port;
    }
    config
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");

        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                if let Err(error) = result {
                    eprintln!("failed to listen for Ctrl-C: {error}");
                }
            }
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        if let Err(error) = tokio::signal::ctrl_c().await {
            eprintln!("failed to listen for Ctrl-C: {error}");
        }
    }

    println!("shutdown signal received");
}

fn print_usage() {
    println!(
        "\
Usage:
  lumen-hub --config <path> [--port <port>] [--log-level <level>]

Options:
  --config <path>       Path to lumen-config JSON file.
  --port <port>         Override config.server.port.
  --log-level <level>   DEBUG, INFO, WARNING, ERROR, or CRITICAL. Default: INFO.
  -h, --help            Show this help text.
"
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    config_path: PathBuf,
    port_override: Option<u16>,
    log_level: LogLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl std::str::FromStr for LogLevel {
    type Err = StartupError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "DEBUG" | "debug" => Ok(Self::Debug),
            "INFO" | "info" => Ok(Self::Info),
            "WARNING" | "warning" | "WARN" | "warn" => Ok(Self::Warning),
            "ERROR" | "error" => Ok(Self::Error),
            "CRITICAL" | "critical" => Ok(Self::Critical),
            _ => Err(StartupError::InvalidArgument(format!(
                "invalid log level `{value}`"
            ))),
        }
    }
}

impl LogLevel {
    fn enables(self, level: &Level) -> bool {
        level_rank(level) <= self.max_rank()
    }

    fn max_rank(self) -> u8 {
        match self {
            Self::Critical | Self::Error => 1,
            Self::Warning => 2,
            Self::Info => 3,
            Self::Debug => 4,
        }
    }

    fn level_filter(self) -> LevelFilter {
        match self {
            Self::Critical | Self::Error => LevelFilter::ERROR,
            Self::Warning => LevelFilter::WARN,
            Self::Info => LevelFilter::INFO,
            Self::Debug => LevelFilter::DEBUG,
        }
    }
}

struct SimpleSubscriber {
    max_level: LogLevel,
    next_span_id: AtomicU64,
}

impl SimpleSubscriber {
    fn new(max_level: LogLevel) -> Self {
        Self {
            max_level,
            next_span_id: AtomicU64::new(1),
        }
    }
}

impl Subscriber for SimpleSubscriber {
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if self.enabled(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.max_level.enables(metadata.level())
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(self.max_level.level_filter())
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(self.next_span_id.fetch_add(1, Ordering::Relaxed))
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event_enabled(&self, event: &Event<'_>) -> bool {
        self.enabled(event.metadata())
    }

    fn event(&self, event: &Event<'_>) {
        let metadata = event.metadata();
        if !self.enabled(metadata) {
            return;
        }

        let mut fields = LogFields::default();
        event.record(&mut fields);

        if let Some(message) = fields.message {
            if fields.fields.is_empty() {
                eprintln!("{} [{}] {}", metadata.level(), metadata.target(), message);
            } else {
                eprintln!(
                    "{} [{}] {} {}",
                    metadata.level(),
                    metadata.target(),
                    message,
                    fields.fields.join(" ")
                );
            }
        } else {
            eprintln!(
                "{} [{}] {}",
                metadata.level(),
                metadata.target(),
                fields.fields.join(" ")
            );
        }
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[derive(Default)]
struct LogFields {
    message: Option<String>,
    fields: Vec<String>,
}

impl LogFields {
    fn record_value(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            self.fields.push(format!("{}={value}", field.name()));
        }
    }
}

impl Visit for LogFields {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_owned());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}

fn level_rank(level: &Level) -> u8 {
    match *level {
        Level::ERROR => 1,
        Level::WARN => 2,
        Level::INFO => 3,
        Level::DEBUG => 4,
        Level::TRACE => 5,
    }
}

type StartupResult<T> = Result<T, StartupError>;

#[derive(Debug, Error)]
enum StartupError {
    #[error("help requested")]
    Help,

    #[error("{0}")]
    InvalidArgument(String),

    #[error("failed to read config `{}`: {source}", path.display())]
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("invalid config: {0}")]
    Config(#[from] ConfigValidationError),

    #[error("failed to initialize logging: {0}")]
    Logging(#[from] SetGlobalDefaultError),

    #[cfg(any(feature = "clip", feature = "siglip"))]
    #[error("failed to initialize ML context: {0}")]
    ContextInit(String),

    #[error("this binary currently supports hub deployment mode only, got {mode:?}")]
    InvalidDeploymentMode { mode: Mode },

    #[error("unknown service package `{package}`")]
    UnknownPackage { package: String },

    #[cfg(any(not(feature = "clip"), not(feature = "siglip")))]
    #[error(
        "service package `{package}` was not enabled at compile time; rebuild with feature `{feature}`"
    )]
    PackageDisabled {
        package: String,
        feature: &'static str,
    },

    #[cfg(any(feature = "clip", feature = "siglip"))]
    #[error("failed to construct service `{service}`: {message}")]
    ServiceConstruction { service: String, message: String },

    #[error("{0}")]
    Daemon(#[from] DaemonError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_required_config_and_overrides() {
        let args = parse_args([
            "lumen-hub",
            "--config",
            "config/lumen-config.json",
            "--port=50052",
            "--log-level",
            "DEBUG",
        ])
        .unwrap();

        assert_eq!(args.config_path, PathBuf::from("config/lumen-config.json"));
        assert_eq!(args.port_override, Some(50_052));
        assert_eq!(args.log_level, LogLevel::Debug);
    }

    #[test]
    fn parse_args_requires_config() {
        let err = parse_args(["lumen-hub"]).unwrap_err();

        assert!(matches!(err, StartupError::InvalidArgument(_)));
    }

    #[test]
    fn parse_port_rejects_privileged_ports() {
        let err = parse_port("80").unwrap_err();

        assert!(matches!(err, StartupError::InvalidArgument(_)));
    }

    #[test]
    fn server_config_with_override_updates_port_only() {
        let config = ServerConfig {
            port: 50_051,
            host: "127.0.0.1".to_owned(),
            mdns: None,
            batching: Default::default(),
        };

        let overridden = server_config_with_override(&config, Some(50_052));

        assert_eq!(overridden.port, 50_052);
        assert_eq!(overridden.host, "127.0.0.1");
    }

    #[test]
    fn expand_tilde_replaces_home() {
        let home = env::var("HOME").unwrap();
        let expanded = expand_tilde("~/.lumen/models");
        assert_eq!(expanded, format!("{home}/.lumen/models"));
    }

    #[test]
    fn expand_tilde_passthrough_non_tilde() {
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
    }
}
