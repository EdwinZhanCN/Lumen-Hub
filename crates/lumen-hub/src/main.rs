use std::{
    env,
    ffi::OsString,
    fmt, fs,
    path::{Path, PathBuf},
    process,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use console::style;
#[cfg(feature = "clip")]
use lumen_hub::models::bioclip::BioclipService;
#[cfg(feature = "insightface")]
use lumen_hub::models::insightface::InsightFaceService;
#[cfg(feature = "ppocr")]
use lumen_hub::models::ppocr::PpocrService;
#[cfg(feature = "siglip")]
use lumen_hub::models::siglip::SiglipService;
use lumen_hub::{
    backend::{BACKEND_NAME, configure_runtime, default_device},
    daemon::{
        DaemonError, HubGrpcService, MdnsAdvertisement, ReadyHandle, advertised_metadata,
        bind_addr, control_plane, hub_batcher_config,
    },
    docker_config::{DockerConfigError, DockerConfigInput},
    inference_worker,
    model_download::{DownloadObserver, ModelDownloadError, ensure_models_for_config_observed},
    service::ServiceHub,
    status::{self, DownloadProgress, LogBuffer, Phase, ServiceState, StatusBus},
    warmup::{WarmupError, default_warmup_dir, run_startup_warmup},
};
#[cfg(test)]
use lumen_schema::Mdns;
use lumen_schema::{ConfigValidationError, LumenConfig, Mode, ServerConfig};
use thiserror::Error;
use tonic_health::pb::{
    HealthCheckRequest, health_check_response::ServingStatus, health_client::HealthClient,
};
use tracing::{
    Event, Level, Metadata, Subscriber,
    field::{Field, Visit},
    info,
    level_filters::LevelFilter,
    span::{Attributes, Id, Record},
    subscriber::{Interest, SetGlobalDefaultError, set_global_default},
};

/// Stack size for the runtime driver thread and the inference worker.
///
/// The ONNX-generated model `new`/`forward` functions are monolithic with very
/// large stack frames; on GPU backends (where tensor handles are larger than
/// Flex CPU's) the default ~2 MB thread stack overflows. Model construction runs on
/// the dedicated inference worker, and the Tokio runtime still drives async IO,
/// so both need a generous stack.
const RUNTIME_STACK_SIZE: usize = 256 * 1024 * 1024;

fn main() {
    let args = env::args_os().collect::<Vec<_>>();
    if is_healthcheck_command(&args) {
        run_healthcheck_command(args);
        return;
    }

    configure_runtime();

    // Drive the runtime from a thread with a large stack so model construction
    // around startup and async orchestration does not overflow.
    let worker = std::thread::Builder::new()
        .stack_size(RUNTIME_STACK_SIZE)
        .spawn(move || run_main(args))
        .expect("failed to spawn runtime thread");
    worker.join().expect("runtime thread panicked");
}

fn run_main(args: Vec<OsString>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(RUNTIME_STACK_SIZE)
        .build()
        .expect("failed to build tokio runtime");

    runtime.block_on(async {
        match run(args).await {
            Ok(()) => {}
            Err(StartupError::Help) => {
                print_usage();
            }
            Err(error) => {
                eprintln!("error: {error}");
                process::exit(1);
            }
        }
    });
}

fn is_healthcheck_command(args: &[OsString]) -> bool {
    args.get(1).and_then(|arg| arg.to_str()) == Some("healthcheck")
}

fn run_healthcheck_command(args: Vec<OsString>) {
    let result = parse_healthcheck_args(args).and_then(|args| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| StartupError::HealthRuntime(error.to_string()))?
            .block_on(check_health(&args.endpoint))
    });

    if let Err(error) = result {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

async fn check_health(endpoint: &str) -> StartupResult<()> {
    let channel = tonic::transport::Endpoint::new(endpoint.to_owned())
        .map_err(|source| StartupError::HealthConnect {
            endpoint: endpoint.to_owned(),
            source,
        })?
        .connect()
        .await
        .map_err(|source| StartupError::HealthConnect {
            endpoint: endpoint.to_owned(),
            source,
        })?;
    let mut client = HealthClient::new(channel);
    let status = client
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await?
        .into_inner()
        .status;
    let status = ServingStatus::try_from(status).unwrap_or(ServingStatus::Unknown);

    if status == ServingStatus::Serving {
        Ok(())
    } else {
        Err(StartupError::HealthNotServing { status })
    }
}

async fn run<I>(args: I) -> StartupResult<()>
where
    I: IntoIterator<Item = OsString>,
{
    let cli = parse_args(args)?;
    let logs = Arc::new(LogBuffer::new());
    init_logging(cli.log_level, Arc::clone(&logs))?;
    info!("lumen-hub starting");
    info!(config = %cli.config_path.display(), "loading config");
    let mut config = load_config(&cli.config_path)?;
    let docker_input = DockerConfigInput::from_process_env()?;
    if !docker_input.is_empty() {
        docker_input.apply(&mut config)?;
        info!(
            region = ?config.metadata.region,
            services = %config.deployment_service_names().join(", "),
            "applied Docker environment config"
        );
    }

    if config.deployment.mode != Mode::Hub {
        return Err(StartupError::InvalidDeploymentMode {
            mode: config.deployment.mode,
        });
    }

    let cache_dir = expand_tilde(&config.metadata.cache_dir);
    info!(
        region = ?config.metadata.region,
        cache = %cache_dir,
        services = %config.deployment_service_names().join(", "),
        "startup plan"
    );

    let server_config = server_config_with_override(&config.server, cli.port_override);
    let addr = bind_addr(&server_config)?;

    // The control plane binds before any model work so supervisors can watch
    // download progress and errors over gRPC instead of tailing logs. The
    // data plane answers UNAVAILABLE until initialization completes.
    let bus = Arc::new(StatusBus::new(
        env!("CARGO_PKG_VERSION").to_owned(),
        BACKEND_NAME.to_owned(),
    ));
    let (server, ready) = control_plane(Arc::clone(&bus), logs);
    ready.init().await;

    // mDNS is advertised only once the hub is ready; the guard lives here so
    // the advertisement survives until the server stops.
    let mdns_guard: Arc<Mutex<Option<MdnsAdvertisement>>> = Arc::default();

    {
        let bus = Arc::clone(&bus);
        let config = config.clone();
        let cache_dir = cache_dir.clone();
        let server_config = server_config.clone();
        let mdns_guard = Arc::clone(&mdns_guard);
        let port = addr.port();
        tokio::spawn(async move {
            match initialize(
                &bus,
                &config,
                &cache_dir,
                &server_config,
                port,
                &ready,
                &mdns_guard,
            )
            .await
            {
                Ok(()) => info!("lumen-hub ready"),
                Err(error) => {
                    // Keep serving: the control plane stays queryable so the
                    // failure (and its cause) is visible to supervisors;
                    // health reports NOT_SERVING.
                    tracing::error!(%error, "lumen-hub startup failed");
                    ready.set_failed().await;
                    bus.fail(error.to_string());
                }
            }
        });
    }

    info!(addr = %addr, "control plane listening");
    let shutdown = {
        let bus = Arc::clone(&bus);
        async move {
            shutdown_signal().await;
            bus.set_phase(Phase::Stopping);
        }
    };
    server.serve_with_shutdown(addr, shutdown).await?;
    Ok(())
}

/// Download → build → warmup → ready, publishing each transition on the bus.
async fn initialize(
    bus: &Arc<StatusBus>,
    config: &LumenConfig,
    cache_dir: &str,
    server_config: &ServerConfig,
    port: u16,
    ready: &ReadyHandle,
    mdns_guard: &Mutex<Option<MdnsAdvertisement>>,
) -> StartupResult<()> {
    bus.set_phase(Phase::Downloading);
    info!("ensuring model cache");
    {
        // ureq downloads are blocking; keep them off the async runtime.
        let config = config.clone();
        let cache_dir = cache_dir.to_owned();
        let observer_bus = Arc::clone(bus);
        tokio::task::spawn_blocking(move || {
            let observer = BusDownloadObserver::new(observer_bus);
            ensure_models_for_config_observed(&config, &cache_dir, &observer)
        })
        .await
        .expect("model download task panicked")?;
    }

    bus.set_phase(Phase::Loading);
    info!("building service hub");
    let hub_config = config.clone();
    let hub_cache_dir = cache_dir.to_owned();
    let hub =
        inference_worker::run(move || build_service_hub_from_config(&hub_config, &hub_cache_dir))
            .await??;

    bus.set_phase(Phase::Warmup);
    info!("running mandatory warmup");
    run_startup_warmup(&hub, &default_warmup_dir()).await?;

    let hub = Arc::new(hub);
    bus.set_services(
        hub.capabilities()
            .into_iter()
            .map(|capability| ServiceState {
                service: capability.service_name,
                phase: Phase::Ready,
                error: String::new(),
            })
            .collect(),
    );

    let advertisement =
        MdnsAdvertisement::register(&server_config.mdns, port, &advertised_metadata(&hub))?;
    *mdns_guard.lock().expect("mdns guard poisoned") = advertisement;

    ready
        .set_ready(HubGrpcService::new(
            Arc::clone(&hub),
            hub_batcher_config(server_config),
        ))
        .await;
    bus.set_phase(Phase::Ready);
    Ok(())
}

/// Bridges download progress onto the status bus, tracking per-model file
/// counts (the byte counter is throttled by the bus itself).
struct BusDownloadObserver {
    bus: Arc<StatusBus>,
    counts: Mutex<(String, u32, u32)>, // (model, files_done, files_total)
}

impl BusDownloadObserver {
    fn new(bus: Arc<StatusBus>) -> Self {
        Self {
            bus,
            counts: Mutex::new((String::new(), 0, 0)),
        }
    }

    fn progress(
        &self,
        model: &str,
        file: &str,
        bytes_done: u64,
        bytes_total: u64,
    ) -> DownloadProgress {
        let counts = self.counts.lock().expect("download counts poisoned");
        DownloadProgress {
            model: model.to_owned(),
            file: file.to_owned(),
            bytes_done,
            bytes_total,
            files_done: counts.1,
            files_total: counts.2,
        }
    }
}

impl DownloadObserver for BusDownloadObserver {
    fn on_model_plan(&self, model: &str, files_total: u32) {
        *self.counts.lock().expect("download counts poisoned") = (model.to_owned(), 0, files_total);
    }

    fn on_file_progress(&self, model: &str, file: &str, bytes_done: u64, bytes_total: u64) {
        self.bus
            .set_download(self.progress(model, file, bytes_done, bytes_total));
    }

    fn on_file_complete(&self, model: &str, file: &str) {
        {
            let mut counts = self.counts.lock().expect("download counts poisoned");
            if counts.0 == model {
                counts.1 = counts.1.saturating_add(1);
            }
        }
        self.bus.set_download(self.progress(model, file, 0, 0));
    }
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

fn parse_healthcheck_args<I, S>(args: I) -> StartupResult<HealthcheckArgs>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter().map(Into::into);
    let _program = args.next();
    let command = args.next().and_then(|arg| arg.into_string().ok());
    if command.as_deref() != Some("healthcheck") {
        return Err(StartupError::InvalidArgument(
            "expected `healthcheck` command".to_owned(),
        ));
    }

    let mut endpoint = "http://127.0.0.1:50051".to_owned();
    while let Some(arg) = args.next() {
        let arg = arg.into_string().map_err(|_| {
            StartupError::InvalidArgument("arguments must be valid UTF-8".to_owned())
        })?;

        match arg.as_str() {
            "--endpoint" => endpoint = next_arg_value(&mut args, "--endpoint")?,
            _ if arg.starts_with("--endpoint=") => {
                endpoint = arg.trim_start_matches("--endpoint=").to_owned();
            }
            _ => {
                return Err(StartupError::InvalidArgument(format!(
                    "unknown healthcheck argument `{arg}`"
                )));
            }
        }
    }

    if !(endpoint.starts_with("http://") || endpoint.starts_with("https://")) {
        return Err(StartupError::InvalidArgument(
            "healthcheck endpoint must start with http:// or https://".to_owned(),
        ));
    }

    Ok(HealthcheckArgs { endpoint })
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
    if is_yaml_path(path) {
        let config = serde_yaml::from_str::<LumenConfig>(&contents).map_err(|source| {
            StartupError::YamlConfig {
                path: path.clone(),
                source,
            }
        })?;
        config.validate_config()?;
        Ok(config)
    } else {
        Ok(LumenConfig::from_json_str(&contents)?)
    }
}

fn is_yaml_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension, "yaml" | "yml"))
}

fn build_service_hub_from_config(
    config: &LumenConfig,
    cache_dir: &str,
) -> StartupResult<ServiceHub> {
    #[cfg_attr(
        not(any(
            feature = "clip",
            feature = "insightface",
            feature = "ppocr",
            feature = "siglip"
        )),
        allow(unused_variables)
    )]
    let device = Arc::new(default_device());

    let requested_services = config
        .deployment_service_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();

    #[cfg_attr(
        not(any(
            feature = "clip",
            feature = "insightface",
            feature = "ppocr",
            feature = "siglip"
        )),
        allow(unused_mut)
    )]
    let mut hub = ServiceHub::new();

    for service_name in &requested_services {
        let svc_config = config.services.get(service_name).ok_or_else(|| {
            StartupError::InvalidArgument(format!(
                "deployment references unknown service `{service_name}`"
            ))
        })?;
        info!(
            service = %service_name,
            package = %svc_config.package,
            models = svc_config.models.len(),
            "loading service"
        );

        match svc_config.package.as_str() {
            #[cfg(feature = "clip")]
            "clip" | "lumen_clip" | "bioclip" => {
                let service = BioclipService::from_config(
                    service_name,
                    svc_config,
                    cache_dir,
                    Arc::clone(&device),
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
            "clip" | "lumen_clip" | "bioclip" => {
                return Err(StartupError::PackageDisabled {
                    package: svc_config.package.clone(),
                    feature: "clip",
                });
            }
            #[cfg(feature = "insightface")]
            "insightface" | "lumen_insightface" => {
                let service = InsightFaceService::from_config(
                    service_name,
                    svc_config,
                    cache_dir,
                    Arc::clone(&device),
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
            #[cfg(not(feature = "insightface"))]
            "insightface" | "lumen_insightface" => {
                return Err(StartupError::PackageDisabled {
                    package: svc_config.package.clone(),
                    feature: "insightface",
                });
            }
            #[cfg(feature = "siglip")]
            "siglip" | "lumen_siglip" => {
                let service = SiglipService::from_config(
                    service_name,
                    svc_config,
                    cache_dir,
                    Arc::clone(&device),
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
            #[cfg(feature = "ppocr")]
            "ppocr" | "lumen_ppocr" => {
                let service = PpocrService::from_config(
                    service_name,
                    svc_config,
                    cache_dir,
                    Arc::clone(&device),
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
            #[cfg(not(feature = "ppocr"))]
            "ppocr" | "lumen_ppocr" => {
                return Err(StartupError::PackageDisabled {
                    package: svc_config.package.clone(),
                    feature: "ppocr",
                });
            }
            #[cfg(feature = "qa")]
            "qa" => {
                let service = lumen_hub::models::qa::QaService::from_config(
                    service_name,
                    svc_config,
                    &cache_dir,
                    Arc::clone(&device),
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
            #[cfg(not(feature = "qa"))]
            "qa" => {
                return Err(StartupError::PackageDisabled {
                    package: svc_config.package.clone(),
                    feature: "qa",
                });
            }
            other => {
                return Err(StartupError::UnknownPackage {
                    package: other.to_owned(),
                });
            }
        }
        info!(service = %service_name, "service ready");
    }

    Ok(hub)
}

fn home_dir() -> Option<String> {
    env::var("HOME")
        .ok()
        .or_else(|| env::var("USERPROFILE").ok())
        .or_else(|| {
            let drive = env::var("HOMEDRIVE").ok()?;
            let path = env::var("HOMEPATH").ok()?;
            Some(format!("{drive}{path}"))
        })
}

fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix('~')
        && let Some(home) = home_dir()
    {
        let mut expanded = home;
        expanded.push_str(rest);
        return expanded;
    }
    path.to_owned()
}

fn init_logging(level: LogLevel, sink: Arc<LogBuffer>) -> StartupResult<()> {
    set_global_default(SimpleSubscriber::new(level, sink))?;
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

    info!("shutdown signal received");
}

fn print_usage() {
    println!(
        "\
Usage:
  lumen-hub --config <path> [--port <port>] [--log-level <level>]
  lumen-hub healthcheck [--endpoint <url>]

Options:
  --config <path>       Path to lumen-config JSON file.
  --port <port>         Override config.server.port.
  --log-level <level>   DEBUG, INFO, WARNING, ERROR, or CRITICAL. Default: INFO.
  --endpoint <url>      gRPC endpoint to probe. Default: http://127.0.0.1:50051.
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct HealthcheckArgs {
    endpoint: String,
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
    /// Structured copy of every emitted event, served by `Control.TailLogs`.
    sink: Arc<LogBuffer>,
}

impl SimpleSubscriber {
    fn new(max_level: LogLevel, sink: Arc<LogBuffer>) -> Self {
        Self {
            max_level,
            next_span_id: AtomicU64::new(1),
            sink,
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
        let level = styled_level(metadata.level());
        let target = style(metadata.target()).dim();
        let rendered_fields = fields
            .fields
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>();

        if let Some(message) = &fields.message {
            if rendered_fields.is_empty() {
                eprintln!("{level} [{target}] {message}");
            } else {
                eprintln!(
                    "{level} [{target}] {message} {}",
                    style(rendered_fields.join(" ")).dim()
                );
            }
        } else {
            eprintln!(
                "{level} [{target}] {}",
                style(rendered_fields.join(" ")).dim()
            );
        }

        self.sink.push(status::LogEntry {
            time_unix_ms: status::unix_ms_now(),
            level: status_level(metadata.level()),
            target: metadata.target().to_owned(),
            message: fields.message.unwrap_or_default(),
            fields: fields.fields,
        });
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[derive(Default)]
struct LogFields {
    message: Option<String>,
    fields: Vec<(String, String)>,
}

impl LogFields {
    fn record_value(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            self.fields.push((field.name().to_owned(), value));
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

fn status_level(level: &Level) -> status::LogLevel {
    match *level {
        Level::ERROR => status::LogLevel::Error,
        Level::WARN => status::LogLevel::Warn,
        Level::INFO => status::LogLevel::Info,
        Level::DEBUG => status::LogLevel::Debug,
        Level::TRACE => status::LogLevel::Trace,
    }
}

fn styled_level(level: &Level) -> console::StyledObject<&'static str> {
    match *level {
        Level::ERROR => style("ERROR").red().bold(),
        Level::WARN => style("WARN ").yellow().bold(),
        Level::INFO => style("INFO ").green(),
        Level::DEBUG => style("DEBUG").blue(),
        Level::TRACE => style("TRACE").dim(),
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

    #[error("failed to parse YAML config `{}`: {source}", path.display())]
    YamlConfig {
        path: PathBuf,
        source: serde_yaml::Error,
    },

    #[error("invalid config: {0}")]
    Config(#[from] ConfigValidationError),

    #[error("invalid Docker environment config: {0}")]
    DockerConfig(#[from] DockerConfigError),

    #[error("model download failed: {0}")]
    ModelDownload(#[from] ModelDownloadError),

    #[error("startup warmup failed: {0}")]
    Warmup(#[from] WarmupError),

    #[error("inference worker failed: {0}")]
    InferenceWorker(#[from] inference_worker::InferenceWorkerError),

    #[error("failed to initialize logging: {0}")]
    Logging(#[from] SetGlobalDefaultError),

    #[error("failed to create healthcheck runtime: {0}")]
    HealthRuntime(String),

    #[error("failed to connect to Lumen Hub health endpoint `{endpoint}`: {source}")]
    HealthConnect {
        endpoint: String,
        source: tonic::transport::Error,
    },

    #[error("healthcheck RPC failed: {0}")]
    HealthRpc(#[from] tonic::Status),

    #[error("Lumen Hub is not serving ({status:?})")]
    HealthNotServing { status: ServingStatus },

    #[error("this binary currently supports hub deployment mode only, got {mode:?}")]
    InvalidDeploymentMode { mode: Mode },

    #[error("unknown service package `{package}`")]
    UnknownPackage { package: String },

    // Only constructed when a model feature is disabled; absent from the
    // all-features default build.
    #[allow(dead_code)]
    #[error(
        "service package `{package}` was not enabled at compile time; rebuild with feature `{feature}`"
    )]
    PackageDisabled {
        package: String,
        feature: &'static str,
    },

    #[cfg(any(
        feature = "clip",
        feature = "insightface",
        feature = "ppocr",
        feature = "siglip"
    ))]
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
    fn parse_healthcheck_args_uses_default_endpoint() {
        let args = parse_healthcheck_args(["lumen-hub", "healthcheck"]).unwrap();

        assert_eq!(args.endpoint, "http://127.0.0.1:50051");
    }

    #[test]
    fn parse_healthcheck_args_accepts_endpoint_override() {
        let args = parse_healthcheck_args([
            "lumen-hub",
            "healthcheck",
            "--endpoint=http://192.0.2.1:50052",
        ])
        .unwrap();

        assert_eq!(args.endpoint, "http://192.0.2.1:50052");
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
            mdns: Mdns::default(),
            batching: Default::default(),
        };

        let overridden = server_config_with_override(&config, Some(50_052));

        assert_eq!(overridden.port, 50_052);
        assert_eq!(overridden.host, "127.0.0.1");
    }

    #[test]
    fn load_config_accepts_yaml_examples() {
        for name in [
            "minimal.yaml",
            "basic.yaml",
            "brave.yaml",
            "bench-siglip.yaml",
            "bench-bioclip.yaml",
            "bench-mixed-siglip-bioclip.yaml",
        ] {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join(name);
            load_config(&path)
                .unwrap_or_else(|error| panic!("expected `{}` to parse: {error}", path.display()));
        }

        // The default config baked into the container image must stay parseable.
        let docker_default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../packaging/docker/config.default.yaml");
        load_config(&docker_default).unwrap_or_else(|error| {
            panic!("expected `{}` to parse: {error}", docker_default.display())
        });
    }

    #[test]
    fn expand_tilde_replaces_home() {
        let home = home_dir().expect("home directory should be available in tests");
        let expanded = expand_tilde("~/.lumen/models");
        assert_eq!(expanded, format!("{home}/.lumen/models"));
    }

    #[test]
    fn expand_tilde_passthrough_non_tilde() {
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
    }
}
