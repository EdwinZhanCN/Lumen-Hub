use std::{
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use thiserror::Error;
use tonic::transport::Endpoint;

use crate::control_proto::lumen::control::v1::{
    DownloadProgress as ProtoDownloadProgress, Phase as ProtoPhase, StatusSnapshot,
    control_client::ControlClient,
};

#[derive(Debug, Clone)]
pub struct DaemonPaths {
    pub pid_file: PathBuf,
    pub log_dir: PathBuf,
    pub log_file: PathBuf,
}

pub fn daemon_paths(lumen_dir: &Path) -> DaemonPaths {
    DaemonPaths {
        pid_file: lumen_dir.join("lumen-hub.pid"),
        log_dir: lumen_dir.join("logs"),
        log_file: lumen_dir.join("logs").join("lumen-hub.log"),
    }
}

pub fn write_pid_file(path: &Path, pid: u32) -> Result<(), DaemonError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| DaemonError::Io(path.to_path_buf(), source))?;
    }
    fs::write(path, format!("{pid}\n"))
        .map_err(|source| DaemonError::Io(path.to_path_buf(), source))?;
    Ok(())
}

pub fn read_pid_file(path: &Path) -> Result<Option<u32>, DaemonError> {
    match fs::read_to_string(path) {
        Ok(contents) => {
            let pid = contents
                .trim()
                .parse::<u32>()
                .map_err(|_| DaemonError::InvalidPidFile(path.to_path_buf()))?;
            Ok(Some(pid))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(DaemonError::Io(path.to_path_buf(), source)),
    }
}

pub fn remove_pid_file(path: &Path) -> Result<(), DaemonError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(DaemonError::Io(path.to_path_buf(), source)),
    }
}

#[cfg(unix)]
pub fn is_process_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(windows)]
pub fn is_process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return false;
        }
        CloseHandle(handle);
        true
    }
}

#[cfg(not(any(unix, windows)))]
pub fn is_process_alive(_pid: u32) -> bool {
    false
}

/// Returns `Some(pid)` if the process is alive, `None` if not running or stale.
/// Removes the PID file if the recorded process is dead.
pub fn check_running(pid_file: &Path) -> Result<Option<u32>, DaemonError> {
    let Some(pid) = read_pid_file(pid_file)? else {
        return Ok(None);
    };
    if is_process_alive(pid) {
        Ok(Some(pid))
    } else {
        remove_pid_file(pid_file)?;
        Ok(None)
    }
}

pub struct BackgroundSpawnConfig {
    pub hub_path: PathBuf,
    pub config_path: PathBuf,
    pub log_file: PathBuf,
}

/// Spawns lumen-hub as a detached background process. Returns the child PID.
pub fn spawn_background(config: &BackgroundSpawnConfig) -> Result<u32, DaemonError> {
    let log_file = prepare_log_file(&config.log_file, MAX_LOG_SIZE)?;

    let mut command = Command::new(&config.hub_path);
    command
        .arg("--config")
        .arg(&config.config_path)
        .stdin(std::process::Stdio::null())
        .stdout(
            log_file
                .try_clone()
                .map_err(|e| DaemonError::Io(config.log_file.clone(), e))?,
        )
        .stderr(log_file);

    configure_detached(&mut command);

    let child = command
        .spawn()
        .map_err(|source| DaemonError::SpawnFailed(config.hub_path.clone(), source))?;
    Ok(child.id())
}

#[cfg(unix)]
fn configure_detached(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
}

#[cfg(windows)]
fn configure_detached(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    command.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
}

#[cfg(not(any(unix, windows)))]
fn configure_detached(_command: &mut Command) {}

pub struct ReadyWaitConfig {
    pub addr: SocketAddr,
    pub timeout: Duration,
    pub interval: Duration,
}

impl Default for ReadyWaitConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:50051".parse().unwrap(),
            // First launch may download several gigabytes of models. This is
            // an upper bound, not a blind sleep: FAILED returns immediately.
            timeout: Duration::from_secs(30 * 60),
            interval: Duration::from_millis(500),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubPhase {
    Starting,
    Downloading,
    Loading,
    Warmup,
    Ready,
    Failed,
    Stopping,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubDownloadProgress {
    pub model: String,
    pub file: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub files_done: u32,
    pub files_total: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubStatus {
    pub phase: HubPhase,
    pub download: Option<HubDownloadProgress>,
    pub error: String,
    pub seq: u64,
}

/// Waits for the Hub control plane to report READY. Unlike a TCP probe, this
/// observes model download/loading/warmup and surfaces terminal startup errors.
pub fn wait_for_ready(
    config: &ReadyWaitConfig,
    on_status: impl FnMut(&HubStatus),
) -> Result<(), DaemonError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| DaemonError::ControlRuntime(error.to_string()))?;
    runtime.block_on(wait_for_ready_async(config, on_status))
}

async fn wait_for_ready_async(
    config: &ReadyWaitConfig,
    mut on_status: impl FnMut(&HubStatus),
) -> Result<(), DaemonError> {
    let endpoint = Endpoint::from_shared(format!("http://{}", config.addr))
        .map_err(|error| DaemonError::ControlEndpoint(error.to_string()))?
        .connect_timeout(Duration::from_secs(2));
    let mut client = ControlClient::new(endpoint.connect_lazy());
    let deadline = tokio::time::Instant::now() + config.timeout;
    let mut last_seq = None;

    loop {
        let Some(connect_window) = deadline.checked_duration_since(tokio::time::Instant::now())
        else {
            return Err(DaemonError::ReadyWaitTimeout(config.addr, config.timeout));
        };
        let connect_window = connect_window.min(Duration::from_secs(2));

        if let Ok(Ok(response)) =
            tokio::time::timeout(connect_window, client.watch_status(())).await
        {
            let mut updates = response.into_inner();
            loop {
                let Some(remaining) = deadline.checked_duration_since(tokio::time::Instant::now())
                else {
                    return Err(DaemonError::ReadyWaitTimeout(config.addr, config.timeout));
                };
                let snapshot = match tokio::time::timeout(remaining, updates.message()).await {
                    Ok(Ok(Some(snapshot))) => snapshot,
                    Ok(Ok(None)) | Ok(Err(_)) => break,
                    Err(_) => {
                        return Err(DaemonError::ReadyWaitTimeout(config.addr, config.timeout));
                    }
                };
                let status = hub_status(snapshot);
                if last_seq != Some(status.seq) {
                    on_status(&status);
                    last_seq = Some(status.seq);
                }
                match status.phase {
                    HubPhase::Ready => return Ok(()),
                    HubPhase::Failed => {
                        let message = if status.error.is_empty() {
                            "Lumen Hub reported a startup failure without details".to_owned()
                        } else {
                            status.error
                        };
                        return Err(DaemonError::HubStartupFailed(message));
                    }
                    HubPhase::Stopping => {
                        return Err(DaemonError::HubStartupFailed(
                            "Lumen Hub stopped before becoming ready".to_owned(),
                        ));
                    }
                    _ => {}
                }
            }
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(DaemonError::ReadyWaitTimeout(config.addr, config.timeout));
        }
        tokio::time::sleep(config.interval).await;
    }
}

fn hub_status(snapshot: StatusSnapshot) -> HubStatus {
    HubStatus {
        phase: match ProtoPhase::try_from(snapshot.phase) {
            Ok(ProtoPhase::Starting) => HubPhase::Starting,
            Ok(ProtoPhase::Downloading) => HubPhase::Downloading,
            Ok(ProtoPhase::Loading) => HubPhase::Loading,
            Ok(ProtoPhase::Warmup) => HubPhase::Warmup,
            Ok(ProtoPhase::Ready) => HubPhase::Ready,
            Ok(ProtoPhase::Failed) => HubPhase::Failed,
            Ok(ProtoPhase::Stopping) => HubPhase::Stopping,
            Ok(ProtoPhase::Unspecified) | Err(_) => HubPhase::Unknown,
        },
        download: snapshot.download.map(hub_download_progress),
        error: snapshot.error,
        seq: snapshot.seq,
    }
}

fn hub_download_progress(progress: ProtoDownloadProgress) -> HubDownloadProgress {
    HubDownloadProgress {
        model: progress.model,
        file: progress.file,
        bytes_done: progress.bytes_done,
        bytes_total: progress.bytes_total,
        files_done: progress.files_done,
        files_total: progress.files_total,
    }
}

/// Sends SIGTERM (Unix) or terminates (Windows), waits up to `grace_period`,
/// then force-kills if still alive.
pub fn stop_process(pid: u32, grace_period: Duration) -> Result<(), DaemonError> {
    send_terminate(pid)?;

    let deadline = Instant::now() + grace_period;
    loop {
        if !is_process_alive(pid) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    force_kill(pid)
}

#[cfg(unix)]
fn send_terminate(pid: u32) -> Result<(), DaemonError> {
    let ret = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if ret != 0 {
        return Err(DaemonError::SignalFailed(pid, io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(windows)]
fn send_terminate(pid: u32) -> Result<(), DaemonError> {
    force_kill(pid)
}

#[cfg(not(any(unix, windows)))]
fn send_terminate(pid: u32) -> Result<(), DaemonError> {
    Err(DaemonError::UnsupportedPlatform("stop".to_owned()))
}

#[cfg(unix)]
fn force_kill(pid: u32) -> Result<(), DaemonError> {
    let ret = unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    if ret != 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ESRCH) {
            return Err(DaemonError::SignalFailed(pid, err));
        }
    }
    Ok(())
}

#[cfg(windows)]
fn force_kill(pid: u32) -> Result<(), DaemonError> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess};
    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if handle.is_null() {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(87) {
                return Ok(());
            }
            return Err(DaemonError::SignalFailed(pid, err));
        }
        TerminateProcess(handle, 1);
        CloseHandle(handle);
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn force_kill(_pid: u32) -> Result<(), DaemonError> {
    Err(DaemonError::UnsupportedPlatform("kill".to_owned()))
}

const MAX_LOG_SIZE: u64 = 50 * 1024 * 1024;

/// Opens the log file in append mode, rotating to `.old` if it exceeds `max_size`.
pub fn prepare_log_file(log_file: &Path, max_size: u64) -> Result<fs::File, DaemonError> {
    if let Some(parent) = log_file.parent() {
        fs::create_dir_all(parent)
            .map_err(|source| DaemonError::Io(parent.to_path_buf(), source))?;
    }

    if log_file.exists()
        && let Ok(meta) = fs::metadata(log_file)
        && meta.len() > max_size
    {
        let old = log_file.with_extension("log.old");
        let _ = fs::remove_file(&old);
        let _ = fs::rename(log_file, &old);
    }

    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)
        .map_err(|source| DaemonError::Io(log_file.to_path_buf(), source))
}

#[derive(Debug, Error)]
pub enum DaemonError {
    #[error("io error on `{}`: {}", _0.display(), _1)]
    Io(PathBuf, io::Error),

    #[error("invalid pid file `{}`", _0.display())]
    InvalidPidFile(PathBuf),

    #[error("failed to spawn `{}`: {}", _0.display(), _1)]
    SpawnFailed(PathBuf, io::Error),

    #[error("timed out waiting for Lumen Hub readiness on {0} after {1:?}")]
    ReadyWaitTimeout(SocketAddr, Duration),

    #[error("failed to create the Lumen Hub control client runtime: {0}")]
    ControlRuntime(String),

    #[error("invalid Lumen Hub control endpoint: {0}")]
    ControlEndpoint(String),

    #[error("Lumen Hub startup failed: {0}")]
    HubStartupFailed(String),

    #[error("failed to signal process {0}: {1}")]
    SignalFailed(u32, io::Error),

    #[error("{0} is not supported on this platform")]
    UnsupportedPlatform(String),
}
