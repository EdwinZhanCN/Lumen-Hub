use std::{
    env, fs, io,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio},
};

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub mod setup;

pub const DEFAULT_MANIFEST_URL: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download/manifest.json";
const OFFICIAL_RELEASE_DOWNLOAD_PREFIX: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/download/";
const OFFICIAL_RELEASE_LATEST_DOWNLOAD_PREFIX: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download/";

#[derive(Debug, Clone, Default)]
pub struct StartOptions {
    pub config_path: Option<PathBuf>,
    pub bootstrap_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub profile: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StartPlan {
    pub lumen_dir: PathBuf,
    pub bootstrap_path: PathBuf,
    pub bootstrap: Option<Bootstrap>,
    pub config_path: PathBuf,
    pub manifest_url: String,
    pub profile: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubStdio {
    Inherit,
    Piped,
}

#[derive(Debug)]
pub struct RunningHub {
    child: Child,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    hub_path: PathBuf,
    config_path: PathBuf,
}

impl RunningHub {
    pub fn stdout(&mut self) -> Option<ChildStdout> {
        self.stdout.take()
    }

    pub fn stderr(&mut self) -> Option<ChildStderr> {
        self.stderr.take()
    }

    pub fn hub_path(&self) -> &Path {
        &self.hub_path
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        self.child.wait()
    }

    pub fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }
}

pub trait LaunchObserver {
    fn manifest_fetch_started(&mut self, _url: &str) {}
    fn manifest_fetched(&mut self, _version: &str) {}
    fn hub_already_installed(&mut self, _hub_path: &Path) {}
    fn download_started(&mut self, _file_name: &str, _total: Option<u64>) {}
    fn download_progress(
        &mut self,
        _file_name: &str,
        _delta: u64,
        _written: u64,
        _total: Option<u64>,
    ) {
    }
    fn download_finished(&mut self, _file_name: &str, _written: u64) {}
    fn verify_started(&mut self, _path: &Path) {}
    fn verify_finished(&mut self, _path: &Path) {}
    fn extract_started(&mut self, _path: &Path) {}
    fn hub_installed(&mut self, _hub_path: &Path) {}
    fn hub_starting(&mut self, _hub_path: &Path) {}
}

#[derive(Debug, Default)]
pub struct NoopObserver;

impl LaunchObserver for NoopObserver {}

pub fn default_lumen_dir() -> Result<PathBuf, LauncherError> {
    Ok(home_dir()
        .ok_or(LauncherError::HomeDirUnavailable)?
        .join(".lumen"))
}

pub fn resolve_start_plan(options: StartOptions) -> Result<StartPlan, LauncherError> {
    let lumen_dir = default_lumen_dir()?;
    let bootstrap_path = options
        .bootstrap_path
        .clone()
        .unwrap_or_else(|| lumen_dir.join("bootstrap.json"));
    let bootstrap = if bootstrap_path.is_file() {
        Some(read_bootstrap(&bootstrap_path)?)
    } else {
        None
    };
    let config_path = options
        .config_path
        .clone()
        .or_else(|| {
            bootstrap
                .as_ref()
                .map(|bootstrap| PathBuf::from(&bootstrap.config_path))
        })
        .ok_or_else(|| {
            LauncherError::InvalidArgument(format!(
                "bootstrap `{}` was not found; run `lumen-cli init` first or pass both `--config <path>` and `--profile <profile>`",
                bootstrap_path.display()
            ))
        })?;
    if !config_path.is_file() {
        return Err(LauncherError::InvalidArgument(format!(
            "config `{}` does not exist; run `lumen-cli init` first or pass `--config <path>`",
            config_path.display()
        )));
    }

    let manifest_url = options
        .manifest_url
        .clone()
        .or_else(|| env::var("LUMEN_RELEASE_MANIFEST_URL").ok())
        .unwrap_or_else(|| DEFAULT_MANIFEST_URL.to_owned());
    let profile = options
        .profile
        .as_deref()
        .or_else(|| {
            bootstrap
                .as_ref()
                .map(|bootstrap| bootstrap.release_profile.as_str())
        })
        .ok_or_else(|| {
            LauncherError::InvalidArgument(
                "missing release profile; pass `--profile <profile>`".to_owned(),
            )
        })?
        .to_owned();

    Ok(StartPlan {
        lumen_dir,
        bootstrap_path,
        bootstrap,
        config_path,
        manifest_url,
        profile,
    })
}

pub fn prepare_hub<O>(plan: &StartPlan, observer: &mut O) -> Result<PathBuf, LauncherError>
where
    O: LaunchObserver,
{
    let manifest = fetch_manifest(&plan.manifest_url, observer)?;
    validate_release_component(&manifest.version, "manifest version")?;
    let artifact = manifest
        .hub
        .iter()
        .find(|artifact| artifact.profile == plan.profile)
        .ok_or_else(|| {
            LauncherError::InvalidArgument(format!(
                "release manifest `{}` does not contain hub profile `{}`",
                manifest.version, plan.profile
            ))
        })?;
    validate_hub_artifact(artifact)?;
    let install_dir = plan
        .lumen_dir
        .join("hub")
        .join(&manifest.version)
        .join(&artifact.profile);
    ensure_hub_installed(&install_dir, artifact, observer)
}

pub fn spawn_hub<O>(
    plan: &StartPlan,
    hub_path: impl AsRef<Path>,
    stdio: HubStdio,
    observer: &mut O,
) -> Result<RunningHub, LauncherError>
where
    O: LaunchObserver,
{
    let hub_path = hub_path.as_ref().to_path_buf();
    observer.hub_starting(&hub_path);
    let mut command = Command::new(&hub_path);
    command.arg("--config").arg(&plan.config_path);
    match stdio {
        HubStdio::Inherit => {
            command
                .stdin(Stdio::inherit())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());
        }
        HubStdio::Piped => {
            command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
        }
    }

    let mut child = command.spawn().map_err(|source| LauncherError::SpawnHub {
        path: hub_path.clone(),
        source,
    })?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    Ok(RunningHub {
        child,
        stdout,
        stderr,
        hub_path,
        config_path: plan.config_path.clone(),
    })
}

pub fn start_and_wait<O>(options: StartOptions, observer: &mut O) -> Result<(), LauncherError>
where
    O: LaunchObserver,
{
    let plan = resolve_start_plan(options)?;
    let hub = prepare_hub(&plan, observer)?;
    let mut running = spawn_hub(&plan, &hub, HubStdio::Inherit, observer)?;
    let status = running
        .wait()
        .map_err(|source| LauncherError::SpawnHub { path: hub, source })?;
    if !status.success() {
        return Err(LauncherError::HubExited(FormattedExitStatus(status)));
    }
    Ok(())
}

pub fn read_bootstrap(path: &Path) -> Result<Bootstrap, LauncherError> {
    let contents = fs::read_to_string(path).map_err(|source| LauncherError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(serde_json::from_str(&contents)?)
}

pub fn fetch_manifest<O>(url: &str, observer: &mut O) -> Result<ReleaseManifest, LauncherError>
where
    O: LaunchObserver,
{
    validate_manifest_url(url)?;
    observer.manifest_fetch_started(url);
    let mut response = ureq::get(url).call()?;
    let body = response.body_mut().read_to_string()?;
    let manifest = serde_json::from_str::<ReleaseManifest>(&body)?;
    observer.manifest_fetched(&manifest.version);
    Ok(manifest)
}

pub fn ensure_hub_installed<O>(
    install_dir: &Path,
    artifact: &HubArtifact,
    observer: &mut O,
) -> Result<PathBuf, LauncherError>
where
    O: LaunchObserver,
{
    let exe_name = hub_exe_name();
    let hub_path = install_dir.join("bin").join(exe_name);
    let marker = install_dir.join(".lumen-hub-installed.json");
    if hub_path.is_file() && marker.is_file() {
        observer.hub_already_installed(&hub_path);
        return Ok(hub_path);
    }

    fs::create_dir_all(install_dir).map_err(|source| LauncherError::CreateDir {
        path: install_dir.to_path_buf(),
        source,
    })?;
    let downloads_dir = install_dir.join(".downloads");
    fs::create_dir_all(&downloads_dir).map_err(|source| LauncherError::CreateDir {
        path: downloads_dir.clone(),
        source,
    })?;
    let archive_path = downloads_dir.join(&artifact.file_name);
    download_artifact(artifact, &archive_path, observer)?;
    verify_sha256(&archive_path, &artifact.sha256, observer)?;
    extract_artifact(&archive_path, install_dir, artifact, observer)?;
    fs::write(&marker, serde_json::to_string_pretty(artifact)? + "\n").map_err(|source| {
        LauncherError::WriteFile {
            path: marker,
            source,
        }
    })?;

    if !hub_path.is_file() {
        return Err(LauncherError::InvalidArgument(format!(
            "installed artifact did not contain `{}`",
            hub_path.display()
        )));
    }
    make_executable(&hub_path)?;
    observer.hub_installed(&hub_path);
    Ok(hub_path)
}

fn download_artifact<O>(
    artifact: &HubArtifact,
    target: &Path,
    observer: &mut O,
) -> Result<(), LauncherError>
where
    O: LaunchObserver,
{
    validate_hub_artifact(artifact)?;
    if target.is_file() {
        if sha256_file(target)? == artifact.sha256 {
            observer.download_finished(&artifact.file_name, fs::metadata(target)?.len());
            return Ok(());
        }
        fs::remove_file(target).map_err(|source| LauncherError::WriteFile {
            path: target.to_path_buf(),
            source,
        })?;
    }

    let tmp = target.with_extension("download");
    if tmp.exists() {
        fs::remove_file(&tmp).map_err(|source| LauncherError::WriteFile {
            path: tmp.clone(),
            source,
        })?;
    }

    let mut response = ureq::get(&artifact.url).call()?;
    let content_len = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let mut output = fs::File::create(&tmp).map_err(|source| LauncherError::WriteFile {
        path: tmp.clone(),
        source,
    })?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; 128 * 1024];
    let mut written = 0_u64;
    observer.download_started(&artifact.file_name, content_len);

    loop {
        let read = reader.read(&mut buffer).map_err(LauncherError::Io)?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|source| LauncherError::WriteFile {
                path: tmp.clone(),
                source,
            })?;
        written += read as u64;
        observer.download_progress(&artifact.file_name, read as u64, written, content_len);
    }
    output.flush().map_err(|source| LauncherError::WriteFile {
        path: tmp.clone(),
        source,
    })?;
    observer.download_finished(&artifact.file_name, written);

    fs::rename(&tmp, target).map_err(|source| LauncherError::WriteFile {
        path: target.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn verify_sha256<O>(path: &Path, expected: &str, observer: &mut O) -> Result<(), LauncherError>
where
    O: LaunchObserver,
{
    observer.verify_started(path);
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(LauncherError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        });
    }
    observer.verify_finished(path);
    Ok(())
}

fn extract_artifact<O>(
    archive_path: &Path,
    install_dir: &Path,
    artifact: &HubArtifact,
    observer: &mut O,
) -> Result<(), LauncherError>
where
    O: LaunchObserver,
{
    observer.extract_started(archive_path);
    if artifact.file_name.ends_with(".zip") {
        extract_zip(archive_path, install_dir)?;
    } else if artifact.file_name.ends_with(".tar.gz") || artifact.file_name.ends_with(".tgz") {
        extract_tar_gz(archive_path, install_dir)?;
    } else {
        return Err(LauncherError::InvalidArgument(format!(
            "unsupported archive format `{}`",
            artifact.file_name
        )));
    }
    Ok(())
}

fn extract_zip(archive_path: &Path, install_dir: &Path) -> Result<(), LauncherError> {
    let file = fs::File::open(archive_path).map_err(|source| LauncherError::ReadFile {
        path: archive_path.to_path_buf(),
        source,
    })?;
    let mut archive = zip::ZipArchive::new(file)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let path = normalize_zip_archive_path(entry.name())?;
        if entry.is_dir() {
            continue;
        }
        let relative = strip_archive_root(&path);
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = install_dir.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|source| LauncherError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut output = fs::File::create(&target).map_err(|source| LauncherError::WriteFile {
            path: target.clone(),
            source,
        })?;
        io::copy(&mut entry, &mut output).map_err(LauncherError::Io)?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&target, fs::Permissions::from_mode(mode)).map_err(|source| {
                LauncherError::WriteFile {
                    path: target.clone(),
                    source,
                }
            })?;
        }
    }
    Ok(())
}

fn extract_tar_gz(archive_path: &Path, install_dir: &Path) -> Result<(), LauncherError> {
    let file = fs::File::open(archive_path).map_err(|source| LauncherError::ReadFile {
        path: archive_path.to_path_buf(),
        source,
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries().map_err(LauncherError::Io)? {
        let mut entry = entry.map_err(LauncherError::Io)?;
        let raw_path = entry.path().map_err(LauncherError::Io)?.into_owned();
        validate_archive_path(&raw_path)?;
        let entry_type = entry.header().entry_type();
        if entry_type.is_symlink() || entry_type.is_hard_link() {
            return Err(LauncherError::InvalidArgument(format!(
                "archive contains link entry `{}`",
                raw_path.display()
            )));
        }

        let relative = strip_archive_root(&raw_path);
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = install_dir.join(relative);
        if entry_type.is_dir() {
            fs::create_dir_all(&target).map_err(|source| LauncherError::CreateDir {
                path: target,
                source,
            })?;
            continue;
        }
        if !entry_type.is_file() {
            return Err(LauncherError::InvalidArgument(format!(
                "archive contains unsupported entry `{}`",
                raw_path.display()
            )));
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|source| LauncherError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut output = fs::File::create(&target).map_err(|source| LauncherError::WriteFile {
            path: target.clone(),
            source,
        })?;
        io::copy(&mut entry, &mut output).map_err(LauncherError::Io)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = entry.header().mode().map_err(LauncherError::Io)?;
            fs::set_permissions(&target, fs::Permissions::from_mode(mode)).map_err(|source| {
                LauncherError::WriteFile {
                    path: target.clone(),
                    source,
                }
            })?;
        }
    }
    Ok(())
}

fn strip_archive_root(path: &Path) -> PathBuf {
    let mut components = path.components();
    let _ = components.next();
    components.as_path().to_path_buf()
}

pub fn validate_manifest_url(url: &str) -> Result<(), LauncherError> {
    validate_https_url(url, "manifest URL")?;
    if untrusted_release_urls_allowed() {
        return Ok(());
    }
    if url == DEFAULT_MANIFEST_URL || matches_official_release_asset_url(url, "manifest.json") {
        return Ok(());
    }
    Err(LauncherError::InvalidArgument(format!(
        "refusing untrusted manifest URL `{url}`; set LUMEN_ALLOW_UNTRUSTED_RELEASE_URLS=1 only if you control that mirror"
    )))
}

pub fn validate_hub_artifact(artifact: &HubArtifact) -> Result<(), LauncherError> {
    validate_release_component(&artifact.profile, "release profile")?;
    validate_artifact_file_name(&artifact.file_name)?;
    validate_sha256_text(&artifact.sha256, &artifact.file_name)?;
    validate_artifact_url(&artifact.url, &artifact.file_name)
}

fn validate_artifact_url(url: &str, file_name: &str) -> Result<(), LauncherError> {
    validate_https_url(url, "artifact URL")?;
    if untrusted_release_urls_allowed() || matches_official_release_asset_url(url, file_name) {
        return Ok(());
    }
    Err(LauncherError::InvalidArgument(format!(
        "refusing untrusted artifact URL `{url}` for `{file_name}`; set LUMEN_ALLOW_UNTRUSTED_RELEASE_URLS=1 only if you control that mirror"
    )))
}

fn validate_https_url(url: &str, label: &str) -> Result<(), LauncherError> {
    if url.bytes().any(|byte| byte <= b' ' || byte == 0x7f) {
        return Err(LauncherError::InvalidArgument(format!(
            "{label} contains whitespace or control characters"
        )));
    }
    if !url.starts_with("https://") {
        return Err(LauncherError::InvalidArgument(format!(
            "{label} must use https"
        )));
    }
    Ok(())
}

fn matches_official_release_asset_url(url: &str, file_name: &str) -> bool {
    if let Some(actual) = url.strip_prefix(OFFICIAL_RELEASE_LATEST_DOWNLOAD_PREFIX) {
        return actual == file_name;
    }
    let Some(rest) = url.strip_prefix(OFFICIAL_RELEASE_DOWNLOAD_PREFIX) else {
        return false;
    };
    let Some((tag, actual)) = rest.rsplit_once('/') else {
        return false;
    };
    !tag.is_empty() && !tag.contains('/') && actual == file_name
}

fn untrusted_release_urls_allowed() -> bool {
    env::var("LUMEN_ALLOW_UNTRUSTED_RELEASE_URLS").is_ok_and(|value| value == "1")
}

fn validate_release_component(value: &str, label: &str) -> Result<(), LauncherError> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(LauncherError::InvalidArgument(format!(
            "invalid {label} `{value}`"
        )));
    }
    Ok(())
}

fn validate_artifact_file_name(file_name: &str) -> Result<(), LauncherError> {
    validate_release_component(file_name, "artifact file name")?;
    if file_name.ends_with(".zip") || file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz")
    {
        Ok(())
    } else {
        Err(LauncherError::InvalidArgument(format!(
            "unsupported artifact file name `{file_name}`"
        )))
    }
}

fn validate_sha256_text(value: &str, file_name: &str) -> Result<(), LauncherError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LauncherError::InvalidArgument(format!(
            "invalid sha256 for `{file_name}`"
        )));
    }
    Ok(())
}

pub fn validate_archive_path(path: &Path) -> Result<(), LauncherError> {
    let path_text = path.to_string_lossy();
    if path_text.contains('\\') || is_unsafe_normalized_archive_name(&path_text) {
        return Err(LauncherError::InvalidArgument(format!(
            "unsafe archive entry `{}`",
            path.display()
        )));
    }
    Ok(())
}

pub fn normalize_zip_archive_path(name: &str) -> Result<PathBuf, LauncherError> {
    let normalized = name.replace('\\', "/");
    if is_unsafe_normalized_archive_name(&normalized) {
        return Err(LauncherError::InvalidArgument(format!(
            "unsafe archive entry `{name}`"
        )));
    }
    Ok(PathBuf::from(normalized))
}

fn is_unsafe_normalized_archive_name(name: &str) -> bool {
    name.is_empty()
        || name.starts_with('/')
        || name.contains(':')
        || name.split('/').any(|part| part == "..")
}

pub fn sha256_file(path: &Path) -> Result<String, LauncherError> {
    let mut file = fs::File::open(path).map_err(|source| LauncherError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(LauncherError::Io)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hub_exe_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "lumen-hub.exe"
    } else {
        "lumen-hub"
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), LauncherError> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)
        .map_err(|source| LauncherError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    fs::set_permissions(path, permissions).map_err(|source| LauncherError::WriteFile {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), LauncherError> {
    Ok(())
}

pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.1} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.1} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Bootstrap {
    pub version: String,
    pub region: String,
    pub preset: String,
    pub platform: String,
    pub backend: String,
    pub release_profile: String,
    pub cache_dir: String,
    pub config_path: String,
}

#[derive(Debug, Deserialize)]
pub struct ReleaseManifest {
    pub version: String,
    pub hub: Vec<HubArtifact>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HubArtifact {
    pub profile: String,
    pub file_name: String,
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Error)]
pub enum LauncherError {
    #[error("{0}")]
    InvalidArgument(String),

    #[error("could not determine home directory")]
    HomeDirUnavailable,

    #[error("failed to create directory `{}`: {source}", path.display())]
    CreateDir { path: PathBuf, source: io::Error },

    #[error("failed to read file `{}`: {source}", path.display())]
    ReadFile { path: PathBuf, source: io::Error },

    #[error("failed to write file `{}`: {source}", path.display())]
    WriteFile { path: PathBuf, source: io::Error },

    #[error("failed to spawn lumen-hub `{}`: {source}", path.display())]
    SpawnHub { path: PathBuf, source: io::Error },

    #[error("lumen-hub {0}")]
    HubExited(FormattedExitStatus),

    #[error("checksum mismatch for `{}`: expected {expected}, got {actual}", path.display())]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("http error: {0}")]
    Http(#[from] ureq::Error),

    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct FormattedExitStatus(pub ExitStatus);

impl std::fmt::Display for FormattedExitStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = self.0;
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if let Some(code) = status.code() {
                write!(f, "exited with status code: {code}")
            } else if let Some(signal) = status.signal() {
                let signal_desc = match signal {
                    1 => "SIGHUP (hangup)",
                    2 => "SIGINT (interrupt)",
                    3 => "SIGQUIT (quit)",
                    4 => "SIGILL (illegal instruction)",
                    5 => "SIGTRAP (trace trap)",
                    6 => "SIGABRT (abort)",
                    8 => "SIGFPE (floating-point exception)",
                    9 => "SIGKILL (kill)",
                    11 => "SIGSEGV (segmentation fault)",
                    13 => "SIGPIPE (broken pipe)",
                    14 => "SIGALRM (alarm clock)",
                    15 => "SIGTERM (termination)",
                    _ => "unknown signal",
                };
                write!(f, "was terminated by signal {signal} ({signal_desc})")
            } else {
                write!(f, "exited for unknown reasons")
            }
        }
        #[cfg(not(unix))]
        {
            if let Some(code) = status.code() {
                write!(f, "exited with status code: {code}")
            } else {
                write!(f, "exited")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_release_artifact_metadata() {
        let sha256 = "a".repeat(64);
        let artifact = HubArtifact {
            profile: "linux-x64-cuda".to_owned(),
            file_name: "lumen-hub-linux-x64-cuda.tar.gz".to_owned(),
            url: "https://github.com/EdwinZhanCN/Lumen-Hub/releases/download/v0.1.0/lumen-hub-linux-x64-cuda.tar.gz".to_owned(),
            sha256,
        };
        validate_hub_artifact(&artifact).unwrap();

        let mut bad_file = artifact.clone();
        bad_file.file_name = "../lumen-hub.tar.gz".to_owned();
        assert!(validate_hub_artifact(&bad_file).is_err());

        let mut bad_url = artifact.clone();
        bad_url.url = "https://example.com/lumen-hub-linux-x64-cuda.tar.gz".to_owned();
        assert!(validate_hub_artifact(&bad_url).is_err());
    }

    #[test]
    fn validates_archive_entry_paths() {
        assert!(validate_archive_path(Path::new("lumen-hub/bin/lumen-hub")).is_ok());
        assert!(validate_archive_path(Path::new("../bin/lumen-hub")).is_err());
        assert!(validate_archive_path(Path::new("lumen-hub/../bin/lumen-hub")).is_err());
        assert!(validate_archive_path(Path::new("/tmp/lumen-hub")).is_err());
        assert!(validate_archive_path(Path::new(r"lumen-hub\bin\lumen-hub")).is_err());
    }

    #[test]
    fn normalizes_legacy_windows_zip_entry_paths() {
        assert_eq!(
            normalize_zip_archive_path(r"lumen-hub-windows-x64-dml\README.md")
                .unwrap()
                .to_string_lossy(),
            "lumen-hub-windows-x64-dml/README.md"
        );
        assert!(normalize_zip_archive_path(r"..\lumen-hub.exe").is_err());
        assert!(normalize_zip_archive_path(r"C:\tmp\lumen-hub.exe").is_err());
        assert!(normalize_zip_archive_path(r"\\server\share\lumen-hub.exe").is_err());
    }
}
