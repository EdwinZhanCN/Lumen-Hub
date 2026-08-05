use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

pub use lumen_schema::Preset;
use lumen_schema::{ConfigTarget, LumenConfig, RenderOptions, preset_yaml};
use thiserror::Error;

use crate::Bootstrap;

pub const REGION_OTHER: &str = "other";
pub const REGION_CN: &str = "cn";

#[derive(Debug, Clone)]
pub struct SetupPaths {
    pub lumen_dir: PathBuf,
    pub config_path: PathBuf,
    pub bootstrap_path: PathBuf,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct SetupSelection {
    pub version: String,
    pub region: String,
    pub preset: Preset,
    pub platform: PlatformProfile,
    pub backend: Backend,
    pub cache_dir: PathBuf,
    pub config_path: PathBuf,
    pub bootstrap_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WrittenSetup {
    pub config_path: PathBuf,
    pub bootstrap_path: PathBuf,
    pub bootstrap: Bootstrap,
}

pub fn default_setup_paths() -> Result<SetupPaths, SetupError> {
    let lumen_dir = home_dir()
        .ok_or(SetupError::HomeDirUnavailable)?
        .join(".lumen");
    Ok(SetupPaths {
        config_path: lumen_dir.join("config.yaml"),
        bootstrap_path: lumen_dir.join("bootstrap.json"),
        cache_dir: lumen_dir.join("models"),
        lumen_dir,
    })
}

pub fn write_setup(selection: &SetupSelection) -> Result<WrittenSetup, SetupError> {
    let config_yaml = render_config(selection.preset, &selection.region, &selection.cache_dir)?;
    validate_yaml_config(&config_yaml)?;

    if let Some(parent) = selection.config_path.parent() {
        fs::create_dir_all(parent).map_err(|source| SetupError::CreateDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(&selection.config_path, config_yaml).map_err(|source| SetupError::WriteFile {
        path: selection.config_path.clone(),
        source,
    })?;

    let bootstrap = Bootstrap {
        version: selection.version.clone(),
        region: selection.region.clone(),
        preset: selection.preset.name.to_owned(),
        platform: selection.platform.name.to_owned(),
        backend: selection.backend.name.to_owned(),
        release_profile: selection.backend.release_profile.to_owned(),
        cache_dir: selection.cache_dir.display().to_string(),
        config_path: selection.config_path.display().to_string(),
    };
    let bootstrap_json = serde_json::to_string_pretty(&bootstrap)?;

    if let Some(parent) = selection.bootstrap_path.parent() {
        fs::create_dir_all(parent).map_err(|source| SetupError::CreateDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(&selection.bootstrap_path, bootstrap_json + "\n").map_err(|source| {
        SetupError::WriteFile {
            path: selection.bootstrap_path.clone(),
            source,
        }
    })?;

    Ok(WrittenSetup {
        config_path: selection.config_path.clone(),
        bootstrap_path: selection.bootstrap_path.clone(),
        bootstrap,
    })
}

pub fn validate_yaml_config(config_yaml: &str) -> Result<(), SetupError> {
    let config = serde_yaml::from_str::<LumenConfig>(config_yaml)?;
    config.validate_config()?;
    Ok(())
}

pub fn render_config(preset: Preset, region: &str, cache_dir: &Path) -> Result<String, SetupError> {
    let cache_dir = cache_dir.display().to_string();
    preset_yaml(
        preset,
        &RenderOptions {
            region,
            cache_dir: &cache_dir,
            target: ConfigTarget::Network,
        },
    )
    .map_err(SetupError::RenderConfig)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backend {
    pub name: &'static str,
    pub release_profile: &'static str,
}

impl Backend {
    /// All inference runs through the Burn runtime with fp16q8 presets; only the compute
    /// backend (selected at build time and reflected in the release profile)
    /// differs between packages.
    pub fn burn(name: &'static str, release_profile: &'static str) -> Self {
        Self {
            name,
            release_profile,
        }
    }

    pub fn metal() -> Self {
        Self::burn("metal", "darwin-arm64-metal")
    }

    /// Portable wgpu backend (Vulkan / GL / DX12 at runtime).
    pub fn gpu(release_profile: &'static str) -> Self {
        Self::burn("gpu", release_profile)
    }

    pub fn cuda(release_profile: &'static str) -> Self {
        Self::burn("cuda", release_profile)
    }

    pub fn cpu(release_profile: &'static str) -> Self {
        Self::burn("cpu", release_profile)
    }
}

#[derive(Debug, Clone)]
pub struct BackendChoice {
    pub label: String,
    pub backend: Option<Backend>,
    pub disabled_reason: Option<String>,
}

impl BackendChoice {
    pub fn available(backend: Backend) -> Self {
        Self {
            label: format!("{} ({})", backend.name, backend.release_profile),
            backend: Some(backend),
            disabled_reason: None,
        }
    }

    pub fn new(backend: Backend, available: Option<()>, disabled_reason: &str) -> Self {
        if available.is_some() {
            Self::available(backend)
        } else {
            Self {
                label: format!("{} ({})", backend.name, backend.release_profile),
                backend: None,
                disabled_reason: Some(disabled_reason.to_owned()),
            }
        }
    }
}

pub fn backend_choices(platform: PlatformProfile) -> Vec<BackendChoice> {
    match platform.name {
        "darwin-arm64" => vec![
            BackendChoice::available(Backend::metal()),
            BackendChoice::available(Backend::cpu("darwin-arm64-cpu")),
        ],
        "windows-x64" => vec![
            BackendChoice::available(Backend::gpu("windows-x64-gpu")),
            BackendChoice::available(Backend::cpu("windows-x64-cpu")),
        ],
        "linux-x64" => vec![
            BackendChoice::new(
                Backend::cuda("linux-x64-cuda"),
                detect_nvidia().then_some(()),
                "NVIDIA runtime was not detected",
            ),
            BackendChoice::available(Backend::gpu("linux-x64-gpu")),
            BackendChoice::available(Backend::cpu("linux-x64-cpu")),
        ],
        "linux-arm64" => vec![BackendChoice::available(Backend::cpu("linux-arm64-cpu"))],
        _ => vec![BackendChoice::available(Backend::cpu("linux-x64-cpu"))],
    }
}

pub fn recommended_backend(platform: PlatformProfile) -> Result<Backend, SetupError> {
    backend_choices(platform)
        .into_iter()
        .find_map(|choice| choice.backend)
        .ok_or_else(|| SetupError::InvalidArgument("no available backend for platform".to_owned()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformProfile {
    pub name: &'static str,
}

pub fn current_platform_profile() -> Result<PlatformProfile, SetupError> {
    platform_profile(&detect_system())
}

pub fn platform_profile(system: &SystemInfo) -> Result<PlatformProfile, SetupError> {
    match (system.os, system.arch.as_str()) {
        (OsKind::Macos, "aarch64" | "arm64") => Ok(PlatformProfile {
            name: "darwin-arm64",
        }),
        (OsKind::Windows, "x86_64" | "amd64") => Ok(PlatformProfile {
            name: "windows-x64",
        }),
        (OsKind::Linux, "x86_64" | "amd64") => Ok(PlatformProfile { name: "linux-x64" }),
        (OsKind::Linux, "aarch64" | "arm64") => Ok(PlatformProfile {
            name: "linux-arm64",
        }),
        _ => Err(SetupError::UnsupportedPlatform(format!(
            "{} / {} is not in the release matrix",
            system.os_label(),
            system.arch
        ))),
    }
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os: OsKind,
    pub arch: String,
}

impl SystemInfo {
    pub fn os_label(&self) -> &'static str {
        match self.os {
            OsKind::Macos => "macOS",
            OsKind::Windows => "Windows",
            OsKind::Linux => "Linux",
            OsKind::Other => "Other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsKind {
    Macos,
    Windows,
    Linux,
    Other,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryInfo {
    pub total_gb: Option<f64>,
}

pub fn detect_system() -> SystemInfo {
    SystemInfo {
        os: if cfg!(target_os = "macos") {
            OsKind::Macos
        } else if cfg!(target_os = "windows") {
            OsKind::Windows
        } else if cfg!(target_os = "linux") {
            OsKind::Linux
        } else {
            OsKind::Other
        },
        arch: env::consts::ARCH.to_owned(),
    }
}

pub fn detect_memory() -> MemoryInfo {
    MemoryInfo {
        total_gb: total_memory_bytes().map(|bytes| bytes as f64 / 1024.0 / 1024.0 / 1024.0),
    }
}

pub fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    }

    #[cfg(target_os = "windows")]
    {
        let mut command = Command::new("powershell");
        hide_console_window(&mut command);
        let output = command
            .args([
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ])
            .output()
            .ok()?;
        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

pub fn detect_nvidia() -> bool {
    command_success("nvidia-smi")
        || Path::new("/proc/driver/nvidia/version").is_file()
        || command_output_contains("ldconfig", &["-p"], "libcuda")
}

pub fn free_disk_gb(path: &Path) -> Option<f64> {
    #[cfg(unix)]
    {
        let output = Command::new("df")
            .args(["-Pk", path.to_str()?])
            .output()
            .ok()?;
        let stdout = String::from_utf8(output.stdout).ok()?;
        let line = stdout.lines().nth(1)?;
        let available_kb = line.split_whitespace().nth(3)?.parse::<u64>().ok()?;
        Some(available_kb as f64 / 1024.0 / 1024.0)
    }

    #[cfg(windows)]
    {
        let root = path.components().next()?.as_os_str().to_string_lossy();
        let mut command = Command::new("powershell");
        hide_console_window(&mut command);
        let output = command
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "(Get-PSDrive -Name '{}').Free",
                    root.trim_end_matches([':', '\\'])
                ),
            ])
            .output()
            .ok()?;
        let bytes = String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

pub fn is_writable_dir(path: &Path) -> bool {
    let probe = path.join(format!(".lumen-write-test-{}", std::process::id()));
    match fs::write(&probe, b"test") {
        Ok(()) => {
            let _ = fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}

pub fn is_dangerous_cache_dir(path: &Path) -> bool {
    let path = path.components().collect::<Vec<_>>();
    path.len() <= 1
}

pub fn ensure_cache_dir(path: &Path) -> Result<(), SetupError> {
    if !path.exists() {
        fs::create_dir_all(path).map_err(|source| SetupError::CreateDir {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

pub fn display_tilde(path: &Path) -> PathBuf {
    let Some(home) = home_dir() else {
        return path.to_path_buf();
    };
    if let Ok(rest) = path.strip_prefix(&home) {
        return PathBuf::from("~").join(rest);
    }
    path.to_path_buf()
}

fn command_success(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn command_output_contains(name: &str, args: &[&str], needle: &str) -> bool {
    command_stdout(name, args).is_some_and(|stdout| stdout.contains(needle))
}

fn command_stdout(name: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(name).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(target_os = "windows")]
fn hide_console_window(command: &mut Command) {
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    command.creation_flags(CREATE_NO_WINDOW);
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("{0}")]
    InvalidArgument(String),

    #[error("unsupported platform: {0}")]
    UnsupportedPlatform(String),

    #[error("could not determine home directory")]
    HomeDirUnavailable,

    #[error("failed to create directory `{}`: {source}", path.display())]
    CreateDir { path: PathBuf, source: io::Error },

    #[error("failed to write file `{}`: {source}", path.display())]
    WriteFile { path: PathBuf, source: io::Error },

    #[error("failed to render config: {0}")]
    RenderConfig(String),

    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("generated config failed validation: {0}")]
    Config(#[from] lumen_schema::ConfigValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(preset: &str, region: &str, cache_dir: &Path) -> LumenConfig {
        let yaml = render_config(Preset::by_name(preset).unwrap(), region, cache_dir).unwrap();
        serde_yaml::from_str(&yaml).unwrap()
    }

    #[test]
    fn renders_valid_configs_for_all_presets() {
        for preset in Preset::all() {
            let yaml = render_config(*preset, REGION_OTHER, Path::new("/tmp/lumen")).unwrap();
            validate_yaml_config(&yaml).unwrap();
        }
    }

    #[test]
    fn minimal_preset_selects_siglip_and_face_only() {
        let config = parse("minimal", REGION_OTHER, Path::new("/tmp/lumen"));
        assert_eq!(config.deployment_service_names(), vec!["siglip", "face"]);
        assert!(config.services["siglip"].enabled);
        assert!(config.services["face"].enabled);
        assert!(!config.services["ocr"].enabled);
        assert!(!config.services["bioclip"].enabled);
    }

    #[test]
    fn basic_and_brave_presets_select_the_canonical_models() {
        let basic = parse("basic", REGION_OTHER, Path::new("/tmp/lumen"));
        assert_eq!(
            basic.services["bioclip"].models["default"]
                .dataset
                .as_deref(),
            Some("TreeOfLife200MCore")
        );
        let brave = parse("brave", REGION_OTHER, Path::new("/tmp/lumen"));
        assert_eq!(
            brave.services["siglip"].models["default"].model,
            "siglip2-so400m-patch14-384"
        );
        assert_eq!(
            brave.services["bioclip"].models["default"]
                .dataset
                .as_deref(),
            Some("TreeOfLife200M")
        );
    }

    #[test]
    fn launcher_configs_are_network_scoped_and_disable_batching() {
        for preset in Preset::all() {
            let config = parse(preset.name, REGION_OTHER, Path::new("/tmp/lumen"));
            assert_eq!(config.server.host, "0.0.0.0");
            assert!(!config.server.batching.enabled);
        }
    }

    #[test]
    fn detects_linux_arm64_platform_profile() {
        let profile = platform_profile(&SystemInfo {
            os: OsKind::Linux,
            arch: "aarch64".to_owned(),
        })
        .unwrap();
        assert_eq!(profile.name, "linux-arm64");
    }

    #[test]
    fn linux_arm64_offers_only_the_published_cpu_profile() {
        let choices = backend_choices(PlatformProfile {
            name: "linux-arm64",
        });
        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0]
                .backend
                .expect("arm64 cpu backend is available")
                .release_profile,
            "linux-arm64-cpu"
        );
    }

    #[test]
    fn linux_x64_offers_only_published_profiles() {
        let choices = backend_choices(PlatformProfile { name: "linux-x64" });
        assert_eq!(choices.len(), 3);
        assert_eq!(choices[0].label.split_whitespace().next(), Some("cuda"));
        assert_eq!(choices[1].backend.unwrap().release_profile, "linux-x64-gpu");
        assert_eq!(choices[2].backend.unwrap().release_profile, "linux-x64-cpu");
    }

    #[test]
    fn windows_cache_paths_round_trip_as_yaml() {
        let path = Path::new(r"C:\Users\edwin\.lumen\models");
        let config = parse("minimal", REGION_OTHER, path);
        assert_eq!(config.metadata.cache_dir, path.display().to_string());
    }
}
