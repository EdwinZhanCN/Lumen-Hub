use std::{
    env, fs, io,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
};

use cliclack::{confirm, input, intro, log, note, outro, progress_bar, select, spinner};
use flate2::read::GzDecoder;
use lumen_schema::LumenConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_MANIFEST_URL: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download/manifest.json";
const OFFICIAL_RELEASE_DOWNLOAD_PREFIX: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/download/";
const OFFICIAL_RELEASE_LATEST_DOWNLOAD_PREFIX: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download/";

fn main() -> ExitCode {
    match run(env::args().collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Help) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), CliError> {
    match args.get(1).map(String::as_str) {
        Some("init") => init(),
        Some("start") => start(&args[2..]),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(CliError::InvalidArgument(format!(
            "unknown command `{other}`"
        ))),
    }
}

fn init() -> Result<(), CliError> {
    intro(format!(" lumen-cli {VERSION} "))?;
    log::info("Create a Lumen preset config for Lumen Hub.")?;

    let home = home_dir().ok_or(CliError::HomeDirUnavailable)?;
    let lumen_dir = home.join(".lumen");
    let default_config_path = lumen_dir.join("config.yaml");
    let bootstrap_path = lumen_dir.join("bootstrap.json");

    if default_config_path.exists() || bootstrap_path.exists() {
        let mut existing = String::new();
        if default_config_path.exists() {
            existing.push_str(&format!("config: {}\n", default_config_path.display()));
        }
        if bootstrap_path.exists() {
            existing.push_str(&format!("bootstrap: {}\n", bootstrap_path.display()));
        }
        note("Existing Lumen setup found", existing.trim_end())?;

        let action = choose(
            "How should init continue?",
            &[
                Choice::new("Overwrite ~/.lumen/config.yaml", true, None),
                Choice::new("Create another config next to it", true, None),
                Choice::new("Cancel", true, None),
            ],
        )?;
        match action {
            0 => {}
            1 => {
                return init_to_paths(
                    &lumen_dir.join("config.generated.yaml"),
                    &lumen_dir.join("bootstrap.generated.json"),
                );
            }
            _ => {
                cliclack::outro_cancel("Init cancelled.")?;
                return Ok(());
            }
        }
    }

    init_to_paths(&default_config_path, &bootstrap_path)
}

fn init_to_paths(config_path: &Path, bootstrap_path: &Path) -> Result<(), CliError> {
    let system = detect_system();
    let memory = detect_memory();
    let platform = platform_profile(&system)?;

    let mut detected = format!("OS: {}\nArch: {}", system.os_label(), system.arch);
    if let Some(total_gb) = memory.total_gb {
        detected.push_str(&format!("\nRAM: {total_gb:.1} GB"));
    } else {
        detected.push_str("\nRAM: unknown");
    }
    note("Detected system", detected)?;

    let region = choose(
        "Select download region:",
        &[
            Choice::new("other - Hugging Face", true, None),
            Choice::new("cn - hf-mirror.com", true, None),
        ],
    )?;
    let region = if region == 1 { "cn" } else { "other" };

    let presets = Preset::all();
    let preset_choices = presets
        .iter()
        .map(|preset| {
            let warning = memory
                .total_gb
                .filter(|ram| *ram < preset.min_ram_gb as f64)
                .map(|ram| {
                    format!(
                        "detected RAM {ram:.1} GB below recommended {} GB",
                        preset.min_ram_gb
                    )
                });
            Choice::new(preset.label(), true, warning)
        })
        .collect::<Vec<_>>();
    let preset_index = choose("Select preset:", &preset_choices)?;
    let preset = presets[preset_index];

    if let Some(total_gb) = memory.total_gb
        && total_gb < preset.min_ram_gb as f64
    {
        log::warning(format!(
            "warning: `{}` recommends at least {} GB RAM; detected {total_gb:.1} GB",
            preset.name, preset.min_ram_gb
        ))?;
    }

    let backends = backend_choices(platform);
    let backend_index = choose_backend("Select beta backend package:", &backends)?;
    let backend = backends[backend_index]
        .backend
        .ok_or_else(|| CliError::InvalidArgument("selected backend is unavailable".to_owned()))?;

    let default_cache = home_dir()
        .ok_or(CliError::HomeDirUnavailable)?
        .join(".lumen")
        .join("models");
    let cache_dir = prompt_cache_dir(&default_cache, preset.min_disk_gb)?;

    let config_yaml = render_config(preset, region, backend, &cache_dir);
    validate_yaml_config(&config_yaml)?;

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).map_err(|source| CliError::CreateDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::write(config_path, config_yaml).map_err(|source| CliError::WriteFile {
        path: config_path.to_path_buf(),
        source,
    })?;

    let bootstrap = Bootstrap {
        version: VERSION.to_owned(),
        region: region.to_owned(),
        preset: preset.name.to_owned(),
        platform: platform.name.to_owned(),
        backend: backend.name.to_owned(),
        release_profile: backend.release_profile.to_owned(),
        cache_dir: cache_dir.display().to_string(),
        config_path: config_path.display().to_string(),
    };
    let bootstrap_json = serde_json::to_string_pretty(&bootstrap)?;
    fs::write(bootstrap_path, bootstrap_json + "\n").map_err(|source| CliError::WriteFile {
        path: bootstrap_path.to_path_buf(),
        source,
    })?;

    log::success(format!("Created config: {}", config_path.display()))?;
    log::success(format!("Created bootstrap: {}", bootstrap_path.display()))?;
    note(
        "Next steps",
        format!(
            "Recommended dist profile: {}\nRun:\n  lumen-cli start",
            backend.release_profile,
        ),
    )?;
    outro("Lumen config is ready.")?;
    Ok(())
}

fn start(args: &[String]) -> Result<(), CliError> {
    let args = StartArgs::parse(args)?;
    intro(format!(" lumen-cli {VERSION} "))?;

    let home = home_dir().ok_or(CliError::HomeDirUnavailable)?;
    let lumen_dir = home.join(".lumen");
    let bootstrap_path = args
        .bootstrap_path
        .clone()
        .unwrap_or_else(|| lumen_dir.join("bootstrap.json"));
    let bootstrap = if bootstrap_path.is_file() {
        Some(read_bootstrap(&bootstrap_path)?)
    } else {
        None
    };
    let config_path = args
        .config_path
        .clone()
        .or_else(|| {
            bootstrap
                .as_ref()
                .map(|bootstrap| PathBuf::from(&bootstrap.config_path))
        })
        .ok_or_else(|| {
            CliError::InvalidArgument(format!(
                "bootstrap `{}` was not found; run `lumen-cli init` first or pass both `--config <path>` and `--profile <profile>`",
                bootstrap_path.display()
            ))
        })?;
    if !config_path.is_file() {
        return Err(CliError::InvalidArgument(format!(
            "config `{}` does not exist; run `lumen-cli init` first or pass `--config <path>`",
            config_path.display()
        )));
    }

    let manifest_url = args
        .manifest_url
        .clone()
        .or_else(|| env::var("LUMEN_RELEASE_MANIFEST_URL").ok())
        .unwrap_or_else(|| DEFAULT_MANIFEST_URL.to_owned());
    let profile = args
        .profile
        .as_deref()
        .or_else(|| {
            bootstrap
                .as_ref()
                .map(|bootstrap| bootstrap.release_profile.as_str())
        })
        .ok_or_else(|| {
            CliError::InvalidArgument(
                "missing release profile; pass `--profile <profile>`".to_owned(),
            )
        })?;

    note(
        "Start plan",
        format!(
            "config: {}\nprofile: {profile}\nmanifest: {manifest_url}",
            config_path.display()
        ),
    )?;

    let manifest = fetch_manifest(&manifest_url)?;
    validate_release_component(&manifest.version, "manifest version")?;
    let artifact = manifest
        .hub
        .iter()
        .find(|artifact| artifact.profile == profile)
        .ok_or_else(|| {
            CliError::InvalidArgument(format!(
                "release manifest `{}` does not contain hub profile `{profile}`",
                manifest.version
            ))
        })?;
    validate_hub_artifact(artifact)?;
    let install_dir = lumen_dir
        .join("hub")
        .join(&manifest.version)
        .join(&artifact.profile);
    let hub = ensure_hub_installed(&install_dir, artifact)?;

    log::step(format!("starting {}", hub.display()))?;
    outro("Lumen Hub output follows.")?;
    let status = Command::new(&hub)
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|source| CliError::SpawnHub {
            path: hub.clone(),
            source,
        })?;
    if !status.success() {
        return Err(CliError::HubExited(status.code()));
    }
    Ok(())
}

fn read_bootstrap(path: &Path) -> Result<Bootstrap, CliError> {
    let contents = fs::read_to_string(path).map_err(|source| CliError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(serde_json::from_str(&contents)?)
}

fn fetch_manifest(url: &str) -> Result<ReleaseManifest, CliError> {
    validate_manifest_url(url)?;
    let spinner = spinner();
    spinner.start(format!("fetching release manifest"));
    let mut response = ureq::get(url).call()?;
    let body = response.body_mut().read_to_string()?;
    let manifest = serde_json::from_str::<ReleaseManifest>(&body)?;
    spinner.stop(format!("release manifest {}", manifest.version));
    Ok(manifest)
}

fn ensure_hub_installed(install_dir: &Path, artifact: &HubArtifact) -> Result<PathBuf, CliError> {
    let exe_name = hub_exe_name();
    let hub_path = install_dir.join("bin").join(exe_name);
    let marker = install_dir.join(".lumen-hub-installed.json");
    if hub_path.is_file() && marker.is_file() {
        log::success(format!(
            "lumen-hub already installed: {}",
            hub_path.display()
        ))?;
        return Ok(hub_path);
    }

    fs::create_dir_all(install_dir).map_err(|source| CliError::CreateDir {
        path: install_dir.to_path_buf(),
        source,
    })?;
    let downloads_dir = install_dir.join(".downloads");
    fs::create_dir_all(&downloads_dir).map_err(|source| CliError::CreateDir {
        path: downloads_dir.clone(),
        source,
    })?;
    let archive_path = downloads_dir.join(&artifact.file_name);
    download_artifact(artifact, &archive_path)?;
    verify_sha256(&archive_path, &artifact.sha256)?;
    extract_artifact(&archive_path, install_dir, artifact)?;
    fs::write(&marker, serde_json::to_string_pretty(artifact)? + "\n").map_err(|source| {
        CliError::WriteFile {
            path: marker,
            source,
        }
    })?;

    if !hub_path.is_file() {
        return Err(CliError::InvalidArgument(format!(
            "installed artifact did not contain `{}`",
            hub_path.display()
        )));
    }
    make_executable(&hub_path)?;
    log::success(format!("lumen-hub ready: {}", hub_path.display()))?;
    Ok(hub_path)
}

fn download_artifact(artifact: &HubArtifact, target: &Path) -> Result<(), CliError> {
    validate_hub_artifact(artifact)?;
    if target.is_file() {
        if sha256_file(target)? == artifact.sha256 {
            log::success(format!("using cached {}", target.display()))?;
            return Ok(());
        }
        fs::remove_file(target).map_err(|source| CliError::WriteFile {
            path: target.to_path_buf(),
            source,
        })?;
    }

    let tmp = target.with_extension("download");
    if tmp.exists() {
        fs::remove_file(&tmp).map_err(|source| CliError::WriteFile {
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
    let mut output = fs::File::create(&tmp).map_err(|source| CliError::WriteFile {
        path: tmp.clone(),
        source,
    })?;
    let mut reader = response.body_mut().as_reader();
    let mut buffer = [0_u8; 128 * 1024];
    let progress = content_len.map(|len| {
        let progress = progress_bar(len).with_download_template();
        progress.start(format!("downloading {}", artifact.file_name));
        progress
    });
    let fallback_spinner = if progress.is_none() {
        let spinner = spinner();
        spinner.start(format!("downloading {}", artifact.file_name));
        Some(spinner)
    } else {
        None
    };
    let mut written = 0_u64;

    loop {
        let read = reader.read(&mut buffer).map_err(CliError::Io)?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|source| CliError::WriteFile {
                path: tmp.clone(),
                source,
            })?;
        written += read as u64;
        if let Some(progress) = &progress {
            progress.inc(read as u64);
        }
    }
    output.flush().map_err(|source| CliError::WriteFile {
        path: tmp.clone(),
        source,
    })?;
    if let Some(progress) = progress {
        progress.stop(format!("downloaded {}", artifact.file_name));
    }
    if let Some(spinner) = fallback_spinner {
        spinner.stop(format!(
            "downloaded {} ({})",
            artifact.file_name,
            format_bytes(written)
        ));
    }

    fs::rename(&tmp, target).map_err(|source| CliError::WriteFile {
        path: target.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), CliError> {
    log::step(format!("verifying {}", path.display()))?;
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(CliError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        });
    }
    log::success("checksum ok")?;
    Ok(())
}

fn extract_artifact(
    archive_path: &Path,
    install_dir: &Path,
    artifact: &HubArtifact,
) -> Result<(), CliError> {
    log::step(format!("extracting {}", archive_path.display()))?;
    if artifact.file_name.ends_with(".zip") {
        extract_zip(archive_path, install_dir)?;
    } else if artifact.file_name.ends_with(".tar.gz") || artifact.file_name.ends_with(".tgz") {
        extract_tar_gz(archive_path, install_dir)?;
    } else {
        return Err(CliError::InvalidArgument(format!(
            "unsupported archive format `{}`",
            artifact.file_name
        )));
    }
    Ok(())
}

fn extract_zip(archive_path: &Path, install_dir: &Path) -> Result<(), CliError> {
    let file = fs::File::open(archive_path).map_err(|source| CliError::ReadFile {
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
            fs::create_dir_all(parent).map_err(|source| CliError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut output = fs::File::create(&target).map_err(|source| CliError::WriteFile {
            path: target.clone(),
            source,
        })?;
        io::copy(&mut entry, &mut output).map_err(CliError::Io)?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&target, fs::Permissions::from_mode(mode)).map_err(|source| {
                CliError::WriteFile {
                    path: target.clone(),
                    source,
                }
            })?;
        }
    }
    Ok(())
}

fn extract_tar_gz(archive_path: &Path, install_dir: &Path) -> Result<(), CliError> {
    let file = fs::File::open(archive_path).map_err(|source| CliError::ReadFile {
        path: archive_path.to_path_buf(),
        source,
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries().map_err(CliError::Io)? {
        let mut entry = entry.map_err(CliError::Io)?;
        let raw_path = entry.path().map_err(CliError::Io)?.into_owned();
        validate_archive_path(&raw_path)?;
        let entry_type = entry.header().entry_type();
        if entry_type.is_symlink() || entry_type.is_hard_link() {
            return Err(CliError::InvalidArgument(format!(
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
            fs::create_dir_all(&target).map_err(|source| CliError::CreateDir {
                path: target,
                source,
            })?;
            continue;
        }
        if !entry_type.is_file() {
            return Err(CliError::InvalidArgument(format!(
                "archive contains unsupported entry `{}`",
                raw_path.display()
            )));
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|source| CliError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut output = fs::File::create(&target).map_err(|source| CliError::WriteFile {
            path: target.clone(),
            source,
        })?;
        io::copy(&mut entry, &mut output).map_err(CliError::Io)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = entry.header().mode().map_err(CliError::Io)?;
            fs::set_permissions(&target, fs::Permissions::from_mode(mode)).map_err(|source| {
                CliError::WriteFile {
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

fn validate_manifest_url(url: &str) -> Result<(), CliError> {
    validate_https_url(url, "manifest URL")?;
    if untrusted_release_urls_allowed() {
        return Ok(());
    }
    if url == DEFAULT_MANIFEST_URL || matches_official_release_asset_url(url, "manifest.json") {
        return Ok(());
    }
    Err(CliError::InvalidArgument(format!(
        "refusing untrusted manifest URL `{url}`; set LUMEN_ALLOW_UNTRUSTED_RELEASE_URLS=1 only if you control that mirror"
    )))
}

fn validate_hub_artifact(artifact: &HubArtifact) -> Result<(), CliError> {
    validate_release_component(&artifact.profile, "release profile")?;
    validate_artifact_file_name(&artifact.file_name)?;
    validate_sha256_text(&artifact.sha256, &artifact.file_name)?;
    validate_artifact_url(&artifact.url, &artifact.file_name)
}

fn validate_artifact_url(url: &str, file_name: &str) -> Result<(), CliError> {
    validate_https_url(url, "artifact URL")?;
    if untrusted_release_urls_allowed() || matches_official_release_asset_url(url, file_name) {
        return Ok(());
    }
    Err(CliError::InvalidArgument(format!(
        "refusing untrusted artifact URL `{url}` for `{file_name}`; set LUMEN_ALLOW_UNTRUSTED_RELEASE_URLS=1 only if you control that mirror"
    )))
}

fn validate_https_url(url: &str, label: &str) -> Result<(), CliError> {
    if url.bytes().any(|byte| byte <= b' ' || byte == 0x7f) {
        return Err(CliError::InvalidArgument(format!(
            "{label} contains whitespace or control characters"
        )));
    }
    if !url.starts_with("https://") {
        return Err(CliError::InvalidArgument(format!("{label} must use https")));
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

fn validate_release_component(value: &str, label: &str) -> Result<(), CliError> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err(CliError::InvalidArgument(format!(
            "invalid {label} `{value}`"
        )));
    }
    Ok(())
}

fn validate_artifact_file_name(file_name: &str) -> Result<(), CliError> {
    validate_release_component(file_name, "artifact file name")?;
    if file_name.ends_with(".zip") || file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz")
    {
        Ok(())
    } else {
        Err(CliError::InvalidArgument(format!(
            "unsupported artifact file name `{file_name}`"
        )))
    }
}

fn validate_sha256_text(value: &str, file_name: &str) -> Result<(), CliError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CliError::InvalidArgument(format!(
            "invalid sha256 for `{file_name}`"
        )));
    }
    Ok(())
}

fn validate_archive_path(path: &Path) -> Result<(), CliError> {
    let path_text = path.to_string_lossy();
    if path_text.contains('\\') || is_unsafe_normalized_archive_name(&path_text) {
        return Err(CliError::InvalidArgument(format!(
            "unsafe archive entry `{}`",
            path.display()
        )));
    }
    Ok(())
}

fn normalize_zip_archive_path(name: &str) -> Result<PathBuf, CliError> {
    let normalized = name.replace('\\', "/");
    if is_unsafe_normalized_archive_name(&normalized) {
        return Err(CliError::InvalidArgument(format!(
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

fn sha256_file(path: &Path) -> Result<String, CliError> {
    let mut file = fs::File::open(path).map_err(|source| CliError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(CliError::Io)?;
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
fn make_executable(path: &Path) -> Result<(), CliError> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)
        .map_err(|source| CliError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    fs::set_permissions(path, permissions).map_err(|source| CliError::WriteFile {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), CliError> {
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
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

fn prompt_cache_dir(default_cache: &Path, min_disk_gb: u64) -> Result<PathBuf, CliError> {
    loop {
        let selected_input: String = input("Model cache directory")
            .default_input(&display_tilde(default_cache).display().to_string())
            .interact()?;
        let selected = expand_tilde(selected_input.trim());

        if is_dangerous_cache_dir(&selected) {
            log::warning(format!(
                "warning: `{}` is not a safe model cache directory",
                selected.display()
            ))?;
            continue;
        }

        if !selected.exists() {
            let create = confirm(format!("Create `{}`?", selected.display()))
                .initial_value(true)
                .interact()?;
            if !create {
                continue;
            }
            fs::create_dir_all(&selected).map_err(|source| CliError::CreateDir {
                path: selected.clone(),
                source,
            })?;
        }

        if !selected.is_dir() {
            log::warning(format!("`{}` is not a directory", selected.display()))?;
            continue;
        }
        if !is_writable_dir(&selected) {
            log::warning(format!("`{}` is not writable", selected.display()))?;
            continue;
        }
        if let Some(free_gb) = free_disk_gb(&selected)
            && free_gb < min_disk_gb as f64
        {
            log::warning(format!(
                "warning: `{}` has {free_gb:.1} GB free; selected preset recommends at least {min_disk_gb} GB",
                selected.display()
            ))?;
            let keep = confirm("Continue anyway?")
                .initial_value(false)
                .interact()?;
            if !keep {
                continue;
            }
        }

        return Ok(selected);
    }
}

fn choose(prompt: &str, choices: &[Choice]) -> Result<usize, CliError> {
    for choice in choices {
        if let Some(reason) = &choice.note {
            log::warning(format!("{}: {reason}", choice.label))?;
        }
    }

    let mut prompt = select(prompt);
    for (index, choice) in choices
        .iter()
        .enumerate()
        .filter(|(_, choice)| choice.enabled)
    {
        prompt = prompt.item(index, &choice.label, "");
    }
    Ok(prompt.interact()?)
}

fn validate_yaml_config(config_yaml: &str) -> Result<(), CliError> {
    let config = serde_yaml::from_str::<LumenConfig>(config_yaml)?;
    config.validate_config()?;
    Ok(())
}

fn render_config(preset: Preset, region: &str, backend: Backend, cache_dir: &Path) -> String {
    let mut services = vec!["ocr", "siglip", "face"];
    if preset.includes_bioclip {
        services.push("bioclip");
    }

    let mut yaml = String::new();
    yaml.push_str("# Generated by lumen-cli init.\n");
    yaml.push_str(&format!(
        "# Preset: {} ({})\n",
        preset.name,
        preset.components.join(", ")
    ));
    yaml.push_str(&format!(
        "# Total minimum requirement: RAM {} GB, GPU/Unified memory {} GB, disk {} GB.\n\n",
        preset.min_ram_gb, preset.min_vram_gb, preset.min_disk_gb
    ));
    yaml.push_str("metadata:\n");
    yaml.push_str("  version: \"0.1.0\"\n");
    yaml.push_str(&format!("  region: {region}\n"));
    yaml.push_str(&format!(
        "  cache_dir: {}\n\n",
        yaml_single_quoted(&cache_dir.display().to_string())
    ));
    yaml.push_str("deployment:\n");
    yaml.push_str("  mode: hub\n");
    yaml.push_str("  services:\n");
    for service in &services {
        yaml.push_str(&format!("    - {service}\n"));
    }
    yaml.push_str("\nserver:\n");
    yaml.push_str("  host: \"0.0.0.0\"\n");
    yaml.push_str("  port: 50051\n");
    yaml.push_str("  batching:\n");
    yaml.push_str("    enabled: true\n");
    yaml.push_str("    max_batch_size: 8\n");
    yaml.push_str("    queue_latency_ms: 2\n\n");
    yaml.push_str("services:\n");
    yaml.push_str("  # Minimum: RAM 1 GB, GPU/Unified memory 512 MB.\n");
    yaml.push_str("  ocr:\n");
    yaml.push_str("    enabled: true\n");
    yaml.push_str("    package: ppocr\n");
    yaml.push_str("    models:\n");
    yaml.push_str("      default:\n");
    yaml.push_str("        model: pp-ocrv5\n");
    yaml.push_str(&format!("        runtime: {}\n", backend.cv_runtime));
    yaml.push_str("        precision: fp32\n\n");

    let siglip = siglip_preset_config(preset, backend);
    yaml.push_str(&format!(
        "  # Minimum: RAM {} GB, GPU/Unified memory {} GB.\n",
        siglip.min_ram_gb, siglip.min_vram_gb
    ));
    yaml.push_str("  siglip:\n");
    yaml.push_str("    enabled: true\n");
    yaml.push_str("    package: siglip\n");
    yaml.push_str("    models:\n");
    yaml.push_str("      default:\n");
    yaml.push_str(&format!("        model: {}\n", siglip.model));
    yaml.push_str(&format!("        runtime: {}\n", backend.semantic_runtime));
    yaml.push_str(&format!("        precision: {}\n\n", siglip.precision));

    yaml.push_str("  # Minimum: RAM 1 GB, GPU/Unified memory 1 GB.\n");
    yaml.push_str("  face:\n");
    yaml.push_str("    enabled: true\n");
    yaml.push_str("    package: insightface\n");
    yaml.push_str("    models:\n");
    yaml.push_str("      default:\n");
    yaml.push_str("        model: antelopev2\n");
    yaml.push_str(&format!("        runtime: {}\n", backend.cv_runtime));
    yaml.push_str("        precision: fp32\n");

    if preset.includes_bioclip {
        yaml.push('\n');
        yaml.push_str(&format!(
            "  # Minimum: RAM {} GB, GPU/Unified memory {} GB.\n",
            if preset.name == "brave" { 16 } else { 8 },
            if preset.name == "brave" { 4 } else { 3 }
        ));
        yaml.push_str("  bioclip:\n");
        yaml.push_str("    enabled: true\n");
        yaml.push_str("    package: clip\n");
        yaml.push_str("    models:\n");
        yaml.push_str("      default:\n");
        yaml.push_str("        model: bioclip-2\n");
        yaml.push_str(&format!("        runtime: {}\n", backend.semantic_runtime));
        yaml.push_str(&format!(
            "        precision: {}\n",
            backend.semantic_precision
        ));
        yaml.push_str(&format!(
            "        dataset: {}\n",
            if preset.name == "brave" {
                "TreeOfLife200M"
            } else {
                "TreeOfLife200MCore"
            }
        ));
    }

    yaml
}

#[derive(Debug, Clone, Copy)]
struct SiglipPresetConfig {
    model: &'static str,
    precision: &'static str,
    min_ram_gb: u64,
    min_vram_gb: u64,
}

fn siglip_preset_config(preset: Preset, backend: Backend) -> SiglipPresetConfig {
    if preset.name == "brave" {
        let precision = if backend.semantic_runtime == "mnn" {
            "q4"
        } else {
            "fp16"
        };
        SiglipPresetConfig {
            model: "siglip2-so400m-patch14-384",
            precision,
            min_ram_gb: 4,
            min_vram_gb: 3,
        }
    } else {
        SiglipPresetConfig {
            model: "siglip2-base-patch16-224",
            precision: backend.semantic_precision,
            min_ram_gb: 3,
            min_vram_gb: 2,
        }
    }
}

fn yaml_single_quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn backend_choices(platform: PlatformProfile) -> Vec<BackendChoice> {
    match platform.name {
        "darwin-arm64" => vec![
            BackendChoice::available(Backend::mnn_metal()),
            BackendChoice::available(Backend::ort_cpu_xnnpack("darwin-arm64")),
            BackendChoice::available(Backend::cpu_only()),
        ],
        "windows-x64" => vec![
            BackendChoice::available(Backend::ort_dml()),
            BackendChoice::available(Backend::cpu_only()),
        ],
        "linux-x64" => vec![
            BackendChoice::new(
                Backend::ort_cuda(),
                detect_nvidia().then_some(()),
                "NVIDIA runtime was not detected",
            ),
            BackendChoice::new(
                Backend::ort_openvino(),
                glibc_meets(2, 28).then_some(()),
                "requires Linux x64 with glibc 2.28+",
            ),
            BackendChoice::available(Backend::cpu_only()),
        ],
        "linux-arm64" => vec![
            BackendChoice::available(Backend::ort_jetson_cuda()),
            BackendChoice::available(Backend::cpu_only_profile("linux-arm64")),
        ],
        _ => vec![BackendChoice::available(Backend::cpu_only())],
    }
}

fn platform_profile(system: &SystemInfo) -> Result<PlatformProfile, CliError> {
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
        _ => Err(CliError::UnsupportedPlatform(format!(
            "{} / {} is not in the beta matrix",
            system.os_label(),
            system.arch
        ))),
    }
}

fn detect_system() -> SystemInfo {
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

fn detect_memory() -> MemoryInfo {
    MemoryInfo {
        total_gb: total_memory_bytes().map(|bytes| bytes as f64 / 1024.0 / 1024.0 / 1024.0),
    }
}

fn total_memory_bytes() -> Option<u64> {
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
        let output = Command::new("powershell")
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

fn detect_nvidia() -> bool {
    command_success("nvidia-smi")
        || Path::new("/proc/driver/nvidia/version").is_file()
        || command_output_contains("ldconfig", &["-p"], "libcuda")
}

fn glibc_meets(major: u32, minor: u32) -> bool {
    let Some(output) = command_stdout("getconf", &["GNU_LIBC_VERSION"]) else {
        return false;
    };
    let version = output
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()));
    let Some(version) = version else {
        return false;
    };
    let mut parts = version.split('.');
    let got_major = parts.next().and_then(|part| part.parse::<u32>().ok());
    let got_minor = parts.next().and_then(|part| part.parse::<u32>().ok());
    match (got_major, got_minor) {
        (Some(got_major), Some(got_minor)) => {
            got_major > major || (got_major == major && got_minor >= minor)
        }
        _ => false,
    }
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

fn free_disk_gb(path: &Path) -> Option<f64> {
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
        let output = Command::new("powershell")
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

fn is_writable_dir(path: &Path) -> bool {
    let probe = path.join(format!(".lumen-write-test-{}", std::process::id()));
    match fs::write(&probe, b"test") {
        Ok(()) => {
            let _ = fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}

fn is_dangerous_cache_dir(path: &Path) -> bool {
    let path = path.components().collect::<Vec<_>>();
    path.len() <= 1
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

fn display_tilde(path: &Path) -> PathBuf {
    let Some(home) = home_dir() else {
        return path.to_path_buf();
    };
    if let Ok(rest) = path.strip_prefix(&home) {
        return PathBuf::from("~").join(rest);
    }
    path.to_path_buf()
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn print_help() {
    println!(
        "Usage:\n  lumen-cli init\n  lumen-cli start [--config <path>] [--profile <profile>] [--manifest-url <url>]\n\nCommands:\n  init     Create a Lumen preset config for lumen-hub\n  start    Ensure the matching lumen-hub dist is installed, then run it"
    );
}

#[derive(Debug, Clone, Copy)]
struct Preset {
    name: &'static str,
    components: &'static [&'static str],
    min_ram_gb: u64,
    min_vram_gb: u64,
    min_disk_gb: u64,
    includes_bioclip: bool,
}

impl Preset {
    fn all() -> &'static [Self] {
        &[
            Self {
                name: "minimal",
                components: &["siglip", "ocr", "face"],
                min_ram_gb: 8,
                min_vram_gb: 4,
                min_disk_gb: 10,
                includes_bioclip: false,
            },
            Self {
                name: "basic",
                components: &["siglip", "ocr", "face", "bioclip"],
                min_ram_gb: 16,
                min_vram_gb: 6,
                min_disk_gb: 20,
                includes_bioclip: true,
            },
            Self {
                name: "brave",
                components: &["siglip", "ocr", "face", "bioclip"],
                min_ram_gb: 32,
                min_vram_gb: 12,
                min_disk_gb: 40,
                includes_bioclip: true,
            },
        ]
    }

    fn label(self) -> String {
        format!(
            "{} ({}) - RAM {} GB, GPU/Unified {} GB",
            self.name,
            self.components.join(", "),
            self.min_ram_gb,
            self.min_vram_gb
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct Backend {
    name: &'static str,
    release_profile: &'static str,
    cv_runtime: &'static str,
    semantic_runtime: &'static str,
    semantic_precision: &'static str,
}

impl Backend {
    fn mnn_metal() -> Self {
        Self {
            name: "mnn-metal",
            release_profile: "darwin-arm64",
            cv_runtime: "mnn",
            semantic_runtime: "mnn",
            semantic_precision: "fp16",
        }
    }

    fn ort_cpu_xnnpack(release_profile: &'static str) -> Self {
        Self {
            name: "ort-cpu-xnnpack",
            release_profile,
            cv_runtime: "onnx",
            semantic_runtime: "onnx",
            semantic_precision: "fp16",
        }
    }

    fn ort_dml() -> Self {
        Self {
            name: "ort-dml",
            release_profile: "windows-x64-dml",
            cv_runtime: "onnx",
            semantic_runtime: "onnx",
            semantic_precision: "fp16",
        }
    }

    fn ort_cuda() -> Self {
        Self {
            name: "ort-cuda",
            release_profile: "linux-x64-cuda",
            cv_runtime: "onnx",
            semantic_runtime: "onnx",
            semantic_precision: "fp16",
        }
    }

    fn ort_openvino() -> Self {
        Self {
            name: "ort-openvino",
            release_profile: "linux-x64-openvino",
            cv_runtime: "onnx",
            semantic_runtime: "onnx",
            semantic_precision: "fp16",
        }
    }

    fn ort_jetson_cuda() -> Self {
        Self {
            name: "ort-jetson-cuda",
            release_profile: "linux-arm64-jetson",
            cv_runtime: "onnx",
            semantic_runtime: "onnx",
            semantic_precision: "fp16",
        }
    }

    fn cpu_only() -> Self {
        Self::cpu_only_profile("universal-cpu")
    }

    fn cpu_only_profile(release_profile: &'static str) -> Self {
        Self {
            name: "cpu-only",
            release_profile,
            cv_runtime: "onnx",
            semantic_runtime: "onnx",
            semantic_precision: "fp32",
        }
    }
}

struct BackendChoice {
    label: String,
    backend: Option<Backend>,
    disabled_reason: Option<String>,
}

impl BackendChoice {
    fn available(backend: Backend) -> Self {
        Self {
            label: format!("{} ({})", backend.name, backend.release_profile),
            backend: Some(backend),
            disabled_reason: None,
        }
    }

    fn new(backend: Backend, available: Option<()>, disabled_reason: &str) -> Self {
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

impl From<&BackendChoice> for Choice {
    fn from(value: &BackendChoice) -> Self {
        Choice::new(
            value.label.clone(),
            value.backend.is_some(),
            value.disabled_reason.clone(),
        )
    }
}

#[derive(Clone)]
struct Choice {
    label: String,
    enabled: bool,
    note: Option<String>,
}

impl Choice {
    fn new(label: impl Into<String>, enabled: bool, note: Option<String>) -> Self {
        Self {
            label: label.into(),
            enabled,
            note,
        }
    }
}

fn choose_backend(prompt: &str, choices: &[BackendChoice]) -> Result<usize, CliError> {
    let display = choices.iter().map(Choice::from).collect::<Vec<_>>();
    choose(prompt, &display)
}

#[derive(Debug, Clone, Copy)]
struct PlatformProfile {
    name: &'static str,
}

#[derive(Debug, Default)]
struct StartArgs {
    config_path: Option<PathBuf>,
    bootstrap_path: Option<PathBuf>,
    manifest_url: Option<String>,
    profile: Option<String>,
}

impl StartArgs {
    fn parse(args: &[String]) -> Result<Self, CliError> {
        let mut parsed = Self::default();
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--config" => {
                    parsed.config_path = Some(PathBuf::from(next_start_value(&mut iter, arg)?));
                }
                "--bootstrap" => {
                    parsed.bootstrap_path = Some(PathBuf::from(next_start_value(&mut iter, arg)?));
                }
                "--manifest-url" => {
                    parsed.manifest_url = Some(next_start_value(&mut iter, arg)?.to_owned());
                }
                "--profile" => {
                    parsed.profile = Some(next_start_value(&mut iter, arg)?.to_owned());
                }
                value if value.starts_with("--config=") => {
                    parsed.config_path = Some(PathBuf::from(value.trim_start_matches("--config=")));
                }
                value if value.starts_with("--bootstrap=") => {
                    parsed.bootstrap_path =
                        Some(PathBuf::from(value.trim_start_matches("--bootstrap=")));
                }
                value if value.starts_with("--manifest-url=") => {
                    parsed.manifest_url =
                        Some(value.trim_start_matches("--manifest-url=").to_owned());
                }
                value if value.starts_with("--profile=") => {
                    parsed.profile = Some(value.trim_start_matches("--profile=").to_owned());
                }
                "--help" | "-h" => {
                    print_help();
                    return Err(CliError::Help);
                }
                other => {
                    return Err(CliError::InvalidArgument(format!(
                        "unknown start argument `{other}`"
                    )));
                }
            }
        }
        Ok(parsed)
    }
}

fn next_start_value<'a, I>(iter: &mut I, flag: &str) -> Result<&'a str, CliError>
where
    I: Iterator<Item = &'a String>,
{
    iter.next()
        .map(String::as_str)
        .ok_or_else(|| CliError::InvalidArgument(format!("missing value for `{flag}`")))
}

#[derive(Debug)]
struct SystemInfo {
    os: OsKind,
    arch: String,
}

impl SystemInfo {
    fn os_label(&self) -> &'static str {
        match self.os {
            OsKind::Macos => "macOS",
            OsKind::Windows => "Windows",
            OsKind::Linux => "Linux",
            OsKind::Other => "Other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OsKind {
    Macos,
    Windows,
    Linux,
    Other,
}

#[derive(Debug)]
struct MemoryInfo {
    total_gb: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Bootstrap {
    version: String,
    region: String,
    preset: String,
    platform: String,
    backend: String,
    release_profile: String,
    cache_dir: String,
    config_path: String,
}

#[derive(Debug, Deserialize)]
struct ReleaseManifest {
    version: String,
    hub: Vec<HubArtifact>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct HubArtifact {
    profile: String,
    file_name: String,
    url: String,
    sha256: String,
}

#[derive(Debug, Error)]
enum CliError {
    #[error("help requested")]
    Help,

    #[error("{0}")]
    InvalidArgument(String),

    #[error("unsupported platform: {0}")]
    UnsupportedPlatform(String),

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

    #[error("lumen-hub exited with status {0:?}")]
    HubExited(Option<i32>),

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

    #[test]
    fn brave_preset_renders_so400m_for_mnn_metal() {
        let config = render_config(
            Preset::all()[2],
            "other",
            Backend::mnn_metal(),
            Path::new("/tmp/lumen"),
        );
        assert!(config.contains("model: siglip2-so400m-patch14-384"));
        assert!(config.contains("precision: q4"));
    }

    #[test]
    fn brave_preset_renders_so400m_fp16_for_onnx() {
        let config = render_config(
            Preset::all()[2],
            "other",
            Backend::ort_cuda(),
            Path::new("/tmp/lumen"),
        );
        assert!(config.contains("model: siglip2-so400m-patch14-384"));
        assert!(config.contains("runtime: onnx"));
        assert!(config.contains("precision: fp16"));
    }

    #[test]
    fn renders_valid_configs_for_all_presets_and_backends() {
        for preset in Preset::all() {
            for backend in [
                Backend::mnn_metal(),
                Backend::ort_cuda(),
                Backend::ort_openvino(),
                Backend::ort_jetson_cuda(),
                Backend::cpu_only_profile("linux-arm64"),
                Backend::cpu_only(),
            ] {
                let config = render_config(*preset, "other", backend, Path::new("/tmp/lumen"));
                validate_yaml_config(&config).unwrap();
            }
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
    fn linux_arm64_offers_jetson_and_native_cpu_profiles() {
        let choices = backend_choices(PlatformProfile {
            name: "linux-arm64",
        });

        assert_eq!(choices.len(), 2);
        let jetson = choices[0]
            .backend
            .expect("linux-arm64 Jetson backend is available");
        assert_eq!(jetson.name, "ort-jetson-cuda");
        assert_eq!(jetson.release_profile, "linux-arm64-jetson");
        let backend = choices[1]
            .backend
            .expect("linux-arm64 backend is available");
        assert_eq!(backend.name, "cpu-only");
        assert_eq!(backend.release_profile, "linux-arm64");
    }

    #[test]
    fn root_dataset_files_are_not_rendered() {
        let config = render_config(
            Preset::all()[1],
            "other",
            Backend::mnn_metal(),
            Path::new("/tmp/lumen"),
        );
        assert!(config.contains("dataset: TreeOfLife200MCore"));
    }

    #[test]
    fn renders_windows_cache_paths_as_valid_yaml() {
        let config = render_config(
            Preset::all()[0],
            "other",
            Backend::ort_dml(),
            Path::new(r"C:\Users\edwin\.lumen\models"),
        );
        validate_yaml_config(&config).unwrap();
        assert!(config.contains(r"cache_dir: 'C:\Users\edwin\.lumen\models'"));
    }

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
