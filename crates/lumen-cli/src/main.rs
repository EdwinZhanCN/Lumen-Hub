use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use cliclack::{confirm, input, intro, log, note, outro, select};
use lumen_launcher::{
    Bootstrap, LaunchObserver, LauncherError, StartOptions, format_bytes, prepare_hub,
    resolve_start_plan, spawn_hub,
};
use lumen_schema::LumenConfig;
use thiserror::Error;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    let backend_index = choose_backend("Select alpha backend package:", &backends)?;
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

    let options = StartOptions {
        config_path: args.config_path,
        bootstrap_path: args.bootstrap_path,
        manifest_url: args.manifest_url,
        profile: args.profile,
    };
    let plan = resolve_start_plan(options)?;

    note(
        "Start plan",
        format!(
            "config: {}\nprofile: {}\nmanifest: {}",
            plan.config_path.display(),
            plan.profile,
            plan.manifest_url
        ),
    )?;

    let mut observer = CliLaunchObserver;
    let hub = prepare_hub(&plan, &mut observer)?;

    outro("Lumen Hub output follows.")?;
    let mut running = spawn_hub(
        &plan,
        &hub,
        lumen_launcher::HubStdio::Inherit,
        &mut observer,
    )?;
    let status = running
        .wait()
        .map_err(|source| LauncherError::SpawnHub { path: hub, source })?;
    if !status.success() {
        return Err(CliError::Launcher(LauncherError::HubExited(
            lumen_launcher::FormattedExitStatus(status),
        )));
    }
    Ok(())
}

struct CliLaunchObserver;

impl LaunchObserver for CliLaunchObserver {
    fn manifest_fetch_started(&mut self, _url: &str) {
        let _ = log::step("fetching release manifest");
    }

    fn manifest_fetched(&mut self, version: &str) {
        let _ = log::success(format!("release manifest {version}"));
    }

    fn hub_already_installed(&mut self, hub_path: &Path) {
        let _ = log::success(format!(
            "lumen-hub already installed: {}",
            hub_path.display()
        ));
    }

    fn download_started(&mut self, file_name: &str, total: Option<u64>) {
        let detail = total
            .map(format_bytes)
            .map(|size| format!(" ({size})"))
            .unwrap_or_default();
        let _ = log::step(format!("downloading {file_name}{detail}"));
    }

    fn download_finished(&mut self, file_name: &str, written: u64) {
        let _ = log::success(format!(
            "downloaded {file_name} ({})",
            format_bytes(written)
        ));
    }

    fn verify_started(&mut self, path: &Path) {
        let _ = log::step(format!("verifying {}", path.display()));
    }

    fn verify_finished(&mut self, _path: &Path) {
        let _ = log::success("checksum ok");
    }

    fn extract_started(&mut self, path: &Path) {
        let _ = log::step(format!("extracting {}", path.display()));
    }

    fn hub_installed(&mut self, hub_path: &Path) {
        let _ = log::success(format!("lumen-hub ready: {}", hub_path.display()));
    }

    fn hub_starting(&mut self, hub_path: &Path) {
        let _ = log::step(format!("starting {}", hub_path.display()));
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
    let mut yaml = String::new();
    yaml.push_str("# Generated by lumen-cli init.\n");
    yaml.push_str(&format!(
        "# Preset: {} ({})\n",
        preset.name,
        preset.components.join(", ")
    ));
    yaml.push_str(&format!(
        "# Resource guidance: RAM {} GB, GPU/Unified memory {} GB, disk {} GB.\n",
        preset.min_ram_gb, preset.min_vram_gb, preset.min_disk_gb
    ));
    yaml.push_str(
        "# Weights and BioCLIP catalogs are memory-mapped: they load on demand and\n\
         # do not all count as resident RAM. A brief warmup spike is reclaimed after startup.\n\n",
    );
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
    for service in preset.components {
        yaml.push_str(&format!("    - {service}\n"));
    }
    yaml.push_str("\nserver:\n");
    yaml.push_str("  host: \"0.0.0.0\"\n");
    yaml.push_str("  port: 50051\n");
    // Dynamic batching is off: on Metal/CubeCL batch>1 regresses per-image
    // latency, so concurrent batch-1 inference is faster and lower-memory.
    yaml.push_str("  batching:\n");
    yaml.push_str("    enabled: false\n");
    yaml.push_str("    max_batch_size: 8\n");
    yaml.push_str("    queue_latency_ms: 2\n\n");
    yaml.push_str("services:\n");
    let siglip = siglip_preset_config(preset, backend);
    yaml.push_str("  # SigLIP: semantic image + text embeddings.\n");
    yaml.push_str("  siglip:\n");
    yaml.push_str("    enabled: true\n");
    yaml.push_str("    package: siglip\n");
    yaml.push_str("    models:\n");
    yaml.push_str("      default:\n");
    yaml.push_str(&format!("        model: {}\n", siglip.model));
    yaml.push_str(&format!("        runtime: {}\n", backend.semantic_runtime));
    yaml.push_str(&format!("        precision: {}\n\n", siglip.precision));

    yaml.push_str("  # InsightFace antelopev2: face detection + recognition.\n");
    yaml.push_str("  face:\n");
    yaml.push_str("    enabled: true\n");
    yaml.push_str("    package: insightface\n");
    yaml.push_str("    models:\n");
    yaml.push_str("      default:\n");
    yaml.push_str("        model: antelopev2\n");
    yaml.push_str(&format!("        runtime: {}\n", backend.cv_runtime));
    yaml.push_str("        precision: fp16q8\n");

    if preset.includes("ocr") {
        yaml.push('\n');
        yaml.push_str("  # PP-OCRv6 small: in-image text detection + recognition.\n");
        yaml.push_str("  ocr:\n");
        yaml.push_str("    enabled: true\n");
        yaml.push_str("    package: ppocr\n");
        yaml.push_str("    models:\n");
        yaml.push_str("      default:\n");
        yaml.push_str("        model: pp-ocrv6-small\n");
        yaml.push_str(&format!("        runtime: {}\n", backend.cv_runtime));
        yaml.push_str("        precision: fp16q8\n");
    }

    if preset.includes("bioclip") {
        yaml.push('\n');
        // brave uses the full TreeOfLife200M catalog for long-tail species
        // coverage; other presets use the smaller Core subset.
        let dataset = if preset.name == "brave" {
            "TreeOfLife200M"
        } else {
            "TreeOfLife200MCore"
        };
        yaml.push_str("  # BioCLIP-2: species classification over the Tree of Life catalog.\n");
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
        yaml.push_str(&format!("        dataset: {dataset}\n"));
    }

    yaml
}

#[derive(Debug, Clone, Copy)]
struct SiglipPresetConfig {
    model: &'static str,
    precision: &'static str,
}

fn siglip_preset_config(preset: Preset, backend: Backend) -> SiglipPresetConfig {
    if preset.name == "brave" {
        SiglipPresetConfig {
            model: "siglip2-so400m-patch14-384",
            precision: backend.semantic_precision,
        }
    } else {
        SiglipPresetConfig {
            model: "siglip2-base-patch16-224",
            precision: backend.semantic_precision,
        }
    }
}

fn yaml_single_quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn backend_choices(platform: PlatformProfile) -> Vec<BackendChoice> {
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
            BackendChoice::new(
                Backend::rocm("linux-x64-rocm"),
                detect_amd().then_some(()),
                "AMD ROCm runtime was not detected",
            ),
            BackendChoice::available(Backend::gpu("linux-x64-gpu")),
            BackendChoice::available(Backend::cpu("linux-x64-cpu")),
        ],
        "linux-arm64" => vec![
            BackendChoice::new(
                Backend::jetson(),
                is_jetson().then_some(()),
                "not running on an NVIDIA Jetson (L4T) device",
            ),
            BackendChoice::available(Backend::gpu("linux-arm64-gpu")),
            BackendChoice::available(Backend::cpu("linux-arm64-cpu")),
        ],
        _ => vec![BackendChoice::available(Backend::cpu("linux-x64-cpu"))],
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
            "{} / {} is not in the alpha matrix",
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

/// True on NVIDIA Jetson / L4T (Tegra) devices, which need the L4T-built CUDA
/// package rather than the generic arm64 build.
fn is_jetson() -> bool {
    Path::new("/etc/nv_tegra_release").is_file()
        || fs::read_to_string("/proc/device-tree/model")
            .map(|model| model.contains("Jetson") || model.contains("NVIDIA Orin"))
            .unwrap_or(false)
}

fn detect_amd() -> bool {
    command_success("rocminfo")
        || Path::new("/dev/kfd").exists()
        || Path::new("/sys/module/amdgpu").is_dir()
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
}

impl Preset {
    fn all() -> &'static [Self] {
        &[
            // RAM/VRAM/disk are measured guidance (Apple M2 Pro, Metal,
            // fp16q8). Weights and BioCLIP catalogs are memory-mapped, so model
            // size lands on disk and cold faults, not resident RAM; the RAM
            // figures cover the Hub working set plus same-host Photos/OS. See
            // docs/lumen-hub-tensor-batching-decision.md.
            Self {
                name: "minimal",
                components: &["siglip", "face"],
                min_ram_gb: 4,
                min_vram_gb: 2,
                min_disk_gb: 2,
            },
            Self {
                name: "basic",
                components: &["siglip", "face", "ocr", "bioclip"],
                min_ram_gb: 6,
                min_vram_gb: 3,
                min_disk_gb: 6,
            },
            Self {
                name: "brave",
                components: &["siglip", "face", "ocr", "bioclip"],
                min_ram_gb: 8,
                min_vram_gb: 4,
                min_disk_gb: 10,
            },
        ]
    }

    fn includes(self, component: &str) -> bool {
        self.components.contains(&component)
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
    /// All inference runs through the Burn runtime with fp16q8 presets; only the compute
    /// backend (selected at build time and reflected in the release profile)
    /// differs between packages.
    fn burn(name: &'static str, release_profile: &'static str) -> Self {
        Self {
            name,
            release_profile,
            cv_runtime: "burn",
            semantic_runtime: "burn",
            semantic_precision: "fp16q8",
        }
    }

    fn metal() -> Self {
        Self::burn("metal", "darwin-arm64-metal")
    }

    /// Portable wgpu backend (Vulkan / GL / DX12 at runtime).
    fn gpu(release_profile: &'static str) -> Self {
        Self::burn("gpu", release_profile)
    }

    fn cuda(release_profile: &'static str) -> Self {
        Self::burn("cuda", release_profile)
    }

    fn rocm(release_profile: &'static str) -> Self {
        Self::burn("rocm", release_profile)
    }

    /// NVIDIA Jetson / L4T CUDA build.
    fn jetson() -> Self {
        Self::burn("jetson", "linux-arm64-jetson")
    }

    fn cpu(release_profile: &'static str) -> Self {
        Self::burn("cpu", release_profile)
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

    #[error("failed to write file `{}`: {source}", path.display())]
    WriteFile { path: PathBuf, source: io::Error },

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("generated config failed validation: {0}")]
    Config(#[from] lumen_schema::ConfigValidationError),

    #[error("{0}")]
    Launcher(#[from] LauncherError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brave_preset_renders_so400m_on_burn() {
        let config = render_config(
            Preset::all()[2],
            "other",
            Backend::metal(),
            Path::new("/tmp/lumen"),
        );
        assert!(config.contains("model: siglip2-so400m-patch14-384"));
        assert!(config.contains("model: pp-ocrv6-small"));
        assert!(config.contains("runtime: burn"));
        assert!(config.contains("precision: fp16q8"));
    }

    #[test]
    fn minimal_preset_renders_siglip_and_face_only() {
        let config = render_config(
            Preset::all()[0],
            "other",
            Backend::metal(),
            Path::new("/tmp/lumen"),
        );
        assert!(config.contains("deployment:\n  mode: hub\n  services:\n    - siglip\n    - face"));
        assert!(config.contains("model: siglip2-base-patch16-224"));
        assert!(config.contains("model: antelopev2"));
        assert!(!config.contains("\n  ocr:\n"));
        assert!(!config.contains("\n  bioclip:\n"));
    }

    #[test]
    fn basic_preset_adds_ocr_and_core_bioclip() {
        let config = render_config(
            Preset::all()[1],
            "other",
            Backend::metal(),
            Path::new("/tmp/lumen"),
        );
        assert!(config.contains(
            "deployment:\n  mode: hub\n  services:\n    - siglip\n    - face\n    - ocr\n    - bioclip"
        ));
        assert!(config.contains("model: siglip2-base-patch16-224"));
        assert!(config.contains("model: pp-ocrv6-small"));
        assert!(config.contains("\n  bioclip:\n"));
        assert!(config.contains("dataset: TreeOfLife200MCore"));
    }

    #[test]
    fn presets_disable_batching() {
        for preset in Preset::all() {
            let config = render_config(*preset, "other", Backend::metal(), Path::new("/tmp/lumen"));
            assert!(
                config.contains("batching:\n    enabled: false"),
                "{} preset must disable batching",
                preset.name
            );
        }
    }

    #[test]
    fn renders_valid_configs_for_all_presets_and_backends() {
        for preset in Preset::all() {
            for backend in [
                Backend::metal(),
                Backend::gpu("linux-x64-gpu"),
                Backend::cuda("linux-x64-cuda"),
                Backend::rocm("linux-x64-rocm"),
                Backend::jetson(),
                Backend::cpu("linux-x64-cpu"),
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
    fn linux_arm64_offers_jetson_gpu_and_cpu_profiles() {
        let choices = backend_choices(PlatformProfile {
            name: "linux-arm64",
        });

        // jetson (hardware-gated) + portable gpu + cpu fallback.
        assert_eq!(choices.len(), 3);
        assert_eq!(choices[0].label.split_whitespace().next(), Some("jetson"));
        let gpu = choices[1].backend.expect("arm64 gpu backend is available");
        assert_eq!(gpu.name, "gpu");
        assert_eq!(gpu.release_profile, "linux-arm64-gpu");
        let cpu = choices[2].backend.expect("arm64 cpu backend is available");
        assert_eq!(cpu.name, "cpu");
        assert_eq!(cpu.release_profile, "linux-arm64-cpu");
    }

    #[test]
    fn linux_x64_offers_cuda_rocm_gpu_and_cpu_profiles() {
        let choices = backend_choices(PlatformProfile { name: "linux-x64" });
        assert_eq!(choices.len(), 4);
        assert_eq!(choices[0].label.split_whitespace().next(), Some("cuda"));
        assert_eq!(choices[1].label.split_whitespace().next(), Some("rocm"));
        assert_eq!(
            choices[2].backend.expect("gpu available").release_profile,
            "linux-x64-gpu"
        );
        assert_eq!(
            choices[3].backend.expect("cpu available").release_profile,
            "linux-x64-cpu"
        );
    }

    #[test]
    fn brave_preset_renders_full_bioclip_dataset() {
        let config = render_config(
            Preset::all()[2],
            "other",
            Backend::metal(),
            Path::new("/tmp/lumen"),
        );
        // Match with the trailing newline so the full dataset is not satisfied
        // by the "TreeOfLife200MCore" substring.
        assert!(config.contains("dataset: TreeOfLife200M\n"));
        assert!(!config.contains("dataset: TreeOfLife200MCore"));
    }

    #[test]
    fn renders_windows_cache_paths_as_valid_yaml() {
        let config = render_config(
            Preset::all()[0],
            "other",
            Backend::cpu("windows-x64-cpu"),
            Path::new(r"C:\Users\edwin\.lumen\models"),
        );
        validate_yaml_config(&config).unwrap();
        assert!(config.contains(r"cache_dir: 'C:\Users\edwin\.lumen\models'"));
    }
}
