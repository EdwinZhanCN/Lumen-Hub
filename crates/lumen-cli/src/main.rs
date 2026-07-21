use std::{
    env, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use cliclack::{confirm, input, intro, log, note, outro, select};
use lumen_launcher::{
    LaunchObserver, LauncherError, StartOptions, daemon, format_bytes, prepare_hub,
    read_server_port, resolve_start_plan, setup, spawn_hub,
};
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
        Some("run") => run_foreground(&args[2..]),
        Some("start") => start_background(&args[2..]),
        Some("stop") => stop(&args[2..]),
        Some("reload") => reload(&args[2..]),
        Some("validate") => validate(&args[2..]),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(CliError::InvalidArgument(format!(
            "unknown command `{other}`"
        ))),
    }
}

// --- Commands ---

fn run_foreground(args: &[String]) -> Result<(), CliError> {
    let args = CommonArgs::parse(args)?;
    intro(format!(" lumen-cli {VERSION} "))?;

    let options = StartOptions {
        config_path: args.config_path.clone(),
        bootstrap_path: args.bootstrap_path,
        manifest_url: args.manifest_url,
        profile: args.profile,
    };
    let plan = resolve_start_plan(options)?;

    note(
        "Run plan",
        format!(
            "config: {}\nprofile: {}\nmanifest: {}",
            plan.config_path.display(),
            plan.profile,
            plan.manifest_url
        ),
    )?;

    let mut observer = CliLaunchObserver;
    let hub = prepare_hub(&plan, &mut observer)?;

    let paths = daemon::daemon_paths(&plan.lumen_dir);
    let mut running = spawn_hub(
        &plan,
        &hub,
        lumen_launcher::HubStdio::Inherit,
        &mut observer,
    )?;
    let pid = running.id();
    daemon::write_pid_file(&paths.pid_file, pid)?;

    outro("Lumen Hub output follows. Press Ctrl-C to stop.")?;
    let status = running
        .wait()
        .map_err(|source| LauncherError::SpawnHub { path: hub, source })?;
    daemon::remove_pid_file(&paths.pid_file)?;

    if !status.success() {
        return Err(CliError::Launcher(LauncherError::HubExited(
            lumen_launcher::FormattedExitStatus(status),
        )));
    }
    Ok(())
}

fn start_background(args: &[String]) -> Result<(), CliError> {
    let args = CommonArgs::parse(args)?;
    intro(format!(" lumen-cli {VERSION} "))?;

    let options = StartOptions {
        config_path: args.config_path.clone(),
        bootstrap_path: args.bootstrap_path,
        manifest_url: args.manifest_url,
        profile: args.profile,
    };
    let plan = resolve_start_plan(options)?;
    let paths = daemon::daemon_paths(&plan.lumen_dir);

    if let Some(pid) = daemon::check_running(&paths.pid_file)? {
        return Err(CliError::InvalidArgument(format!(
            "lumen-hub is already running (pid {pid})"
        )));
    }

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

    log::step("starting lumen-hub in background")?;
    let pid = daemon::spawn_background(&daemon::BackgroundSpawnConfig {
        hub_path: hub,
        config_path: plan.config_path.clone(),
        log_file: paths.log_file.clone(),
    })?;
    daemon::write_pid_file(&paths.pid_file, pid)?;

    let port = read_server_port(&plan.config_path);
    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], port)));

    log::step(format!("waiting for lumen-hub to become healthy on {addr}"))?;
    match daemon::wait_for_healthy(&daemon::HealthCheckConfig {
        addr,
        ..Default::default()
    }) {
        Ok(()) => {
            log::success(format!("lumen-hub started (pid {pid})"))?;
            log::info(format!("logs: {}", paths.log_file.display()))?;
            outro("Lumen Hub is running in the background.")?;
            Ok(())
        }
        Err(e) => {
            let _ = daemon::stop_process(pid, Duration::from_secs(5));
            daemon::remove_pid_file(&paths.pid_file)?;
            Err(CliError::Daemon(format!(
                "{e}\ncheck logs: {}",
                paths.log_file.display()
            )))
        }
    }
}

fn stop(args: &[String]) -> Result<(), CliError> {
    let args = StopArgs::parse(args)?;
    let lumen_dir = lumen_launcher::default_lumen_dir()?;
    let paths = daemon::daemon_paths(&lumen_dir);

    let Some(pid) = daemon::check_running(&paths.pid_file)? else {
        println!("lumen-hub is not running");
        return Ok(());
    };

    println!("stopping lumen-hub (pid {pid})...");
    daemon::stop_process(pid, Duration::from_secs(args.timeout))?;
    daemon::remove_pid_file(&paths.pid_file)?;
    println!("lumen-hub stopped");
    Ok(())
}

fn reload(args: &[String]) -> Result<(), CliError> {
    let args = CommonArgs::parse(args)?;
    let lumen_dir = lumen_launcher::default_lumen_dir()?;
    let paths = daemon::daemon_paths(&lumen_dir);

    if let Some(pid) = daemon::check_running(&paths.pid_file)? {
        println!("stopping lumen-hub (pid {pid})...");
        daemon::stop_process(pid, Duration::from_secs(10))?;
        daemon::remove_pid_file(&paths.pid_file)?;
        println!("lumen-hub stopped");
    } else {
        println!("lumen-hub is not running, starting fresh");
    }

    println!("starting lumen-hub with updated config...");
    let start_args = args.to_vec();
    start_background(&start_args)
}

fn validate(args: &[String]) -> Result<(), CliError> {
    let args = ValidateArgs::parse(args)?;

    let config_path = if let Some(path) = args.config_path {
        path
    } else {
        let lumen_dir = lumen_launcher::default_lumen_dir()?;
        let bootstrap_path = lumen_dir.join("bootstrap.json");
        if bootstrap_path.is_file() {
            let bootstrap = lumen_launcher::read_bootstrap(&bootstrap_path)?;
            PathBuf::from(&bootstrap.config_path)
        } else {
            lumen_dir.join("config.yaml")
        }
    };

    if !config_path.is_file() {
        return Err(CliError::InvalidArgument(format!(
            "config `{}` does not exist",
            config_path.display()
        )));
    }

    let contents = std::fs::read_to_string(&config_path)?;

    match setup::validate_yaml_config(&contents) {
        Ok(()) => {
            println!("config is valid: {}", config_path.display());
            Ok(())
        }
        Err(e) => Err(CliError::InvalidArgument(format!(
            "config validation failed for `{}`:\n{e}",
            config_path.display()
        ))),
    }
}

// --- Init ---

fn init() -> Result<(), CliError> {
    intro(format!(" lumen-cli {VERSION} "))?;
    log::info("Create a Lumen preset config for Lumen Hub.")?;

    let paths = setup::default_setup_paths()?;
    if paths.config_path.exists() || paths.bootstrap_path.exists() {
        let mut existing = String::new();
        if paths.config_path.exists() {
            existing.push_str(&format!("config: {}\n", paths.config_path.display()));
        }
        if paths.bootstrap_path.exists() {
            existing.push_str(&format!("bootstrap: {}\n", paths.bootstrap_path.display()));
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
                    &paths.lumen_dir.join("config.generated.yaml"),
                    &paths.lumen_dir.join("bootstrap.generated.json"),
                );
            }
            _ => {
                cliclack::outro_cancel("Init cancelled.")?;
                return Ok(());
            }
        }
    }

    init_to_paths(&paths.config_path, &paths.bootstrap_path)
}

fn init_to_paths(config_path: &Path, bootstrap_path: &Path) -> Result<(), CliError> {
    let system = setup::detect_system();
    let memory = setup::detect_memory();
    let platform = setup::platform_profile(&system)?;

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
    let region = if region == 1 {
        setup::REGION_CN
    } else {
        setup::REGION_OTHER
    };

    let presets = setup::Preset::all();
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

    let backends = setup::backend_choices(platform);
    let backend_index = choose_backend("Select backend package:", &backends)?;
    let backend = backends[backend_index]
        .backend
        .ok_or_else(|| CliError::InvalidArgument("selected backend is unavailable".to_owned()))?;

    let default_cache = setup::default_setup_paths()?.cache_dir;
    let cache_dir = prompt_cache_dir(&default_cache, preset.min_disk_gb)?;

    let selection = setup::SetupSelection {
        version: VERSION.to_owned(),
        region: region.to_owned(),
        preset,
        platform,
        backend,
        cache_dir,
        config_path: config_path.to_path_buf(),
        bootstrap_path: bootstrap_path.to_path_buf(),
    };
    let written = setup::write_setup(&selection)?;

    log::success(format!("Created config: {}", written.config_path.display()))?;
    log::success(format!(
        "Created bootstrap: {}",
        written.bootstrap_path.display()
    ))?;
    note(
        "Next steps",
        format!(
            "Recommended dist profile: {}\nRun:\n  lumen-cli start",
            written.bootstrap.release_profile,
        ),
    )?;
    outro("Lumen config is ready.")?;
    Ok(())
}

// --- Observer ---

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

// --- UI helpers ---

fn prompt_cache_dir(default_cache: &Path, min_disk_gb: u64) -> Result<PathBuf, CliError> {
    loop {
        let selected_input: String = input("Model cache directory")
            .default_input(&setup::display_tilde(default_cache).display().to_string())
            .interact()?;
        let selected = setup::expand_tilde(selected_input.trim());

        if setup::is_dangerous_cache_dir(&selected) {
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
            setup::ensure_cache_dir(&selected)?;
        }

        if !selected.is_dir() {
            log::warning(format!("`{}` is not a directory", selected.display()))?;
            continue;
        }
        if !setup::is_writable_dir(&selected) {
            log::warning(format!("`{}` is not writable", selected.display()))?;
            continue;
        }
        if let Some(free_gb) = setup::free_disk_gb(&selected)
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

impl From<&setup::BackendChoice> for Choice {
    fn from(value: &setup::BackendChoice) -> Self {
        Choice::new(
            value.label.clone(),
            value.backend.is_some(),
            value.disabled_reason.clone(),
        )
    }
}

fn choose_backend(prompt: &str, choices: &[setup::BackendChoice]) -> Result<usize, CliError> {
    let display = choices.iter().map(Choice::from).collect::<Vec<_>>();
    choose(prompt, &display)
}

// --- Arg parsing ---

#[derive(Debug, Default, Clone)]
struct CommonArgs {
    config_path: Option<PathBuf>,
    bootstrap_path: Option<PathBuf>,
    manifest_url: Option<String>,
    profile: Option<String>,
}

impl CommonArgs {
    fn parse(args: &[String]) -> Result<Self, CliError> {
        let mut parsed = Self::default();
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--config" => {
                    parsed.config_path = Some(PathBuf::from(next_value(&mut iter, arg)?));
                }
                "--bootstrap" => {
                    parsed.bootstrap_path = Some(PathBuf::from(next_value(&mut iter, arg)?));
                }
                "--manifest-url" => {
                    parsed.manifest_url = Some(next_value(&mut iter, arg)?.to_owned());
                }
                "--profile" => {
                    parsed.profile = Some(next_value(&mut iter, arg)?.to_owned());
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
                        "unknown argument `{other}`"
                    )));
                }
            }
        }
        Ok(parsed)
    }

    fn to_vec(&self) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(ref p) = self.config_path {
            args.push("--config".to_owned());
            args.push(p.display().to_string());
        }
        if let Some(ref p) = self.bootstrap_path {
            args.push("--bootstrap".to_owned());
            args.push(p.display().to_string());
        }
        if let Some(ref u) = self.manifest_url {
            args.push("--manifest-url".to_owned());
            args.push(u.clone());
        }
        if let Some(ref p) = self.profile {
            args.push("--profile".to_owned());
            args.push(p.clone());
        }
        args
    }
}

#[derive(Debug)]
struct StopArgs {
    timeout: u64,
}

impl StopArgs {
    fn parse(args: &[String]) -> Result<Self, CliError> {
        let mut timeout = 10u64;
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--timeout" => {
                    timeout = next_value(&mut iter, arg)?
                        .parse()
                        .map_err(|_| CliError::InvalidArgument("invalid --timeout value".into()))?;
                }
                value if value.starts_with("--timeout=") => {
                    timeout = value
                        .trim_start_matches("--timeout=")
                        .parse()
                        .map_err(|_| CliError::InvalidArgument("invalid --timeout value".into()))?;
                }
                "--help" | "-h" => {
                    print_help();
                    return Err(CliError::Help);
                }
                other => {
                    return Err(CliError::InvalidArgument(format!(
                        "unknown stop argument `{other}`"
                    )));
                }
            }
        }
        Ok(Self { timeout })
    }
}

#[derive(Debug, Default)]
struct ValidateArgs {
    config_path: Option<PathBuf>,
}

impl ValidateArgs {
    fn parse(args: &[String]) -> Result<Self, CliError> {
        let mut parsed = Self::default();
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--config" => {
                    parsed.config_path = Some(PathBuf::from(next_value(&mut iter, arg)?));
                }
                value if value.starts_with("--config=") => {
                    parsed.config_path = Some(PathBuf::from(value.trim_start_matches("--config=")));
                }
                "--help" | "-h" => {
                    print_help();
                    return Err(CliError::Help);
                }
                other => {
                    return Err(CliError::InvalidArgument(format!(
                        "unknown validate argument `{other}`"
                    )));
                }
            }
        }
        Ok(parsed)
    }
}

fn next_value<'a, I>(iter: &mut I, flag: &str) -> Result<&'a str, CliError>
where
    I: Iterator<Item = &'a String>,
{
    iter.next()
        .map(String::as_str)
        .ok_or_else(|| CliError::InvalidArgument(format!("missing value for `{flag}`")))
}

fn print_help() {
    println!(
        "\
Usage:
  lumen-cli <command> [options]

Commands:
  init       Create a Lumen preset config for lumen-hub
  run        Run lumen-hub in the foreground (blocks until stopped)
  start      Start lumen-hub in the background
  stop       Stop the background lumen-hub process
  reload     Restart lumen-hub to pick up config changes
  validate   Validate a config file without starting the server

Options for run/start/reload:
  --config <path>        Path to config YAML
  --bootstrap <path>     Path to bootstrap JSON
  --manifest-url <url>   Override release manifest URL
  --profile <profile>    Select a dist profile

Options for stop:
  --timeout <secs>       Grace period before force kill (default: 10)

Options for validate:
  --config <path>        Path to config YAML to validate"
    );
}

#[derive(Debug, Error)]
enum CliError {
    #[error("help requested")]
    Help,

    #[error("{0}")]
    InvalidArgument(String),

    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("{0}")]
    Setup(#[from] setup::SetupError),

    #[error("{0}")]
    Launcher(#[from] LauncherError),

    #[error("{0}")]
    Daemon(String),
}

impl From<daemon::DaemonError> for CliError {
    fn from(e: daemon::DaemonError) -> Self {
        CliError::Daemon(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_args_parse_equals_forms() {
        let args = vec![
            "--config=/tmp/config.yaml".to_owned(),
            "--bootstrap".to_owned(),
            "/tmp/bootstrap.json".to_owned(),
            "--manifest-url=https://example.test/manifest.json".to_owned(),
            "--profile".to_owned(),
            "linux-x64-cpu".to_owned(),
        ];
        let parsed = CommonArgs::parse(&args).unwrap();
        assert_eq!(
            parsed.config_path.unwrap(),
            PathBuf::from("/tmp/config.yaml")
        );
        assert_eq!(
            parsed.bootstrap_path.unwrap(),
            PathBuf::from("/tmp/bootstrap.json")
        );
        assert_eq!(
            parsed.manifest_url.unwrap(),
            "https://example.test/manifest.json"
        );
        assert_eq!(parsed.profile.unwrap(), "linux-x64-cpu");
    }

    #[test]
    fn stop_args_parse_timeout() {
        let args = vec!["--timeout".to_owned(), "30".to_owned()];
        let parsed = StopArgs::parse(&args).unwrap();
        assert_eq!(parsed.timeout, 30);

        let args = vec!["--timeout=5".to_owned()];
        let parsed = StopArgs::parse(&args).unwrap();
        assert_eq!(parsed.timeout, 5);
    }
}
