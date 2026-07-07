use std::{
    env, io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use cliclack::{confirm, input, intro, log, note, outro, select};
use lumen_launcher::{
    LaunchObserver, LauncherError, StartOptions, format_bytes, prepare_hub, resolve_start_plan,
    setup, spawn_hub,
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
    let backend_index = choose_backend("Select alpha backend package:", &backends)?;
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

fn print_help() {
    println!(
        "Usage:\n  lumen-cli init\n  lumen-cli start [--config <path>] [--profile <profile>] [--manifest-url <url>]\n\nCommands:\n  init     Create a Lumen preset config for lumen-hub\n  start    Ensure the matching lumen-hub dist is installed, then run it"
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_args_parse_equals_forms() {
        let args = vec![
            "--config=/tmp/config.yaml".to_owned(),
            "--bootstrap".to_owned(),
            "/tmp/bootstrap.json".to_owned(),
            "--manifest-url=https://example.test/manifest.json".to_owned(),
            "--profile".to_owned(),
            "linux-x64-cpu".to_owned(),
        ];
        let parsed = StartArgs::parse(&args).unwrap();
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
}
