use std::{
    env, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::OnceLock,
    time::Duration,
};

use cliclack::{confirm, input, intro, log, note, outro, select};
use lumen_launcher::{
    Bootstrap, LaunchObserver, LauncherError, StartOptions, daemon, format_bytes, prepare_hub,
    read_server_port, resolve_start_plan, setup, spawn_hub,
};
use thiserror::Error;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Language {
    En,
    ZhCn,
}

static LANGUAGE: OnceLock<Language> = OnceLock::new();

fn language() -> Language {
    *LANGUAGE.get_or_init(Language::detect)
}

fn text(en: &'static str, zh: &'static str) -> &'static str {
    match language() {
        Language::En => en,
        Language::ZhCn => zh,
    }
}

impl Language {
    fn detect() -> Self {
        ["LC_ALL", "LC_MESSAGES", "LANG"]
            .into_iter()
            .find_map(|name| {
                env::var(name)
                    .ok()
                    .and_then(|value| Self::parse_locale(&value))
            })
            .unwrap_or(Self::En)
    }

    fn parse_locale(value: &str) -> Option<Self> {
        let normalized = value
            .trim()
            .split(['.', '@'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-");
        if normalized == "en" || normalized.starts_with("en-") {
            return Some(Self::En);
        }
        if matches!(normalized.as_str(), "zh" | "zh-cn" | "zh-hans")
            || normalized.starts_with("zh-cn-")
            || normalized.starts_with("zh-hans-")
        {
            return Some(Self::ZhCn);
        }
        None
    }

    fn parse_explicit(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "en" => Some(Self::En),
            "zh-cn" => Some(Self::ZhCn),
            _ => None,
        }
    }
}

fn extract_language(mut args: Vec<String>) -> Result<(Language, Vec<String>), String> {
    let mut selected = None;
    let mut index = 1;
    while index < args.len() {
        if args[index] == "--lang" {
            let value = args
                .get(index + 1)
                .ok_or_else(|| "missing value for `--lang`".to_owned())?
                .clone();
            selected = Language::parse_explicit(&value);
            if selected.is_none() {
                return Err(format!(
                    "unsupported language `{value}`; use `en` or `zh-CN`"
                ));
            }
            args.drain(index..=index + 1);
            continue;
        }
        if let Some(value) = args[index].strip_prefix("--lang=") {
            selected = Language::parse_explicit(value);
            if selected.is_none() {
                return Err(format!(
                    "unsupported language `{value}`; use `en` or `zh-CN`"
                ));
            }
            args.remove(index);
            continue;
        }
        index += 1;
    }
    Ok((selected.unwrap_or_else(Language::detect), args))
}

fn main() -> ExitCode {
    let (lang, args) = match extract_language(env::args().collect()) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };
    let _ = LANGUAGE.set(lang);
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Help) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", text("error", "错误"));
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), CliError> {
    match args.get(1).map(String::as_str) {
        Some("configure" | "init") => configure(),
        Some("run") => run_foreground(&args[2..]),
        Some("start") => start_background(&args[2..]),
        Some("stop") => stop(&args[2..]),
        Some("reload") => reload(&args[2..]),
        Some("validate") => validate(&args[2..]),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(CliError::InvalidArgument(match language() {
            Language::En => format!("unknown command `{other}`"),
            Language::ZhCn => format!("未知命令 `{other}`"),
        })),
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
        text("Run plan", "运行计划"),
        format!(
            "{}: {}\n{}: {}\n{}: {}",
            text("config", "配置"),
            plan.config_path.display(),
            text("profile", "发布 profile"),
            plan.profile,
            text("manifest", "发布清单"),
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

    outro(text(
        "Lumen Hub output follows. Press Ctrl-C to stop.",
        "以下为 Lumen Hub 输出。按 Ctrl-C 停止。",
    ))?;
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
        return Err(CliError::InvalidArgument(match language() {
            Language::En => format!("lumen-hub is already running (pid {pid})"),
            Language::ZhCn => format!("lumen-hub 已在运行（pid {pid}）"),
        }));
    }

    note(
        text("Start plan", "启动计划"),
        format!(
            "{}: {}\n{}: {}\n{}: {}",
            text("config", "配置"),
            plan.config_path.display(),
            text("profile", "发布 profile"),
            plan.profile,
            text("manifest", "发布清单"),
            plan.manifest_url
        ),
    )?;

    let mut observer = CliLaunchObserver;
    let hub = prepare_hub(&plan, &mut observer)?;

    log::step(text(
        "starting lumen-hub in background",
        "正在后台启动 lumen-hub",
    ))?;
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

    log::step(format!(
        "{} ({addr})",
        text("waiting for Lumen Hub readiness", "等待 Lumen Hub 就绪")
    ))?;
    let mut last_message = None;
    match daemon::wait_for_ready(
        &daemon::ReadyWaitConfig {
            addr,
            ..Default::default()
        },
        |status| {
            let message = hub_status_message(status);
            if last_message.as_deref() != Some(message.as_str()) {
                let _ = log::info(&message);
                last_message = Some(message);
            }
        },
    ) {
        Ok(()) => {
            log::success(match language() {
                Language::En => format!("lumen-hub started (pid {pid})"),
                Language::ZhCn => format!("lumen-hub 已启动（pid {pid}）"),
            })?;
            log::info(format!(
                "{}: {}",
                text("logs", "日志"),
                paths.log_file.display()
            ))?;
            outro(text(
                "Lumen Hub is running in the background.",
                "Lumen Hub 已在后台运行。",
            ))?;
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

fn hub_status_message(status: &daemon::HubStatus) -> String {
    match status.phase {
        daemon::HubPhase::Starting => text("Starting", "正在启动").to_owned(),
        daemon::HubPhase::Downloading => {
            let Some(progress) = &status.download else {
                return text("Downloading models", "正在下载模型").to_owned();
            };
            let bytes = if progress.bytes_total == 0 {
                format_bytes(progress.bytes_done)
            } else {
                format!(
                    "{} / {}",
                    format_bytes(progress.bytes_done),
                    format_bytes(progress.bytes_total)
                )
            };
            format!(
                "{}: {} · {} · {} ({}/{})",
                text("Downloading models", "正在下载模型"),
                progress.model,
                progress.file,
                bytes,
                progress.files_done,
                progress.files_total
            )
        }
        daemon::HubPhase::Loading => text("Loading models", "正在加载模型").to_owned(),
        daemon::HubPhase::Warmup => text("Warming up models", "正在预热模型").to_owned(),
        daemon::HubPhase::Ready => text("Ready", "已就绪").to_owned(),
        daemon::HubPhase::Failed => {
            format!("{}: {}", text("Startup failed", "启动失败"), status.error)
        }
        daemon::HubPhase::Stopping => text("Stopping", "正在停止").to_owned(),
        daemon::HubPhase::Unknown => text("Waiting for status", "正在等待状态").to_owned(),
    }
}

fn stop(args: &[String]) -> Result<(), CliError> {
    let args = StopArgs::parse(args)?;
    let lumen_dir = lumen_launcher::default_lumen_dir()?;
    let paths = daemon::daemon_paths(&lumen_dir);

    let Some(pid) = daemon::check_running(&paths.pid_file)? else {
        println!("{}", text("lumen-hub is not running", "lumen-hub 未运行"));
        return Ok(());
    };

    println!(
        "{} (pid {pid})...",
        text("stopping lumen-hub", "正在停止 lumen-hub")
    );
    daemon::stop_process(pid, Duration::from_secs(args.timeout))?;
    daemon::remove_pid_file(&paths.pid_file)?;
    println!("{}", text("lumen-hub stopped", "lumen-hub 已停止"));
    Ok(())
}

fn reload(args: &[String]) -> Result<(), CliError> {
    let args = CommonArgs::parse(args)?;
    let lumen_dir = lumen_launcher::default_lumen_dir()?;
    let paths = daemon::daemon_paths(&lumen_dir);

    if let Some(pid) = daemon::check_running(&paths.pid_file)? {
        println!(
            "{} (pid {pid})...",
            text("stopping lumen-hub", "正在停止 lumen-hub")
        );
        daemon::stop_process(pid, Duration::from_secs(10))?;
        daemon::remove_pid_file(&paths.pid_file)?;
        println!("{}", text("lumen-hub stopped", "lumen-hub 已停止"));
    } else {
        println!(
            "{}",
            text(
                "lumen-hub is not running, starting fresh",
                "lumen-hub 未运行，将直接启动"
            )
        );
    }

    println!(
        "{}",
        text(
            "starting lumen-hub with updated config...",
            "正在使用更新后的配置启动 lumen-hub..."
        )
    );
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
        return Err(CliError::InvalidArgument(match language() {
            Language::En => format!("config `{}` does not exist", config_path.display()),
            Language::ZhCn => format!("配置文件 `{}` 不存在", config_path.display()),
        }));
    }

    let contents = std::fs::read_to_string(&config_path)?;

    match setup::validate_yaml_config(&contents) {
        Ok(()) => {
            println!(
                "{}: {}",
                text("config is valid", "配置有效"),
                config_path.display()
            );
            Ok(())
        }
        Err(e) => Err(CliError::InvalidArgument(format!(
            "config validation failed for `{}`:\n{e}",
            config_path.display()
        ))),
    }
}

// --- Configure ---

fn configure() -> Result<(), CliError> {
    intro(format!(" lumen-cli {VERSION} "))?;
    log::info(text(
        "Configure Lumen Intelligence with a canonical Lumen Hub preset.",
        "使用 Lumen Hub 的规范预设配置 Lumen Intelligence。",
    ))?;

    let paths = setup::default_setup_paths()?;
    let existing = if paths.bootstrap_path.is_file() {
        lumen_launcher::read_bootstrap(&paths.bootstrap_path).ok()
    } else {
        None
    };
    let config_path = existing
        .as_ref()
        .map(|bootstrap| PathBuf::from(&bootstrap.config_path))
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| paths.config_path.clone());
    if config_path.exists() || paths.bootstrap_path.exists() {
        let mut details = String::new();
        if config_path.exists() {
            details.push_str(&format!("config: {}\n", config_path.display()));
        }
        if paths.bootstrap_path.exists() {
            details.push_str(&format!("bootstrap: {}\n", paths.bootstrap_path.display()));
        }
        note(
            text("Existing setup will be reconfigured", "将重新配置现有设置"),
            details.trim_end(),
        )?;
    }

    configure_to_paths(&config_path, &paths.bootstrap_path, existing.as_ref())
}

fn configure_to_paths(
    config_path: &Path,
    bootstrap_path: &Path,
    existing: Option<&Bootstrap>,
) -> Result<(), CliError> {
    let previous_config = read_optional_file(config_path)?;
    let previous_bootstrap = read_optional_file(bootstrap_path)?;
    let system = setup::detect_system();
    let memory = setup::detect_memory();
    let platform = setup::platform_profile(&system)?;

    let mut detected = format!("OS: {}\nArch: {}", system.os_label(), system.arch);
    if let Some(total_gb) = memory.total_gb {
        detected.push_str(&format!("\nRAM: {total_gb:.1} GB"));
    } else {
        detected.push_str(&format!("\nRAM: {}", text("unknown", "未知")));
    }
    note(text("Detected system", "检测到的系统"), detected)?;

    let region_order = if existing.is_some_and(|value| value.region == setup::REGION_CN) {
        [setup::REGION_CN, setup::REGION_OTHER]
    } else {
        [setup::REGION_OTHER, setup::REGION_CN]
    };
    let region_choices = region_order
        .iter()
        .map(|region| {
            if *region == setup::REGION_CN {
                Choice::new("cn - hf-mirror.com", true, None)
            } else {
                Choice::new("other - Hugging Face", true, None)
            }
        })
        .collect::<Vec<_>>();
    let region_index = choose(
        text("Select download region:", "选择下载区域："),
        &region_choices,
    )?;
    let region = region_order[region_index];

    let mut presets = setup::Preset::all().to_vec();
    if let Some(current) = existing.and_then(|value| setup::Preset::by_name(&value.preset))
        && let Some(index) = presets
            .iter()
            .position(|preset| preset.name == current.name)
    {
        presets.swap(0, index);
    }
    let preset_choices = presets
        .iter()
        .map(|preset| {
            let warning = memory
                .total_gb
                .filter(|ram| *ram < preset.min_ram_gb as f64)
                .map(|ram| match language() {
                    Language::En => format!(
                        "detected RAM {ram:.1} GB below recommended {} GB",
                        preset.min_ram_gb
                    ),
                    Language::ZhCn => format!(
                        "检测到 {ram:.1} GB 内存，低于建议的 {} GB",
                        preset.min_ram_gb
                    ),
                });
            Choice::new(preset.label(), true, warning)
        })
        .collect::<Vec<_>>();
    let preset_index = choose(text("Select preset:", "选择预设："), &preset_choices)?;
    let preset = presets[preset_index];

    if let Some(total_gb) = memory.total_gb
        && total_gb < preset.min_ram_gb as f64
    {
        log::warning(match language() {
            Language::En => format!(
                "`{}` recommends at least {} GB RAM; detected {total_gb:.1} GB",
                preset.name, preset.min_ram_gb
            ),
            Language::ZhCn => format!(
                "`{}` 建议至少 {} GB 内存；当前检测到 {total_gb:.1} GB",
                preset.name, preset.min_ram_gb
            ),
        })?;
    }

    let mut backends = setup::backend_choices(platform);
    if let Some(profile) = existing.map(|value| value.release_profile.as_str())
        && let Some(index) = backends.iter().position(|choice| {
            choice
                .backend
                .is_some_and(|backend| backend.release_profile == profile)
        })
    {
        backends.swap(0, index);
    }
    let backend_index = choose_backend(
        text("Select backend package:", "选择后端软件包："),
        &backends,
    )?;
    let backend = backends[backend_index].backend.ok_or_else(|| {
        CliError::InvalidArgument(
            text("selected backend is unavailable", "所选后端当前不可用").to_owned(),
        )
    })?;

    let default_cache = existing
        .map(|value| PathBuf::from(&value.cache_dir))
        .unwrap_or(setup::default_setup_paths()?.cache_dir);
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

    note(
        text("Configuration to apply", "将应用的配置"),
        format!(
            "preset: {}\nregion: {}\nprofile: {}\ncache: {}",
            selection.preset.name,
            selection.region,
            selection.backend.release_profile,
            selection.cache_dir.display()
        ),
    )?;
    if !confirm(text("Apply this configuration?", "应用此配置？"))
        .initial_value(true)
        .interact()?
    {
        cliclack::outro_cancel(text("Configuration cancelled.", "已取消配置。"))?;
        return Ok(());
    }

    let written = setup::write_setup(&selection)?;
    log::success(format!(
        "{}: {}",
        text("Config committed", "配置已提交"),
        written.config_path.display()
    ))?;
    log::success(format!(
        "{}: {}",
        text("Bootstrap committed", "启动信息已提交"),
        written.bootstrap_path.display()
    ))?;

    let daemon_paths = daemon::daemon_paths(&setup::default_setup_paths()?.lumen_dir);
    let running = daemon::check_running(&daemon_paths.pid_file)?.is_some();
    if running {
        let restart_now = confirm(text(
            "Lumen Hub is running. Restart it now to apply the configuration?",
            "Lumen Hub 正在运行。现在重启以应用配置？",
        ))
        .initial_value(true)
        .interact()?;
        if restart_now {
            match reload(&[]) {
                Ok(()) => return Ok(()),
                Err(start_error) => {
                    rollback_running_reconfigure(
                        config_path,
                        previous_config.as_deref(),
                        bootstrap_path,
                        previous_bootstrap.as_deref(),
                        start_error,
                    )?;
                    return Ok(());
                }
            }
        }
        note(
            text("Restart required", "需要重启"),
            text(
                "The new configuration is saved but is not active. Run `lumen-cli reload` when ready.",
                "新配置已保存但尚未生效。准备好后运行 `lumen-cli reload`。",
            ),
        )?;
    } else {
        note(
            text("Next step", "下一步"),
            format!(
                "{}: {}\n  lumen-cli start",
                text("Recommended release profile", "建议发布 profile"),
                written.bootstrap.release_profile,
            ),
        )?;
    }
    outro(text(
        "Lumen Intelligence configuration is ready.",
        "Lumen Intelligence 配置已就绪。",
    ))?;
    Ok(())
}

fn read_optional_file(path: &Path) -> Result<Option<Vec<u8>>, CliError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(CliError::Io(error)),
    }
}

fn rollback_running_reconfigure(
    config_path: &Path,
    previous_config: Option<&[u8]>,
    bootstrap_path: &Path,
    previous_bootstrap: Option<&[u8]>,
    start_error: CliError,
) -> Result<(), CliError> {
    let (Some(previous_config), Some(previous_bootstrap)) = (previous_config, previous_bootstrap)
    else {
        return Err(CliError::Daemon(match language() {
            Language::En => format!(
                "the new configuration could not start and no complete previous setup was available to restore: {start_error}"
            ),
            Language::ZhCn => format!("新配置无法启动，且没有完整的旧配置可供恢复：{start_error}"),
        }));
    };

    setup::replace_setup_files(
        config_path,
        previous_config,
        bootstrap_path,
        previous_bootstrap,
    )?;
    log::warning(text(
        "The new configuration could not start. The previous configuration was restored.",
        "新配置无法启动，已恢复之前的配置。",
    ))?;

    match start_background(&[]) {
        Ok(()) => Err(CliError::Daemon(match language() {
            Language::En => format!(
                "the new configuration was rejected; the previous Lumen Hub configuration is running again: {start_error}"
            ),
            Language::ZhCn => {
                format!("新配置已被拒绝；之前的 Lumen Hub 配置已重新运行：{start_error}")
            }
        })),
        Err(rollback_start_error) => Err(CliError::Daemon(match language() {
            Language::En => format!(
                "the new configuration could not start; the previous files were restored, but Lumen Hub could not be restarted. New error: {start_error}; rollback start error: {rollback_start_error}"
            ),
            Language::ZhCn => format!(
                "新配置无法启动；旧文件已恢复，但 Lumen Hub 也无法重新启动。新配置错误：{start_error}；回滚启动错误：{rollback_start_error}"
            ),
        })),
    }
}

// --- Observer ---

struct CliLaunchObserver;

impl LaunchObserver for CliLaunchObserver {
    fn manifest_fetch_started(&mut self, _url: &str) {
        let _ = log::step(text("fetching release manifest", "正在获取发布清单"));
    }

    fn manifest_fetched(&mut self, version: &str) {
        let _ = log::success(format!(
            "{} {version}",
            text("release manifest", "发布清单")
        ));
    }

    fn hub_already_installed(&mut self, hub_path: &Path) {
        let _ = log::success(format!(
            "{}: {}",
            text("lumen-hub already installed", "lumen-hub 已安装"),
            hub_path.display()
        ));
    }

    fn download_started(&mut self, file_name: &str, total: Option<u64>) {
        let detail = total
            .map(format_bytes)
            .map(|size| format!(" ({size})"))
            .unwrap_or_default();
        let _ = log::step(format!(
            "{} {file_name}{detail}",
            text("downloading", "正在下载")
        ));
    }

    fn download_finished(&mut self, file_name: &str, written: u64) {
        let _ = log::success(format!(
            "{} {file_name} ({})",
            text("downloaded", "已下载"),
            format_bytes(written)
        ));
    }

    fn verify_started(&mut self, path: &Path) {
        let _ = log::step(format!(
            "{} {}",
            text("verifying", "正在校验"),
            path.display()
        ));
    }

    fn verify_finished(&mut self, _path: &Path) {
        let _ = log::success(text("checksum ok", "校验和正确"));
    }

    fn extract_started(&mut self, path: &Path) {
        let _ = log::step(format!(
            "{} {}",
            text("extracting", "正在解压"),
            path.display()
        ));
    }

    fn hub_installed(&mut self, hub_path: &Path) {
        let _ = log::success(format!(
            "{}: {}",
            text("lumen-hub ready", "lumen-hub 已就绪"),
            hub_path.display()
        ));
    }

    fn hub_starting(&mut self, hub_path: &Path) {
        let _ = log::step(format!(
            "{} {}",
            text("starting", "正在启动"),
            hub_path.display()
        ));
    }
}

// --- UI helpers ---

fn prompt_cache_dir(default_cache: &Path, min_disk_gb: u64) -> Result<PathBuf, CliError> {
    loop {
        let selected_input: String = input(text("Model cache directory", "模型缓存目录"))
            .default_input(&setup::display_tilde(default_cache).display().to_string())
            .interact()?;
        let selected = setup::expand_tilde(selected_input.trim());

        if setup::is_dangerous_cache_dir(&selected) {
            log::warning(match language() {
                Language::En => format!(
                    "`{}` is not a safe model cache directory",
                    selected.display()
                ),
                Language::ZhCn => format!("`{}` 不是安全的模型缓存目录", selected.display()),
            })?;
            continue;
        }

        if !selected.exists() {
            let create = confirm(match language() {
                Language::En => format!("Create `{}`?", selected.display()),
                Language::ZhCn => format!("创建 `{}`？", selected.display()),
            })
            .initial_value(true)
            .interact()?;
            if !create {
                continue;
            }
            setup::ensure_cache_dir(&selected)?;
        }

        if !selected.is_dir() {
            log::warning(match language() {
                Language::En => format!("`{}` is not a directory", selected.display()),
                Language::ZhCn => format!("`{}` 不是目录", selected.display()),
            })?;
            continue;
        }
        if !setup::is_writable_dir(&selected) {
            log::warning(match language() {
                Language::En => format!("`{}` is not writable", selected.display()),
                Language::ZhCn => format!("`{}` 不可写", selected.display()),
            })?;
            continue;
        }
        if let Some(free_gb) = setup::free_disk_gb(&selected)
            && free_gb < min_disk_gb as f64
        {
            log::warning(match language() {
                Language::En => format!(
                    "`{}` has {free_gb:.1} GB free; selected preset recommends at least {min_disk_gb} GB",
                    selected.display()
                ),
                Language::ZhCn => format!(
                    "`{}` 仅剩 {free_gb:.1} GB；所选预设建议至少 {min_disk_gb} GB",
                    selected.display()
                ),
            })?;
            let keep = confirm(text("Continue anyway?", "仍然继续？"))
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
    match language() {
        Language::En => println!(
            "\
Usage:
  lumen-cli [--lang en|zh-CN] <command> [options]

Commands:
  configure  Configure or reconfigure Lumen Intelligence
  init       Alias for configure
  run        Run lumen-hub in the foreground (blocks until stopped)
  start      Start lumen-hub in the background
  stop       Stop the background lumen-hub process
  reload     Restart lumen-hub to pick up config changes
  validate   Validate a config file without starting the server

Global options:
  --lang <language>      UI language: en or zh-CN; defaults from LC_ALL, LC_MESSAGES, LANG

Options for run/start/reload:
  --config <path>        Path to config YAML
  --bootstrap <path>     Path to bootstrap JSON
  --manifest-url <url>   Override release manifest URL
  --profile <profile>    Select a dist profile

Options for stop:
  --timeout <secs>       Grace period before force kill (default: 10)

Options for validate:
  --config <path>        Path to config YAML to validate"
        ),
        Language::ZhCn => println!(
            "\
用法：
  lumen-cli [--lang en|zh-CN] <命令> [选项]

命令：
  configure  配置或重新配置 Lumen Intelligence
  init       configure 的兼容别名
  run        在前台运行 lumen-hub（阻塞直到停止）
  start      在后台启动 lumen-hub
  stop       停止后台 lumen-hub 进程
  reload     重启 lumen-hub 以应用配置变更
  validate   在不启动服务器的情况下验证配置

全局选项：
  --lang <语言>          界面语言：en 或 zh-CN；默认依次读取 LC_ALL、LC_MESSAGES、LANG

run/start/reload 选项：
  --config <路径>        配置 YAML 路径
  --bootstrap <路径>     bootstrap JSON 路径
  --manifest-url <URL>   覆盖 release manifest URL
  --profile <profile>    选择发布 profile

stop 选项：
  --timeout <秒>         强制停止前的宽限期（默认：10）

validate 选项：
  --config <路径>        要验证的配置 YAML 路径"
        ),
    }
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

    #[test]
    fn language_flag_is_removed_before_command_parsing() {
        let args = vec![
            "lumen-cli".to_owned(),
            "--lang=zh-CN".to_owned(),
            "validate".to_owned(),
        ];
        let (language, args) = extract_language(args).unwrap();
        assert_eq!(language, Language::ZhCn);
        assert_eq!(args, vec!["lumen-cli", "validate"]);
    }

    #[test]
    fn language_parser_accepts_locale_forms_and_rejects_unknown_languages() {
        assert_eq!(Language::parse_locale("zh_CN.UTF-8"), Some(Language::ZhCn));
        assert_eq!(Language::parse_locale("zh-Hans-CN"), Some(Language::ZhCn));
        assert_eq!(Language::parse_locale("zh-TW"), None);
        assert_eq!(Language::parse_locale("en-US"), Some(Language::En));
        assert_eq!(Language::parse_locale("fr-FR"), None);
        assert_eq!(Language::parse_explicit("zh-CN"), Some(Language::ZhCn));
        assert_eq!(Language::parse_explicit("zh-TW"), None);
        assert_eq!(Language::parse_explicit("en-US"), None);
    }
}
