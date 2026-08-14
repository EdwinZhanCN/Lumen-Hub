mod lang;

rust_i18n::i18n!("locales", fallback = "en");

use std::{
    env, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use cliclack::{confirm, input, intro, log, multiselect, note, outro, select};
use lang::{LANGUAGE, extract_language, language};
use lumen_launcher::{
    Bootstrap, LaunchObserver, LauncherError, StartOptions, daemon, format_bytes, prepare_hub,
    read_server_port, resolve_start_plan, setup, spawn_hub,
};
use rust_i18n::t;
use thiserror::Error;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn display_lang() -> setup::DisplayLang {
    match language() {
        lang::Language::En => setup::DisplayLang::En,
        lang::Language::ZhCn => setup::DisplayLang::ZhCn,
    }
}

fn plan_details(
    config: impl std::fmt::Display,
    profile: impl std::fmt::Display,
    manifest: impl std::fmt::Display,
) -> String {
    format!(
        "{}: {}\n{}: {}\n{}: {}",
        t!("common.config"),
        config,
        t!("common.profile"),
        profile,
        t!("common.manifest"),
        manifest
    )
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
    rust_i18n::set_locale(lang.as_str());
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Help) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", t!("errors.prefix"));
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
        Some(other) => Err(CliError::InvalidArgument(
            t!("errors.unknown_command", command = other).into(),
        )),
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
        t!("run.plan_title"),
        plan_details(
            plan.config_path.display(),
            &plan.profile,
            &plan.manifest_url,
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

    outro(t!("run.outro"))?;
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
        return Err(CliError::InvalidArgument(
            t!("errors.already_running", pid = pid).into(),
        ));
    }

    note(
        t!("start.plan_title"),
        plan_details(
            plan.config_path.display(),
            &plan.profile,
            &plan.manifest_url,
        ),
    )?;

    let mut observer = CliLaunchObserver;
    let hub = prepare_hub(&plan, &mut observer)?;

    log::step(t!("start.starting_background"))?;
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

    log::step(t!("start.waiting_ready", addr = addr))?;
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
            log::success(t!("start.started", pid = pid))?;
            log::info(format!(
                "{}: {}",
                t!("common.logs"),
                paths.log_file.display()
            ))?;
            outro(t!("start.outro"))?;
            Ok(())
        }
        Err(e) => {
            let _ = daemon::stop_process(pid, Duration::from_secs(5));
            daemon::remove_pid_file(&paths.pid_file)?;
            Err(CliError::Daemon(
                t!(
                    "start.check_logs",
                    error = e,
                    path = paths.log_file.display()
                )
                .into(),
            ))
        }
    }
}

fn hub_status_message(status: &daemon::HubStatus) -> String {
    match status.phase {
        daemon::HubPhase::Starting => t!("status.starting").into(),
        daemon::HubPhase::Downloading => {
            let Some(progress) = &status.download else {
                return t!("status.downloading").into();
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
            t!(
                "status.downloading_progress",
                model = progress.model.as_str(),
                file = progress.file.as_str(),
                bytes = bytes,
                files_done = progress.files_done,
                files_total = progress.files_total
            )
            .into()
        }
        daemon::HubPhase::Loading => t!("status.loading").into(),
        daemon::HubPhase::Warmup => t!("status.warmup").into(),
        daemon::HubPhase::Ready => t!("status.ready").into(),
        daemon::HubPhase::Failed => t!("status.failed", error = status.error.as_str()).into(),
        daemon::HubPhase::Stopping => t!("status.stopping").into(),
        daemon::HubPhase::Unknown => t!("status.waiting").into(),
    }
}

fn stop(args: &[String]) -> Result<(), CliError> {
    let args = StopArgs::parse(args)?;
    let lumen_dir = lumen_launcher::default_lumen_dir()?;
    let paths = daemon::daemon_paths(&lumen_dir);

    let Some(pid) = daemon::check_running(&paths.pid_file)? else {
        println!("{}", t!("stop.not_running"));
        return Ok(());
    };

    println!("{}", t!("stop.stopping", pid = pid));
    daemon::stop_process(pid, Duration::from_secs(args.timeout))?;
    daemon::remove_pid_file(&paths.pid_file)?;
    println!("{}", t!("stop.stopped"));
    Ok(())
}

fn reload(args: &[String]) -> Result<(), CliError> {
    let args = CommonArgs::parse(args)?;
    let lumen_dir = lumen_launcher::default_lumen_dir()?;
    let paths = daemon::daemon_paths(&lumen_dir);

    if let Some(pid) = daemon::check_running(&paths.pid_file)? {
        println!("{}", t!("stop.stopping", pid = pid));
        daemon::stop_process(pid, Duration::from_secs(10))?;
        daemon::remove_pid_file(&paths.pid_file)?;
        println!("{}", t!("stop.stopped"));
    } else {
        println!("{}", t!("reload.not_running"));
    }

    println!("{}", t!("reload.starting"));
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
        return Err(CliError::InvalidArgument(
            t!("errors.config_missing", path = config_path.display()).into(),
        ));
    }

    let contents = std::fs::read_to_string(&config_path)?;

    match setup::validate_yaml_config(&contents) {
        Ok(()) => {
            println!("{}", t!("validate.ok", path = config_path.display()));
            Ok(())
        }
        Err(e) => Err(CliError::InvalidArgument(
            t!(
                "errors.config_invalid",
                path = config_path.display(),
                error = e
            )
            .into(),
        )),
    }
}

// --- Configure ---

fn configure() -> Result<(), CliError> {
    intro(format!(" lumen-cli {VERSION} "))?;
    log::info(t!("configure.intro"))?;

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
            details.push_str(&format!(
                "{}: {}\n",
                t!("common.config"),
                config_path.display()
            ));
        }
        if paths.bootstrap_path.exists() {
            details.push_str(&format!("bootstrap: {}\n", paths.bootstrap_path.display()));
        }
        note(t!("configure.reconfigure"), details.trim_end())?;
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

    let mut detected = format!(
        "{}\n{}",
        t!("configure.detected_os", os = system.os_label()),
        t!("configure.detected_arch", arch = system.arch.as_str())
    );
    if let Some(total_gb) = memory.total_gb {
        detected.push_str(&format!(
            "\n{}",
            t!("configure.detected_ram", ram = format!("{total_gb:.1} GB"))
        ));
    } else {
        detected.push_str(&format!(
            "\n{}",
            t!("configure.detected_ram", ram = t!("common.unknown"))
        ));
    }
    note(t!("configure.detected_system"), detected)?;

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
    let region_index = choose(&t!("configure.select_region"), &region_choices)?;
    let region = region_order[region_index];

    let intent = prompt_setup_intent(existing, &memory)?;

    if let setup::SetupIntent::Preset(preset) = &intent
        && let Some(total_gb) = memory.total_gb
        && total_gb < preset.min_ram_gb as f64
    {
        log::warning(t!(
            "configure.ram_warning",
            preset = preset.name,
            recommended = preset.min_ram_gb,
            ram = format!("{total_gb:.1}")
        ))?;
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
    let backend_index = choose_backend(&t!("configure.select_backend"), &backends)?;
    let backend = backends[backend_index]
        .backend
        .ok_or_else(|| CliError::InvalidArgument(t!("errors.backend_unavailable").into()))?;

    let default_cache = existing
        .map(|value| PathBuf::from(&value.cache_dir))
        .unwrap_or(setup::default_setup_paths()?.cache_dir);
    let cache_dir = prompt_cache_dir(&default_cache, intent.min_disk_gb())?;

    let selection = setup::SetupSelection {
        version: VERSION.to_owned(),
        region: region.to_owned(),
        intent,
        platform,
        backend,
        cache_dir,
        config_path: config_path.to_path_buf(),
        bootstrap_path: bootstrap_path.to_path_buf(),
    };

    note(t!("configure.apply_title"), apply_details(&selection))?;
    if !confirm(t!("configure.confirm"))
        .initial_value(true)
        .interact()?
    {
        cliclack::outro_cancel(t!("configure.cancelled"))?;
        return Ok(());
    }

    let written = setup::write_setup(&selection)?;
    log::success(t!(
        "configure.config_committed",
        path = written.config_path.display()
    ))?;
    log::success(t!(
        "configure.bootstrap_committed",
        path = written.bootstrap_path.display()
    ))?;

    let daemon_paths = daemon::daemon_paths(&setup::default_setup_paths()?.lumen_dir);
    let running = daemon::check_running(&daemon_paths.pid_file)?.is_some();
    if running {
        let restart_now = confirm(t!("configure.restart_now"))
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
            t!("configure.restart_required"),
            t!("configure.restart_later"),
        )?;
    } else {
        note(
            t!("configure.next_step"),
            t!(
                "configure.recommended_profile",
                profile = written.bootstrap.release_profile.as_str()
            ),
        )?;
    }
    outro(t!("configure.ready"))?;
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
        return Err(CliError::Daemon(
            t!("errors.rollback_no_previous", error = start_error).into(),
        ));
    };

    setup::replace_setup_files(
        config_path,
        previous_config,
        bootstrap_path,
        previous_bootstrap,
    )?;
    log::warning(t!("configure.rollback_restored"))?;

    match start_background(&[]) {
        Ok(()) => Err(CliError::Daemon(
            t!("errors.rollback_rejected", error = start_error).into(),
        )),
        Err(rollback_start_error) => Err(CliError::Daemon(
            t!(
                "errors.rollback_restart_failed",
                error = start_error,
                rollback = rollback_start_error
            )
            .into(),
        )),
    }
}

// --- Observer ---

struct CliLaunchObserver;

impl LaunchObserver for CliLaunchObserver {
    fn manifest_fetch_started(&mut self, _url: &str) {
        let _ = log::step(t!("launch.fetching_manifest"));
    }

    fn manifest_fetched(&mut self, version: &str) {
        let _ = log::success(t!("launch.manifest", version = version));
    }

    fn hub_already_installed(&mut self, hub_path: &Path) {
        let _ = log::success(t!("launch.already_installed", path = hub_path.display()));
    }

    fn download_started(&mut self, file_name: &str, total: Option<u64>) {
        let detail = total
            .map(format_bytes)
            .map(|size| format!(" ({size})"))
            .unwrap_or_default();
        let _ = log::step(t!("launch.downloading", file = file_name, detail = detail));
    }

    fn download_finished(&mut self, file_name: &str, written: u64) {
        let _ = log::success(t!(
            "launch.downloaded",
            file = file_name,
            size = format_bytes(written)
        ));
    }

    fn verify_started(&mut self, path: &Path) {
        let _ = log::step(t!("launch.verifying", path = path.display()));
    }

    fn verify_finished(&mut self, _path: &Path) {
        let _ = log::success(t!("launch.checksum_ok"));
    }

    fn extract_started(&mut self, path: &Path) {
        let _ = log::step(t!("launch.extracting", path = path.display()));
    }

    fn hub_installed(&mut self, hub_path: &Path) {
        let _ = log::success(t!("launch.ready", path = hub_path.display()));
    }

    fn hub_starting(&mut self, hub_path: &Path) {
        let _ = log::step(t!("launch.starting", path = hub_path.display()));
    }
}

// --- UI helpers ---

fn prompt_setup_intent(
    existing: Option<&Bootstrap>,
    memory: &setup::MemoryInfo,
) -> Result<setup::SetupIntent, CliError> {
    let mut presets = setup::Preset::all().to_vec();
    if let Some(current) = existing.and_then(|value| setup::Preset::by_name(&value.preset))
        && let Some(index) = presets
            .iter()
            .position(|preset| preset.name == current.name)
    {
        presets.swap(0, index);
    }
    let custom_first = existing.is_some_and(|value| value.preset == "custom");
    let lang = display_lang();
    let mut choices = presets
        .iter()
        .map(|preset| {
            let warning = memory
                .total_gb
                .filter(|ram| *ram < preset.min_ram_gb as f64)
                .map(|ram| {
                    t!(
                        "configure.ram_below",
                        ram = format!("{ram:.1}"),
                        recommended = preset.min_ram_gb
                    )
                    .into()
                });
            Choice::new(
                t!(
                    "configure.preset_label",
                    title = preset.display_title(lang),
                    capabilities = preset.capability_summary(lang),
                    ram = preset.min_ram_gb,
                    vram = preset.min_vram_gb
                ),
                true,
                warning,
            )
        })
        .collect::<Vec<_>>();
    let custom_choice = Choice::new(t!("configure.custom_label"), true, None);
    if custom_first {
        choices.insert(0, custom_choice);
    } else {
        choices.push(custom_choice);
    }
    let preset_index = choose(&t!("configure.select_preset"), &choices)?;
    let selected_custom = if custom_first {
        preset_index == 0
    } else {
        preset_index == presets.len()
    };
    if selected_custom {
        prompt_custom_intent(existing)
    } else {
        let official_index = if custom_first {
            preset_index - 1
        } else {
            preset_index
        };
        Ok(setup::SetupIntent::Preset(presets[official_index]))
    }
}

fn prompt_custom_intent(existing: Option<&Bootstrap>) -> Result<setup::SetupIntent, CliError> {
    let lang = display_lang();
    let initial_services = custom_initial_services(existing);
    let mut prompt = multiselect(t!("configure.select_capabilities"));
    for service in setup::SERVICE_ORDER {
        let label = setup::capability_term(service)
            .map(|term| term.label(lang))
            .unwrap_or(service);
        prompt = prompt.item(service, label, "");
    }
    let selected: Vec<&str> = prompt.initial_values(initial_services).interact()?;
    let services = setup::SERVICE_ORDER
        .iter()
        .copied()
        .filter(|service| selected.contains(service))
        .collect::<Vec<_>>();
    if services.is_empty() {
        return Err(CliError::InvalidArgument(
            t!("errors.empty_capabilities").into(),
        ));
    }

    let siglip_model = if services.contains(&"siglip") {
        Some(
            choose_str(
                &t!("configure.select_siglip_model"),
                &setup::SIGLIP_MODELS,
                custom_initial_siglip_model(existing),
            )?
            .to_owned(),
        )
    } else {
        None
    };
    let bioclip_dataset = if services.contains(&"bioclip") {
        Some(
            choose_str(
                &t!("configure.select_bioclip_dataset"),
                &setup::BIOCLIP_DATASETS,
                custom_initial_bioclip_dataset(existing),
            )?
            .to_owned(),
        )
    } else {
        None
    };

    Ok(setup::SetupIntent::Custom {
        services: services.into_iter().map(str::to_owned).collect(),
        siglip_model,
        bioclip_dataset,
    })
}

fn custom_initial_services(existing: Option<&Bootstrap>) -> Vec<&'static str> {
    if let Some(existing) = existing {
        if existing.preset == "custom"
            && let Some(services) = existing.services.as_ref()
        {
            return setup::SERVICE_ORDER
                .iter()
                .copied()
                .filter(|service| services.iter().any(|selected| selected == service))
                .collect();
        }
        if let Some(preset) = setup::Preset::by_name(&existing.preset) {
            return preset.components.to_vec();
        }
    }
    vec!["siglip", "face"]
}

fn custom_initial_siglip_model(existing: Option<&Bootstrap>) -> Option<&'static str> {
    let preferred = existing.and_then(|value| {
        value
            .siglip_model
            .as_deref()
            .or_else(|| setup::Preset::by_name(&value.preset).map(|preset| preset.siglip_model))
    })?;
    setup::SIGLIP_MODELS
        .iter()
        .copied()
        .find(|model| *model == preferred)
}

fn custom_initial_bioclip_dataset(existing: Option<&Bootstrap>) -> Option<&'static str> {
    let preferred = existing.and_then(|value| {
        value.bioclip_dataset.as_deref().or_else(|| {
            setup::Preset::by_name(&value.preset)
                .and_then(|preset| preset.bioclip_dataset)
                .or(Some(setup::BIOCLIP_CORE_DATASET))
        })
    })?;
    setup::BIOCLIP_DATASETS
        .iter()
        .copied()
        .find(|dataset| *dataset == preferred)
        .or(Some(setup::BIOCLIP_CORE_DATASET))
}

fn choose_str<'a>(
    prompt: &str,
    options: &'a [&'a str],
    preferred: Option<&str>,
) -> Result<&'a str, CliError> {
    let mut items = options.to_vec();
    if let Some(preferred) = preferred
        && let Some(index) = items.iter().position(|item| *item == preferred)
    {
        items.swap(0, index);
    }
    let choices = items
        .iter()
        .map(|item| Choice::new(*item, true, None))
        .collect::<Vec<_>>();
    let index = choose(prompt, &choices)?;
    Ok(items[index])
}

fn apply_details(selection: &setup::SetupSelection) -> String {
    let mut lines = vec![
        t!(
            "configure.apply_details",
            preset = selection.intent.name(),
            region = selection.region.as_str(),
            profile = selection.backend.release_profile,
            cache = selection.cache_dir.display()
        )
        .to_string(),
    ];
    if let setup::SetupIntent::Custom {
        services,
        siglip_model,
        bioclip_dataset,
    } = &selection.intent
    {
        let lang = display_lang();
        let capabilities = services
            .iter()
            .filter_map(|service| setup::capability_term(service))
            .map(|term| term.label(lang))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("{}: {capabilities}", t!("configure.capabilities")));
        if let Some(model) = siglip_model {
            lines.push(format!("{}: {model}", t!("configure.siglip_model")));
        }
        if let Some(dataset) = bioclip_dataset {
            lines.push(format!("{}: {dataset}", t!("configure.bioclip_dataset")));
        }
    }
    lines.join("\n")
}

fn prompt_cache_dir(default_cache: &Path, min_disk_gb: u64) -> Result<PathBuf, CliError> {
    loop {
        let selected_input: String = input(t!("configure.cache_dir"))
            .default_input(&setup::display_tilde(default_cache).display().to_string())
            .interact()?;
        let selected = setup::expand_tilde(selected_input.trim());

        if setup::is_dangerous_cache_dir(&selected) {
            log::warning(t!("configure.cache_unsafe", path = selected.display()))?;
            continue;
        }

        if !selected.exists() {
            let create = confirm(t!("configure.cache_create", path = selected.display()))
                .initial_value(true)
                .interact()?;
            if !create {
                continue;
            }
            setup::ensure_cache_dir(&selected)?;
        }

        if !selected.is_dir() {
            log::warning(t!("configure.cache_not_dir", path = selected.display()))?;
            continue;
        }
        if !setup::is_writable_dir(&selected) {
            log::warning(t!(
                "configure.cache_not_writable",
                path = selected.display()
            ))?;
            continue;
        }
        if let Some(free_gb) = setup::free_disk_gb(&selected)
            && free_gb < min_disk_gb as f64
        {
            log::warning(t!(
                "configure.cache_low_disk",
                path = selected.display(),
                free = format!("{free_gb:.1}"),
                recommended = min_disk_gb
            ))?;
            let keep = confirm(t!("configure.continue_anyway"))
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
                    return Err(CliError::InvalidArgument(
                        t!("errors.unknown_argument", arg = other).into(),
                    ));
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
                    timeout = next_value(&mut iter, arg)?.parse().map_err(|_| {
                        CliError::InvalidArgument(t!("errors.invalid_timeout").into())
                    })?;
                }
                value if value.starts_with("--timeout=") => {
                    timeout = value
                        .trim_start_matches("--timeout=")
                        .parse()
                        .map_err(|_| {
                            CliError::InvalidArgument(t!("errors.invalid_timeout").into())
                        })?;
                }
                "--help" | "-h" => {
                    print_help();
                    return Err(CliError::Help);
                }
                other => {
                    return Err(CliError::InvalidArgument(
                        t!("errors.unknown_stop_argument", arg = other).into(),
                    ));
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
                    return Err(CliError::InvalidArgument(
                        t!("errors.unknown_validate_argument", arg = other).into(),
                    ));
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
    iter.next().map(String::as_str).ok_or_else(|| {
        CliError::InvalidArgument(t!("errors.missing_flag_value", flag = flag).into())
    })
}

fn print_help() {
    println!("{}", t!("help"));
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
    fn locale_files_have_matching_keys() {
        let en = load_locale_keys(include_str!("../locales/en.yml"));
        let zh = load_locale_keys(include_str!("../locales/zh-CN.yml"));
        let mut missing_zh: Vec<_> = en.difference(&zh).cloned().collect();
        let mut missing_en: Vec<_> = zh.difference(&en).cloned().collect();
        missing_zh.sort();
        missing_en.sort();
        assert!(
            missing_zh.is_empty() && missing_en.is_empty(),
            "locale key mismatch; missing in zh-CN: {missing_zh:?}; missing in en: {missing_en:?}"
        );
    }

    #[test]
    fn translations_cover_known_keys_and_interpolation() {
        let mut locales: Vec<_> = rust_i18n::available_locales!()
            .into_iter()
            .map(|locale| locale.to_string())
            .collect();
        locales.sort();
        assert_eq!(locales, vec!["en".to_owned(), "zh-CN".to_owned()]);

        for key in [
            "errors.unknown_command",
            "errors.already_running",
            "help",
            "configure.preset_label",
            "status.downloading_progress",
        ] {
            let en = t!(key, locale = "en");
            let zh = t!(key, locale = "zh-CN");
            assert_ne!(en.as_ref(), key, "missing English translation for {key}");
            assert_ne!(zh.as_ref(), key, "missing Chinese translation for {key}");
            assert_ne!(
                en.as_ref(),
                zh.as_ref(),
                "{key} is identical in both locales"
            );
        }

        assert_eq!(
            t!("errors.already_running", locale = "en", pid = 12).as_ref(),
            "lumen-hub is already running (pid 12)"
        );
        assert_eq!(
            t!("errors.already_running", locale = "zh-CN", pid = 12).as_ref(),
            "lumen-hub 已在运行（pid 12）"
        );
        assert_eq!(
            t!("errors.unknown_command", locale = "en", command = "foo").as_ref(),
            "unknown command `foo`"
        );
        assert_eq!(
            t!("errors.unknown_command", locale = "zh-CN", command = "foo").as_ref(),
            "未知命令 `foo`"
        );
    }

    fn load_locale_keys(raw: &str) -> std::collections::BTreeSet<String> {
        let value: serde_yaml::Value = serde_yaml::from_str(raw).expect("locale YAML parses");
        let mut keys = std::collections::BTreeSet::new();
        flatten_keys(&value, "", &mut keys);
        keys
    }

    fn flatten_keys(
        value: &serde_yaml::Value,
        prefix: &str,
        out: &mut std::collections::BTreeSet<String>,
    ) {
        match value {
            serde_yaml::Value::Mapping(map) => {
                for (key, child) in map {
                    let key = key.as_str().expect("locale keys are strings");
                    if key == "_version" {
                        continue;
                    }
                    let next = if prefix.is_empty() {
                        key.to_owned()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    flatten_keys(child, &next, out);
                }
            }
            _ => {
                out.insert(prefix.to_owned());
            }
        }
    }
}
