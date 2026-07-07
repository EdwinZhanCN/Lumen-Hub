#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use lumen_launcher::{
    HubStdio, LaunchObserver, LauncherError, RunningHub, StartOptions, format_bytes, prepare_hub,
    resolve_start_plan, setup, spawn_hub,
};
use slint::{ComponentHandle, SharedString, Timer, TimerMode};

slint::include_modules!();

mod i18n;
mod wizard;

const MAX_LOG_LINES: usize = 600;

fn main() -> Result<(), slint::PlatformError> {
    let app = AppWindow::new()?;
    let process_state = Arc::new(Mutex::new(ProcessState::default()));
    let (tx, rx) = mpsc::channel::<UiMessage>();

    configure_initial_ui(&app);

    let weak = app.as_weak();
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_millis(100), move || {
        while let Ok(message) = rx.try_recv() {
            let Some(app) = weak.upgrade() else {
                return;
            };
            apply_message(&app, message);
        }
    });

    let start_tx = tx.clone();
    let start_process_state = Arc::clone(&process_state);
    let weak = app.as_weak();
    app.on_start_requested(move || {
        let options = weak
            .upgrade()
            .map(|app| start_options_from_ui(&app))
            .unwrap_or_default();
        {
            let mut state = match start_process_state.lock() {
                Ok(state) => state,
                Err(_) => {
                    let _ =
                        start_tx.send(UiMessage::Error("hub state lock was poisoned".to_owned()));
                    return;
                }
            };
            if state.launching || state.hub.is_some() {
                let _ = start_tx.send(UiMessage::Log(
                    "lumen-hub is already starting or running".to_owned(),
                ));
                return;
            }
            state.launching = true;
        }

        let _ = start_tx.send(UiMessage::Controls {
            can_start: false,
            can_stop: false,
        });
        let _ = start_tx.send(UiMessage::Status(i18n::status_preparing().to_string()));
        let tx = start_tx.clone();
        let process_state = Arc::clone(&start_process_state);
        thread::spawn(move || start_hub(tx, process_state, options));
    });

    let stop_tx = tx.clone();
    let stop_process_state = Arc::clone(&process_state);
    app.on_stop_requested(move || {
        let mut state = match stop_process_state.lock() {
            Ok(state) => state,
            Err(_) => {
                let _ = stop_tx.send(UiMessage::Error("hub state lock was poisoned".to_owned()));
                return;
            }
        };
        if let Some(hub) = state.hub.as_mut() {
            match hub.kill() {
                Ok(()) => {
                    let _ = stop_tx.send(UiMessage::Status(i18n::status_stopping().to_string()));
                    let _ =
                        stop_tx.send(UiMessage::Log("sent stop signal to lumen-hub".to_owned()));
                }
                Err(error) => {
                    let _ = stop_tx.send(UiMessage::Error(format!(
                        "failed to stop lumen-hub: {error}"
                    )));
                }
            }
        } else if state.launching {
            let _ = stop_tx.send(UiMessage::Log(
                "lumen-hub is still preparing; stop after it starts".to_owned(),
            ));
        } else {
            let _ = stop_tx.send(UiMessage::Log("lumen-hub is not running".to_owned()));
        }
    });

    let weak = app.as_weak();
    app.on_setup_selection_changed(move || {
        if let Some(app) = weak.upgrade() {
            wizard::sync_wizard(&app);
        }
    });

    let weak = app.as_weak();
    app.on_setup_conflict_action_changed(move || {
        if let Some(app) = weak.upgrade() {
            apply_conflict_action(&app);
            wizard::sync_wizard(&app);
        }
    });

    let weak = app.as_weak();
    app.on_setup_back_requested(move || {
        if let Some(app) = weak.upgrade() {
            wizard::go_back(&app);
        }
    });

    let weak = app.as_weak();
    app.on_setup_next_requested(move || {
        if let Some(app) = weak.upgrade() {
            match wizard::go_next(&app) {
                Ok(()) => {}
                Err(error) => append_log(&app, &error),
            }
        }
    });

    let setup_tx = tx.clone();
    let weak = app.as_weak();
    app.on_setup_create_requested(move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        match build_setup_request(&app) {
            Ok(request) => {
                let _ = setup_tx.send(UiMessage::Status(i18n::status_creating_setup().to_string()));
                let _ = setup_tx.send(UiMessage::Controls {
                    can_start: false,
                    can_stop: false,
                });
                let tx = setup_tx.clone();
                thread::spawn(move || match create_setup(request) {
                    Ok(summary) => {
                        let _ = tx.send(UiMessage::SetupReady(summary));
                    }
                    Err(error) => {
                        let _ = tx.send(UiMessage::SetupRequired(error));
                    }
                });
            }
            Err(error) => {
                append_log(&app, &format!("setup required: {error}"));
                set_setup_required(&app, Some(error));
            }
        }
    });

    let weak = app.as_weak();
    app.on_browse_cache_dir_requested(move || {
        if let Some(app) = weak.upgrade() {
            apply_picker_result(&app, PickerTarget::CacheDir);
        }
    });

    let weak = app.as_weak();
    app.on_browse_config_path_requested(move || {
        if let Some(app) = weak.upgrade() {
            apply_picker_result(&app, PickerTarget::ConfigPath);
        }
    });

    let weak = app.as_weak();
    app.on_browse_bootstrap_path_requested(move || {
        if let Some(app) = weak.upgrade() {
            apply_picker_result(&app, PickerTarget::BootstrapPath);
        }
    });

    app.run()
}

fn configure_initial_ui(app: &AppWindow) {
    i18n::apply_static_labels(app);
    app.set_status_text(i18n::status_checking_setup());
    app.set_profile_text(i18n::bootstrap_missing(&default_bootstrap_label()));
    app.set_config_path(default_config_label().into());
    app.set_detected_text("".into());
    app.set_setup_warning("".into());
    app.set_setup_review_text("".into());
    app.set_setup_visible(false);
    app.set_existing_visible(false);
    app.set_can_start(false);
    app.set_can_stop(false);
    app.set_selected_region_index(0);
    app.set_selected_preset_index(1);
    app.set_selected_backend_index(0);
    app.set_selected_existing_action(0);
    app.set_advanced_settings_open(false);
    app.set_log_text(i18n::log_ready());
    configure_setup_defaults(app);
    refresh_setup_state(app);
}

fn configure_setup_defaults(app: &AppWindow) {
    if let Ok(paths) = setup::default_setup_paths() {
        app.set_cache_dir_input(paths.cache_dir.display().to_string().into());
        app.set_config_path_input(paths.config_path.display().to_string().into());
        app.set_bootstrap_path_input(paths.bootstrap_path.display().to_string().into());
        app.set_existing_visible(paths.config_path.exists() || paths.bootstrap_path.exists());
    }

    let system = setup::detect_system();
    let memory = setup::detect_memory();
    let ram = if let Some(total_gb) = memory.total_gb {
        i18n::bi(&i18n::ram_known(total_gb), &i18n::ram_known_zh(total_gb)).to_string()
    } else {
        i18n::bi(i18n::ram_unknown_en(), i18n::ram_unknown_zh()).to_string()
    };
    app.set_detected_text(i18n::detected_system(
        &system.os_label(),
        &system.arch,
        &ram,
    ));

    match setup::platform_profile(&system) {
        Ok(platform) => configure_backend_choices(app, platform),
        Err(error) => {
            clear_backend_choices(app);
            app.set_setup_warning(error.to_string().into());
        }
    }
    wizard::sync_wizard(app);
}

fn configure_backend_choices(app: &AppWindow, platform: setup::PlatformProfile) {
    let choices = setup::backend_choices(platform);

    let mut first_enabled = None;
    for index in 0..4 {
        if let Some(choice) = choices.get(index) {
            let (choice_title, choice_detail, choice_enabled) = backend_choice_display(choice);
            set_backend_slot(app, index, &choice_title, &choice_detail, choice_enabled);
            if choice_enabled && first_enabled.is_none() {
                first_enabled = Some(index as i32);
            }
        } else {
            set_backend_slot(app, index, "", "", false);
        }
    }

    if let Some(index) = first_enabled {
        let selected = app.get_selected_backend_index();
        let selected_enabled = choices
            .get(selected.max(0) as usize)
            .and_then(|choice| choice.backend)
            .is_some();
        if !selected_enabled {
            app.set_selected_backend_index(index);
        }
    }
}

fn set_backend_slot(app: &AppWindow, index: usize, title: &str, detail: &str, enabled: bool) {
    match index {
        0 => {
            app.set_backend_0_title(title.into());
            app.set_backend_0_detail(detail.into());
            app.set_backend_0_enabled(enabled);
        }
        1 => {
            app.set_backend_1_title(title.into());
            app.set_backend_1_detail(detail.into());
            app.set_backend_1_enabled(enabled);
        }
        2 => {
            app.set_backend_2_title(title.into());
            app.set_backend_2_detail(detail.into());
            app.set_backend_2_enabled(enabled);
        }
        3 => {
            app.set_backend_3_title(title.into());
            app.set_backend_3_detail(detail.into());
            app.set_backend_3_enabled(enabled);
        }
        _ => {}
    }
}

fn clear_backend_choices(app: &AppWindow) {
    app.set_backend_0_title("".into());
    app.set_backend_0_detail("".into());
    app.set_backend_0_enabled(false);
    app.set_backend_1_title("".into());
    app.set_backend_1_detail("".into());
    app.set_backend_1_enabled(false);
    app.set_backend_2_title("".into());
    app.set_backend_2_detail("".into());
    app.set_backend_2_enabled(false);
    app.set_backend_3_title("".into());
    app.set_backend_3_detail("".into());
    app.set_backend_3_enabled(false);
}

fn backend_choice_display(choice: &setup::BackendChoice) -> (String, String, bool) {
    if let Some(backend) = choice.backend {
        (
            backend.name.to_owned(),
            backend.release_profile.to_owned(),
            true,
        )
    } else {
        (
            choice
                .label
                .split_whitespace()
                .next()
                .unwrap_or("unavailable")
                .to_owned(),
            choice
                .disabled_reason
                .clone()
                .unwrap_or_else(|| "Unavailable".to_owned()),
            false,
        )
    }
}

fn refresh_setup_state(app: &AppWindow) {
    match resolve_start_plan(StartOptions::default()) {
        Ok(plan) => {
            app.set_status_text(i18n::status_ready());
            app.set_config_path(plan.config_path.display().to_string().into());
            app.set_profile_text(profile_label(&plan));
            app.set_setup_visible(false);
            app.set_can_start(true);
            app.set_can_stop(false);
        }
        Err(error) => set_setup_required(app, Some(error.to_string())),
    }
}

fn set_setup_required(app: &AppWindow, reason: Option<String>) {
    app.set_status_text(i18n::status_setup_required());
    app.set_config_path(default_config_label().into());
    app.set_profile_text(i18n::bootstrap_missing(&default_bootstrap_label()));
    app.set_setup_visible(true);
    app.set_can_start(false);
    app.set_can_stop(false);
    wizard::reset_wizard(app);
    if let Some(reason) = reason {
        app.set_setup_warning(reason.into());
    }
}

fn profile_label(plan: &lumen_launcher::StartPlan) -> slint::SharedString {
    match &plan.bootstrap {
        Some(bootstrap) => i18n::profile_summary(
            &bootstrap.preset,
            &bootstrap.backend,
            &bootstrap.release_profile,
        ),
        None => i18n::profile_only(&plan.profile),
    }
}

fn apply_conflict_action(app: &AppWindow) {
    let Ok(paths) = setup::default_setup_paths() else {
        return;
    };
    match app.get_selected_existing_action() {
        0 => {
            app.set_config_path_input(paths.config_path.display().to_string().into());
            app.set_bootstrap_path_input(paths.bootstrap_path.display().to_string().into());
        }
        1 => {
            app.set_config_path_input(
                paths
                    .lumen_dir
                    .join("config.generated.yaml")
                    .display()
                    .to_string()
                    .into(),
            );
            app.set_bootstrap_path_input(
                paths
                    .lumen_dir
                    .join("bootstrap.generated.json")
                    .display()
                    .to_string()
                    .into(),
            );
        }
        _ => {}
    }
}

fn build_setup_request(app: &AppWindow) -> Result<SetupRequest, String> {
    if app.get_selected_existing_action() == 2 {
        return Err(i18n::bi("Setup creation was cancelled", "已取消创建设置").to_string());
    }
    let platform = setup::current_platform_profile().map_err(|error| error.to_string())?;
    let preset = wizard::selected_preset(app);
    let backend = wizard::selected_backend(app)?;
    let region = if app.get_selected_region_index() == 1 {
        setup::REGION_CN
    } else {
        setup::REGION_OTHER
    };
    let cache_dir = non_empty_path(&app.get_cache_dir_input(), "cache directory")?;
    let config_path = non_empty_path(&app.get_config_path_input(), "config path")?;
    let bootstrap_path = non_empty_path(&app.get_bootstrap_path_input(), "bootstrap path")?;

    Ok(SetupRequest {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        region: region.to_owned(),
        preset,
        platform,
        backend,
        cache_dir,
        config_path,
        bootstrap_path,
        conflict_action: app.get_selected_existing_action(),
    })
}

fn non_empty_path(value: &SharedString, label: &str) -> Result<PathBuf, String> {
    let value = value.to_string();
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} is required"));
    }
    Ok(setup::expand_tilde(trimmed))
}

fn create_setup(request: SetupRequest) -> Result<SetupSummary, String> {
    if setup::is_dangerous_cache_dir(&request.cache_dir) {
        return Err(format!(
            "`{}` is not a safe model cache directory",
            request.cache_dir.display()
        ));
    }
    setup::ensure_cache_dir(&request.cache_dir).map_err(|error| error.to_string())?;
    if !request.cache_dir.is_dir() {
        return Err(format!(
            "`{}` is not a directory",
            request.cache_dir.display()
        ));
    }
    if !setup::is_writable_dir(&request.cache_dir) {
        return Err(format!("`{}` is not writable", request.cache_dir.display()));
    }

    let mut log_lines = Vec::new();
    if let Some(free_gb) = setup::free_disk_gb(&request.cache_dir)
        && free_gb < request.preset.min_disk_gb as f64
    {
        log_lines.push(format!(
            "warning: `{}` has {free_gb:.1} GB free; `{}` recommends at least {} GB",
            request.cache_dir.display(),
            request.preset.name,
            request.preset.min_disk_gb
        ));
    }

    if request.conflict_action == 0 {
        if request.config_path.exists() {
            log_lines.push(format!(
                "overwriting config: {}",
                request.config_path.display()
            ));
        }
        if request.bootstrap_path.exists() {
            log_lines.push(format!(
                "overwriting bootstrap: {}",
                request.bootstrap_path.display()
            ));
        }
    }

    let selection = setup::SetupSelection {
        version: request.version,
        region: request.region,
        preset: request.preset,
        platform: request.platform,
        backend: request.backend,
        cache_dir: request.cache_dir,
        config_path: request.config_path,
        bootstrap_path: request.bootstrap_path,
    };
    let written = setup::write_setup(&selection).map_err(|error| error.to_string())?;
    log_lines.push(format!(
        "created {} preset config: {}",
        written.bootstrap.preset,
        written.config_path.display()
    ));
    log_lines.push(format!(
        "created bootstrap: {}",
        written.bootstrap_path.display()
    ));
    log_lines.push(format!(
        "selected backend: {} ({})",
        written.bootstrap.backend, written.bootstrap.release_profile
    ));
    log_lines.push("next step: press Start".to_owned());

    Ok(SetupSummary {
        config_path: written.config_path,
        profile_text: i18n::profile_summary(
            &written.bootstrap.preset,
            &written.bootstrap.backend,
            &written.bootstrap.release_profile,
        )
        .to_string(),
        log_lines,
    })
}

fn start_options_from_ui(app: &AppWindow) -> StartOptions {
    let bootstrap_path = optional_path(&app.get_bootstrap_path_input());
    StartOptions {
        bootstrap_path,
        ..StartOptions::default()
    }
}

fn optional_path(value: &SharedString) -> Option<PathBuf> {
    let value = value.to_string();
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(setup::expand_tilde(trimmed))
    }
}

fn start_hub(
    tx: mpsc::Sender<UiMessage>,
    process_state: Arc<Mutex<ProcessState>>,
    options: StartOptions,
) {
    let mut observer = ShellObserver { tx: tx.clone() };
    let plan = match resolve_start_plan(options) {
        Ok(plan) => plan,
        Err(error) => {
            let _ = tx.send(UiMessage::SetupRequired(error.to_string()));
            clear_launching(&process_state, &tx);
            return;
        }
    };

    let _ = tx.send(UiMessage::ConfigPath(
        plan.config_path.display().to_string(),
    ));
    let _ = tx.send(UiMessage::Profile(profile_label(&plan).to_string()));

    let hub_path = match prepare_hub(&plan, &mut observer) {
        Ok(path) => path,
        Err(error) => {
            let _ = tx.send(UiMessage::Controls {
                can_start: true,
                can_stop: false,
            });
            let _ = tx.send(UiMessage::Error(error.to_string()));
            clear_launching(&process_state, &tx);
            return;
        }
    };

    let mut hub = match spawn_hub(&plan, &hub_path, HubStdio::Piped, &mut observer) {
        Ok(hub) => hub,
        Err(error) => {
            let _ = tx.send(UiMessage::Controls {
                can_start: true,
                can_stop: false,
            });
            let _ = tx.send(UiMessage::Error(error.to_string()));
            clear_launching(&process_state, &tx);
            return;
        }
    };

    if let Some(stdout) = hub.stdout() {
        spawn_log_reader(stdout, tx.clone(), "stdout");
    }
    if let Some(stderr) = hub.stderr() {
        spawn_log_reader(stderr, tx.clone(), "stderr");
    }

    {
        let mut state = match process_state.lock() {
            Ok(state) => state,
            Err(_) => {
                let _ = tx.send(UiMessage::Error("hub state lock was poisoned".to_owned()));
                return;
            }
        };
        state.launching = false;
        state.hub = Some(hub);
    }

    let _ = tx.send(UiMessage::Status(i18n::status_running().to_string()));
    let _ = tx.send(UiMessage::Controls {
        can_start: false,
        can_stop: true,
    });
    loop {
        thread::sleep(Duration::from_millis(500));
        let status = {
            let mut state = match process_state.lock() {
                Ok(state) => state,
                Err(_) => {
                    let _ = tx.send(UiMessage::Error("hub state lock was poisoned".to_owned()));
                    return;
                }
            };
            let Some(hub) = state.hub.as_mut() else {
                return;
            };
            match hub.try_wait() {
                Ok(Some(status)) => {
                    state.hub = None;
                    Some(Ok(status))
                }
                Ok(None) => None,
                Err(error) => {
                    state.hub = None;
                    Some(Err(LauncherError::SpawnHub {
                        path: hub_path.clone(),
                        source: error,
                    }))
                }
            }
        };

        match status {
            Some(Ok(status)) if status.success() => {
                let _ = tx.send(UiMessage::Status(i18n::status_stopped().to_string()));
                let _ = tx.send(UiMessage::Controls {
                    can_start: true,
                    can_stop: false,
                });
                let _ = tx.send(UiMessage::Log("lumen-hub exited".to_owned()));
                return;
            }
            Some(Ok(status)) => {
                let _ = tx.send(UiMessage::Status(
                    i18n::status_exited_with_error().to_string(),
                ));
                let _ = tx.send(UiMessage::Controls {
                    can_start: true,
                    can_stop: false,
                });
                let _ = tx.send(UiMessage::Error(format!(
                    "lumen-hub {}",
                    lumen_launcher::FormattedExitStatus(status)
                )));
                return;
            }
            Some(Err(error)) => {
                let _ = tx.send(UiMessage::Controls {
                    can_start: true,
                    can_stop: false,
                });
                let _ = tx.send(UiMessage::Error(error.to_string()));
                return;
            }
            None => {}
        }
    }
}

fn clear_launching(process_state: &Arc<Mutex<ProcessState>>, tx: &mpsc::Sender<UiMessage>) {
    match process_state.lock() {
        Ok(mut state) => {
            state.launching = false;
        }
        Err(_) => {
            let _ = tx.send(UiMessage::Error("hub state lock was poisoned".to_owned()));
        }
    }
}

fn spawn_log_reader<R>(reader: R, tx: mpsc::Sender<UiMessage>, label: &'static str)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            match reader.read_until(b'\n', &mut buffer) {
                Ok(0) => return,
                Ok(_) => {
                    trim_line_end(&mut buffer);
                    let line = decode_output_bytes(&buffer);
                    let _ = tx.send(UiMessage::Log(format!("[{label}] {line}")));
                }
                Err(error) => {
                    let _ = tx.send(UiMessage::Error(format!("failed to read {label}: {error}")));
                    return;
                }
            }
        }
    });
}

fn trim_line_end(bytes: &mut Vec<u8>) {
    while matches!(bytes.last(), Some(b'\n' | b'\r')) {
        bytes.pop();
    }
}

fn decode_output_bytes(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.to_owned();
    }
    decode_non_utf8_output(bytes)
}

#[cfg(target_os = "windows")]
fn decode_non_utf8_output(bytes: &[u8]) -> String {
    let acp = decode_windows_code_page(bytes, windows_sys::Win32::Globalization::CP_ACP);
    let gbk = decode_windows_code_page(bytes, 936);

    match (acp, gbk) {
        (Some(acp), Some(gbk)) if contains_cjk(&gbk) && !contains_cjk(&acp) => gbk,
        (Some(acp), _) => acp,
        (None, Some(gbk)) => gbk,
        (None, None) => String::from_utf8_lossy(bytes).into_owned(),
    }
}

#[cfg(not(target_os = "windows"))]
fn decode_non_utf8_output(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(target_os = "windows")]
fn decode_windows_code_page(bytes: &[u8], code_page: u32) -> Option<String> {
    use windows_sys::Win32::Globalization::{MB_ERR_INVALID_CHARS, MultiByteToWideChar};

    if bytes.is_empty() {
        return Some(String::new());
    }
    let byte_len = i32::try_from(bytes.len()).ok()?;
    unsafe {
        let wide_len = MultiByteToWideChar(
            code_page,
            MB_ERR_INVALID_CHARS,
            bytes.as_ptr(),
            byte_len,
            std::ptr::null_mut(),
            0,
        );
        if wide_len == 0 {
            return None;
        }
        let mut wide = vec![0_u16; wide_len as usize];
        let written = MultiByteToWideChar(
            code_page,
            MB_ERR_INVALID_CHARS,
            bytes.as_ptr(),
            byte_len,
            wide.as_mut_ptr(),
            wide_len,
        );
        if written == 0 {
            return None;
        }
        wide.truncate(written as usize);
        String::from_utf16(&wide).ok()
    }
}

#[cfg(target_os = "windows")]
fn contains_cjk(text: &str) -> bool {
    text.chars().any(|ch| {
        ('\u{3400}'..='\u{4dbf}').contains(&ch) || ('\u{4e00}'..='\u{9fff}').contains(&ch)
    })
}

#[cfg(target_os = "windows")]
fn hide_console_window(command: &mut Command) {
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    command.creation_flags(CREATE_NO_WINDOW);
}

fn apply_message(app: &AppWindow, message: UiMessage) {
    match message {
        UiMessage::Status(status) => app.set_status_text(status.into()),
        UiMessage::ConfigPath(path) => app.set_config_path(path.into()),
        UiMessage::Profile(profile) => app.set_profile_text(profile.into()),
        UiMessage::Controls {
            can_start,
            can_stop,
        } => {
            app.set_can_start(can_start);
            app.set_can_stop(can_stop);
        }
        UiMessage::SetupRequired(reason) => {
            set_setup_required(app, Some(reason.clone()));
            append_log(app, &format!("setup required: {reason}"));
        }
        UiMessage::SetupReady(summary) => {
            app.set_status_text(i18n::status_ready());
            app.set_config_path(summary.config_path.display().to_string().into());
            app.set_profile_text(summary.profile_text.into());
            app.set_setup_visible(false);
            app.set_can_start(true);
            app.set_can_stop(false);
            for line in summary.log_lines {
                append_log(app, &line);
            }
        }
        UiMessage::Log(line) => append_log(app, &line),
        UiMessage::Error(error) => {
            app.set_status_text(i18n::status_error());
            app.set_can_stop(false);
            append_log(app, &format!("error: {error}"));
        }
    }
}

fn append_log(app: &AppWindow, line: &str) {
    let existing = app.get_log_text().to_string();
    let mut lines = existing.lines().map(str::to_owned).collect::<Vec<_>>();
    lines.push(line.to_owned());
    if lines.len() > MAX_LOG_LINES {
        let keep_from = lines.len() - MAX_LOG_LINES;
        lines.drain(0..keep_from);
    }
    app.set_log_text(SharedString::from(format!("{}\n", lines.join("\n"))));
}

fn apply_picker_result(app: &AppWindow, target: PickerTarget) {
    let result = match target {
        PickerTarget::CacheDir => choose_folder(i18n::picker_cache_dir()),
        PickerTarget::ConfigPath => choose_save_file(i18n::picker_config_path(), "config.yaml"),
        PickerTarget::BootstrapPath => {
            choose_save_file(i18n::picker_bootstrap_path(), "bootstrap.json")
        }
    };
    match result {
        Ok(Some(path)) => {
            let value = path.display().to_string();
            match target {
                PickerTarget::CacheDir => app.set_cache_dir_input(value.into()),
                PickerTarget::ConfigPath => app.set_config_path_input(value.into()),
                PickerTarget::BootstrapPath => app.set_bootstrap_path_input(value.into()),
            }
            update_setup_warning(app);
            wizard::sync_wizard(app);
        }
        Ok(None) => {}
        Err(error) => append_log(app, &format!("picker error: {error}")),
    }
}

fn update_setup_warning(app: &AppWindow) {
    if app.get_setup_visible() {
        wizard::update_setup_warning(app);
    }
}

fn choose_folder(prompt: &str) -> Result<Option<PathBuf>, String> {
    #[cfg(target_os = "macos")]
    {
        let script = format!(
            r#"POSIX path of (choose folder with prompt "{}")"#,
            escape_applescript(prompt)
        );
        return run_osascript(&script);
    }

    #[cfg(target_os = "windows")]
    {
        let script = format!(
            r#"
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ $dialog.SelectedPath }}
"#,
            prompt.replace('\'', "''")
        );
        return run_powershell_dialog(&script);
    }

    #[cfg(target_os = "linux")]
    {
        return run_zenity(&["--file-selection", "--directory", "--title", prompt]);
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        let _ = prompt;
        Err("native folder picker is not available on this platform".to_owned())
    }
}

fn choose_save_file(prompt: &str, default_name: &str) -> Result<Option<PathBuf>, String> {
    #[cfg(target_os = "macos")]
    {
        let script = format!(
            r#"POSIX path of (choose file name with prompt "{}" default name "{}")"#,
            escape_applescript(prompt),
            escape_applescript(default_name)
        );
        return run_osascript(&script);
    }

    #[cfg(target_os = "windows")]
    {
        let script = format!(
            r#"
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.Title = '{}'
$dialog.FileName = '{}'
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ $dialog.FileName }}
"#,
            prompt.replace('\'', "''"),
            default_name.replace('\'', "''")
        );
        return run_powershell_dialog(&script);
    }

    #[cfg(target_os = "linux")]
    {
        return run_zenity(&[
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title",
            prompt,
            "--filename",
            default_name,
        ]);
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        let _ = (prompt, default_name);
        Err("native save picker is not available on this platform".to_owned())
    }
}

#[cfg(target_os = "macos")]
fn run_osascript(script: &str) -> Result<Option<PathBuf>, String> {
    let output = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .map_err(|error| format!("failed to run osascript: {error}"))?;
    picker_output(output)
}

#[cfg(target_os = "macos")]
fn escape_applescript(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(target_os = "windows")]
fn run_powershell_dialog(script: &str) -> Result<Option<PathBuf>, String> {
    let script =
        format!("[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); {script}");
    let mut command = Command::new("powershell");
    hide_console_window(&mut command);
    let output = command
        .args(["-NoProfile", "-STA", "-Command", &script])
        .output()
        .map_err(|error| format!("failed to run PowerShell picker: {error}"))?;
    picker_output(output)
}

#[cfg(target_os = "linux")]
fn run_zenity(args: &[&str]) -> Result<Option<PathBuf>, String> {
    let output = Command::new("zenity")
        .args(args)
        .output()
        .map_err(|error| format!("failed to run zenity: {error}"))?;
    picker_output(output)
}

fn picker_output(output: std::process::Output) -> Result<Option<PathBuf>, String> {
    if output.status.success() {
        let stdout = decode_output_bytes(&output.stdout).trim().to_owned();
        if stdout.is_empty() {
            Ok(None)
        } else {
            Ok(Some(PathBuf::from(stdout)))
        }
    } else {
        let stderr = decode_output_bytes(&output.stderr);
        if stderr.trim().is_empty() || stderr.contains("User canceled") || stderr.contains("-128") {
            Ok(None)
        } else {
            Err(stderr.trim().to_owned())
        }
    }
}

fn default_config_label() -> String {
    match setup::default_setup_paths() {
        Ok(paths) => paths.config_path.display().to_string(),
        Err(_) => "~/.lumen/config.yaml".to_owned(),
    }
}

fn default_bootstrap_label() -> String {
    match setup::default_setup_paths() {
        Ok(paths) => paths.bootstrap_path.display().to_string(),
        Err(_) => "~/.lumen/bootstrap.json".to_owned(),
    }
}

#[derive(Debug)]
struct SetupRequest {
    version: String,
    region: String,
    preset: setup::Preset,
    platform: setup::PlatformProfile,
    backend: setup::Backend,
    cache_dir: PathBuf,
    config_path: PathBuf,
    bootstrap_path: PathBuf,
    conflict_action: i32,
}

#[derive(Debug)]
struct SetupSummary {
    config_path: PathBuf,
    profile_text: String,
    log_lines: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum PickerTarget {
    CacheDir,
    ConfigPath,
    BootstrapPath,
}

#[derive(Debug)]
enum UiMessage {
    Status(String),
    ConfigPath(String),
    Profile(String),
    Controls { can_start: bool, can_stop: bool },
    SetupRequired(String),
    SetupReady(SetupSummary),
    Log(String),
    Error(String),
}

struct ShellObserver {
    tx: mpsc::Sender<UiMessage>,
}

#[derive(Default)]
struct ProcessState {
    hub: Option<RunningHub>,
    launching: bool,
}

impl LaunchObserver for ShellObserver {
    fn manifest_fetch_started(&mut self, _url: &str) {
        let _ = self.tx.send(UiMessage::Status(
            i18n::status_fetching_manifest().to_string(),
        ));
    }

    fn manifest_fetched(&mut self, version: &str) {
        let _ = self
            .tx
            .send(UiMessage::Log(format!("release manifest {version}")));
    }

    fn hub_already_installed(&mut self, hub_path: &Path) {
        let _ = self.tx.send(UiMessage::Log(format!(
            "lumen-hub already installed: {}",
            hub_path.display()
        )));
    }

    fn download_started(&mut self, file_name: &str, total: Option<u64>) {
        let detail = total
            .map(format_bytes)
            .map(|size| format!(" ({size})"))
            .unwrap_or_default();
        let _ = self.tx.send(UiMessage::Status(
            i18n::status_downloading_hub().to_string(),
        ));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("downloading {file_name}{detail}")));
    }

    fn download_progress(
        &mut self,
        file_name: &str,
        _delta: u64,
        written: u64,
        total: Option<u64>,
    ) {
        let status = if let Some(total) = total {
            format!(
                "Downloading {file_name}: {} / {}",
                format_bytes(written),
                format_bytes(total)
            )
        } else {
            format!("Downloading {file_name}: {}", format_bytes(written))
        };
        let _ = self.tx.send(UiMessage::Status(status));
    }

    fn download_finished(&mut self, file_name: &str, written: u64) {
        let _ = self.tx.send(UiMessage::Log(format!(
            "downloaded {file_name} ({})",
            format_bytes(written)
        )));
    }

    fn verify_started(&mut self, path: &Path) {
        let _ = self
            .tx
            .send(UiMessage::Status(i18n::status_verifying_hub().to_string()));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("verifying {}", path.display())));
    }

    fn verify_finished(&mut self, _path: &Path) {
        let _ = self.tx.send(UiMessage::Log("checksum ok".to_owned()));
    }

    fn extract_started(&mut self, path: &Path) {
        let _ = self
            .tx
            .send(UiMessage::Status(i18n::status_extracting_hub().to_string()));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("extracting {}", path.display())));
    }

    fn hub_installed(&mut self, hub_path: &Path) {
        let _ = self.tx.send(UiMessage::Log(format!(
            "lumen-hub ready: {}",
            hub_path.display()
        )));
    }

    fn hub_starting(&mut self, hub_path: &Path) {
        let _ = self
            .tx
            .send(UiMessage::Status(i18n::status_starting_hub().to_string()));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("starting {}", hub_path.display())));
    }
}
