#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    borrow::Cow,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

use gpui::{prelude::*, *};use gpui_component::{
    ActiveTheme, Disableable as _, Root, Sizable as _, StyledExt as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    input::{Input, InputEvent, InputState},
    radio::{Radio, RadioGroup},
    scroll::ScrollableElement as _,
    v_flex, Icon, IconName, Theme, ThemeMode,
};
use lumen_launcher::{
    HubStdio, LaunchObserver, LauncherError, RunningHub, StartOptions, format_bytes, prepare_hub,
    resolve_start_plan, setup, spawn_hub,
};

mod i18n;

const MAX_LOG_LINES: usize = 600;

/// Serves the bundled gpui-component icon SVGs so that `Icon`/`svg()` elements
/// resolve instead of rendering blank. Only the icons used by this shell are
/// embedded.
struct IconAssets;

impl AssetSource for IconAssets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        let bytes: &'static [u8] = match path {
            "icons/check.svg" => include_bytes!("../assets/icons/check.svg"),
            "icons/arrow-right.svg" => include_bytes!("../assets/icons/arrow-right.svg"),
            "icons/chevron-down.svg" => include_bytes!("../assets/icons/chevron-down.svg"),
            "icons/chevron-right.svg" => include_bytes!("../assets/icons/chevron-right.svg"),
            "icons/info.svg" => include_bytes!("../assets/icons/info.svg"),
            "icons/folder-open.svg" => include_bytes!("../assets/icons/folder-open.svg"),
            "icons/settings-2.svg" => include_bytes!("../assets/icons/settings-2.svg"),
            "icons/square-terminal.svg" => include_bytes!("../assets/icons/square-terminal.svg"),
            "icons/circle-x.svg" => include_bytes!("../assets/icons/circle-x.svg"),
            "icons/triangle-alert.svg" => include_bytes!("../assets/icons/triangle-alert.svg"),
            "icons/close.svg" => include_bytes!("../assets/icons/close.svg"),
            "icons/sun.svg" => include_bytes!("../assets/icons/sun.svg"),
            "icons/moon.svg" => include_bytes!("../assets/icons/moon.svg"),
            _ => return Ok(None),
        };
        Ok(Some(Cow::Borrowed(bytes)))
    }

    fn list(&self, _path: &str) -> gpui::Result<Vec<SharedString>> {
        Ok(Vec::new())
    }
}

fn main() {
    Application::new()
        .with_assets(IconAssets)
        .run(|cx| {
        gpui_component::init(cx);
        cx.activate(true);

        let process_state = Arc::new(Mutex::new(ProcessState::default()));
        let (tx, rx) = mpsc::channel::<UiMessage>();
        let window_bounds = Bounds::centered(None, size(px(1000.), px(680.)), cx);

        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(window_bounds)),
                    window_min_size: Some(size(px(720.), px(480.))),
                    ..Default::default()
                },
                move |window, cx| {
                    let view = cx.new(|cx| {
                        ShellApp::new(rx, tx.clone(), Arc::clone(&process_state), window, cx)
                    });
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("failed to open Lumen Hub window");

        window
            .update(cx, |_, window, _| {
                window.activate_window();
                window.set_window_title("Lumen Hub");
            })
            .ok();
    });
}

struct ShellApp {
    tx: mpsc::Sender<UiMessage>,
    process_state: Arc<Mutex<ProcessState>>,
    status_text: String,
    status_kind: StatusKind,
    config_path: String,
    profile_text: String,
    detected_text: String,
    setup_warning: String,
    setup_review_text: String,
    setup_visible: bool,
    can_start: bool,
    can_stop: bool,
    existing_visible: bool,
    setup_wizard_cursor: usize,
    selected_region_index: usize,
    selected_preset_index: usize,
    selected_backend_index: usize,
    selected_existing_action: usize,
    advanced_settings_open: bool,
    backend_choices: Vec<BackendChoiceView>,
    log_lines: Vec<String>,
    cache_input: Entity<InputState>,
    config_input: Entity<InputState>,
    bootstrap_input: Entity<InputState>,
    _subscriptions: Vec<Subscription>,
    _message_pump: Task<()>,
}

impl ShellApp {
    fn new(
        rx: mpsc::Receiver<UiMessage>,
        tx: mpsc::Sender<UiMessage>,
        process_state: Arc<Mutex<ProcessState>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let cache_input = cx.new(|cx| InputState::new(window, cx).placeholder(i18n::cache_label()));
        let config_input =
            cx.new(|cx| InputState::new(window, cx).placeholder(i18n::config_field_label()));
        let bootstrap_input =
            cx.new(|cx| InputState::new(window, cx).placeholder(i18n::bootstrap_label()));

        let _subscriptions = vec![
            cx.subscribe(&cache_input, |this, _, event, cx| {
                if matches!(event, InputEvent::Change) {
                    this.sync_wizard(cx);
                    cx.notify();
                }
            }),
            cx.subscribe(&config_input, |this, _, event, cx| {
                if matches!(event, InputEvent::Change) {
                    this.sync_wizard(cx);
                    cx.notify();
                }
            }),
            cx.subscribe(&bootstrap_input, |this, _, event, cx| {
                if matches!(event, InputEvent::Change) {
                    this.sync_wizard(cx);
                    cx.notify();
                }
            }),
        ];

        let _message_pump = cx.spawn(async move |this, cx| {
            loop {
                while let Ok(message) = rx.try_recv() {
                    if this
                        .update(cx, |app, cx| {
                            app.apply_message(message);
                            cx.notify();
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                cx.background_executor()
                    .timer(Duration::from_millis(100))
                    .await;
            }
        });

        let mut app = Self {
            tx,
            process_state,
            status_text: i18n::status_checking_setup().to_string(),
            status_kind: StatusKind::Checking,
            config_path: default_config_label(),
            profile_text: i18n::bootstrap_missing(&default_bootstrap_label()).to_string(),
            detected_text: String::new(),
            setup_warning: String::new(),
            setup_review_text: String::new(),
            setup_visible: false,
            can_start: false,
            can_stop: false,
            existing_visible: false,
            setup_wizard_cursor: 0,
            selected_region_index: 0,
            selected_preset_index: 1,
            selected_backend_index: 0,
            selected_existing_action: 0,
            advanced_settings_open: false,
            backend_choices: Vec::new(),
            log_lines: vec![i18n::log_ready().to_string()],
            cache_input,
            config_input,
            bootstrap_input,
            _subscriptions,
            _message_pump,
        };

        app.configure_setup_defaults(window, cx);
        app.refresh_setup_state(cx);
        app.sync_wizard(cx);
        app
    }

    fn set_status(&mut self, text: impl Into<String>, kind: StatusKind) {
        self.status_text = text.into();
        self.status_kind = kind;
    }

    fn configure_setup_defaults(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Ok(paths) = setup::default_setup_paths() {
            self.set_input_value(
                InputTarget::Cache,
                paths.cache_dir.display().to_string(),
                window,
                cx,
            );
            self.set_input_value(
                InputTarget::Config,
                paths.config_path.display().to_string(),
                window,
                cx,
            );
            self.set_input_value(
                InputTarget::Bootstrap,
                paths.bootstrap_path.display().to_string(),
                window,
                cx,
            );
            self.existing_visible = paths.config_path.exists() || paths.bootstrap_path.exists();
        }

        let system = setup::detect_system();
        let memory = setup::detect_memory();
        let ram = match memory.total_gb {
            Some(total_gb) => i18n::ram_known(total_gb),
            None => i18n::ram_unknown().to_owned(),
        };
        self.detected_text =
            i18n::detected_system(&system.os_label(), &system.arch, &ram).to_string();

        match setup::platform_profile(&system) {
            Ok(platform) => self.configure_backend_choices(platform),
            Err(error) => {
                self.backend_choices.clear();
                self.setup_warning = error.to_string();
            }
        }
    }

    fn configure_backend_choices(&mut self, platform: setup::PlatformProfile) {
        let choices = setup::backend_choices(platform);
        let mut first_enabled = None;
        self.backend_choices = choices
            .iter()
            .enumerate()
            .map(|(index, choice)| {
                let (title, detail, enabled) = backend_choice_display(choice);
                if enabled && first_enabled.is_none() {
                    first_enabled = Some(index);
                }
                BackendChoiceView {
                    title,
                    detail,
                    enabled,
                }
            })
            .collect();

        let selected_enabled = self
            .backend_choices
            .get(self.selected_backend_index)
            .map(|choice| choice.enabled)
            .unwrap_or(false);
        if !selected_enabled {
            if let Some(index) = first_enabled {
                self.selected_backend_index = index;
            }
        }
    }

    fn refresh_setup_state(&mut self, cx: &mut Context<Self>) {
        match resolve_start_plan(StartOptions::default()) {
            Ok(plan) => {
                self.set_status(i18n::status_ready(), StatusKind::Ready);
                self.config_path = plan.config_path.display().to_string();
                self.profile_text = profile_label(&plan);
                self.setup_visible = false;
                self.can_start = true;
                self.can_stop = false;
            }
            Err(error) => self.set_setup_required(Some(error.to_string()), cx),
        }
    }

    fn set_setup_required(&mut self, reason: Option<String>, cx: &mut Context<Self>) {
        self.set_status(i18n::status_setup_required(), StatusKind::Setup);
        self.config_path = default_config_label();
        self.profile_text = i18n::bootstrap_missing(&default_bootstrap_label()).to_string();
        self.setup_visible = true;
        self.can_start = false;
        self.can_stop = false;
        self.setup_wizard_cursor = 0;
        if let Some(reason) = reason {
            self.setup_warning = reason;
        }
        self.sync_wizard(cx);
    }

    fn set_input_value(
        &mut self,
        target: InputTarget,
        value: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let input = self.input_for(target).clone();
        input.update(cx, |state, cx| {
            state.set_value(&value, window, cx);
        });
    }

    fn input_for(&self, target: InputTarget) -> &Entity<InputState> {
        match target {
            InputTarget::Cache => &self.cache_input,
            InputTarget::Config => &self.config_input,
            InputTarget::Bootstrap => &self.bootstrap_input,
        }
    }

    fn input_value(&self, target: InputTarget, cx: &App) -> String {
        self.input_for(target).read(cx).value().to_string()
    }

    fn step_sequence(&self) -> Vec<SetupStep> {
        let mut steps = vec![SetupStep::Welcome];
        if self.existing_visible {
            steps.push(SetupStep::Conflict);
        }
        steps.extend([
            SetupStep::Region,
            SetupStep::Preset,
            SetupStep::Backend,
            SetupStep::Paths,
            SetupStep::Review,
        ]);
        steps
    }

    fn current_step(&self) -> SetupStep {
        let sequence = self.step_sequence();
        let cursor = self
            .setup_wizard_cursor
            .min(sequence.len().saturating_sub(1));
        sequence[cursor]
    }

    fn sync_wizard(&mut self, cx: &mut Context<Self>) {
        let sequence = self.step_sequence();
        let total = sequence.len().max(1);
        self.setup_wizard_cursor = self.setup_wizard_cursor.min(total - 1);
        self.update_setup_warning(cx);
        if self.current_step() == SetupStep::Review {
            self.setup_review_text = self.build_review_text(cx);
        }
    }

    fn go_next(&mut self, cx: &mut Context<Self>) {
        let step = self.current_step();
        if !self.can_advance(step, cx) {
            let reason = self.advance_block_reason(step, cx);
            self.append_log(&reason);
            return;
        }

        let total = self.step_sequence().len();
        self.setup_wizard_cursor = (self.setup_wizard_cursor + 1).min(total.saturating_sub(1));
        self.sync_wizard(cx);
    }

    fn go_back(&mut self, cx: &mut Context<Self>) {
        self.setup_wizard_cursor = self.setup_wizard_cursor.saturating_sub(1);
        self.sync_wizard(cx);
    }

    fn can_advance(&self, step: SetupStep, cx: &App) -> bool {
        match step {
            SetupStep::Conflict => self.selected_existing_action != 2,
            SetupStep::Backend => self.selected_backend().is_ok(),
            SetupStep::Paths => self.paths_valid(cx),
            SetupStep::Review => {
                self.selected_existing_action != 2
                    && self.selected_backend().is_ok()
                    && self.paths_valid(cx)
            }
            _ => true,
        }
    }

    fn advance_block_reason(&self, step: SetupStep, cx: &App) -> String {
        match step {
            SetupStep::Conflict if self.selected_existing_action == 2 => {
                i18n::block_setup_cancelled().to_string()
            }
            SetupStep::Backend => self
                .selected_backend()
                .err()
                .unwrap_or_else(i18n::warning_backend_unavailable),
            SetupStep::Paths | SetupStep::Review if !self.paths_valid(cx) => {
                i18n::block_paths_required().to_string()
            }
            _ => i18n::block_complete_step().to_string(),
        }
    }

    fn paths_valid(&self, cx: &App) -> bool {
        !self.input_value(InputTarget::Cache, cx).trim().is_empty()
            && !self.input_value(InputTarget::Config, cx).trim().is_empty()
            && !self
                .input_value(InputTarget::Bootstrap, cx)
                .trim()
                .is_empty()
    }

    fn build_review_text(&self, cx: &App) -> String {
        let preset = self.selected_preset();
        let region = if self.selected_region_index == 1 {
            i18n::review_region_cn()
        } else {
            i18n::review_region_other()
        };
        let backend = self
            .selected_backend()
            .map(|backend| i18n::review_backend(backend.name, backend.release_profile))
            .unwrap_or_else(|_| i18n::backend_unavailable_label());

        let cache_label = i18n::cache_label();
        let mut lines = vec![
            region.to_string(),
            i18n::review_preset(preset.display_title()).to_string(),
            backend.to_string(),
            i18n::review_path(cache_label, &self.input_value(InputTarget::Cache, cx)).to_string(),
        ];

        if self.advanced_settings_open {
            let config_label = i18n::config_field_label();
            let bootstrap_label = i18n::bootstrap_label();
            lines.push(
                i18n::review_path(config_label, &self.input_value(InputTarget::Config, cx))
                    .to_string(),
            );
            lines.push(
                i18n::review_path(
                    bootstrap_label,
                    &self.input_value(InputTarget::Bootstrap, cx),
                )
                .to_string(),
            );
        }

        lines.join("\n")
    }

    fn selected_preset(&self) -> setup::Preset {
        let presets = setup::Preset::all();
        let index = self.selected_preset_index.min(presets.len() - 1);
        presets[index]
    }

    fn selected_backend(&self) -> Result<setup::Backend, String> {
        let platform = setup::current_platform_profile().map_err(|error| error.to_string())?;
        let choices = setup::backend_choices(platform);
        let index = self.selected_backend_index;
        choices
            .get(index)
            .ok_or_else(|| "selected backend index is out of range".to_owned())?
            .backend
            .ok_or_else(|| {
                choices[index]
                    .disabled_reason
                    .clone()
                    .unwrap_or_else(i18n::warning_backend_unavailable)
            })
    }

    fn update_setup_warning(&mut self, cx: &App) {
        let mut warnings = Vec::new();
        let preset = self.selected_preset();
        let memory = setup::detect_memory();
        if let Some(total_gb) = memory.total_gb
            && total_gb < preset.min_ram_gb as f64
        {
            warnings.push(i18n::warning_ram(
                preset.display_title(),
                preset.min_ram_gb,
                total_gb,
            ));
        }

        if let Ok(platform) = setup::current_platform_profile() {
            let choices = setup::backend_choices(platform);
            if let Some(choice) = choices.get(self.selected_backend_index)
                && choice.backend.is_none()
            {
                warnings.push(
                    choice
                        .disabled_reason
                        .clone()
                        .unwrap_or_else(i18n::warning_backend_unavailable),
                );
            }
        }

        let cache_dir = setup::expand_tilde(self.input_value(InputTarget::Cache, cx).trim());
        if cache_dir.exists()
            && let Some(free_gb) = setup::free_disk_gb(&cache_dir)
            && free_gb < preset.min_disk_gb as f64
        {
            warnings.push(i18n::warning_disk(
                &cache_dir.display().to_string(),
                preset.display_title(),
                free_gb,
                preset.min_disk_gb,
            ));
        }

        self.setup_warning = warnings.join("\n");
        if self.current_step() == SetupStep::Review {
            self.setup_review_text = self.build_review_text(cx);
        }
    }

    fn apply_conflict_action(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Ok(paths) = setup::default_setup_paths() else {
            return;
        };
        match self.selected_existing_action {
            0 => {
                self.set_input_value(
                    InputTarget::Config,
                    paths.config_path.display().to_string(),
                    window,
                    cx,
                );
                self.set_input_value(
                    InputTarget::Bootstrap,
                    paths.bootstrap_path.display().to_string(),
                    window,
                    cx,
                );
            }
            1 => {
                self.set_input_value(
                    InputTarget::Config,
                    paths
                        .lumen_dir
                        .join("config.generated.yaml")
                        .display()
                        .to_string(),
                    window,
                    cx,
                );
                self.set_input_value(
                    InputTarget::Bootstrap,
                    paths
                        .lumen_dir
                        .join("bootstrap.generated.json")
                        .display()
                        .to_string(),
                    window,
                    cx,
                );
            }
            _ => {}
        }
        self.sync_wizard(cx);
    }

    fn build_setup_request(&self, cx: &App) -> Result<SetupRequest, String> {
        if self.selected_existing_action == 2 {
            return Err(i18n::block_setup_creation_cancelled().to_string());
        }
        let platform = setup::current_platform_profile().map_err(|error| error.to_string())?;
        let preset = self.selected_preset();
        let backend = self.selected_backend()?;
        let region = if self.selected_region_index == 1 {
            setup::REGION_CN
        } else {
            setup::REGION_OTHER
        };
        let cache_dir =
            non_empty_path(&self.input_value(InputTarget::Cache, cx), "cache directory")?;
        let config_path =
            non_empty_path(&self.input_value(InputTarget::Config, cx), "config path")?;
        let bootstrap_path = non_empty_path(
            &self.input_value(InputTarget::Bootstrap, cx),
            "bootstrap path",
        )?;

        Ok(SetupRequest {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            region: region.to_owned(),
            preset,
            platform,
            backend,
            cache_dir,
            config_path,
            bootstrap_path,
            conflict_action: self.selected_existing_action,
        })
    }

    fn start_options(&self, cx: &App) -> StartOptions {
        StartOptions {
            bootstrap_path: optional_path(&self.input_value(InputTarget::Bootstrap, cx)),
            ..StartOptions::default()
        }
    }

    fn start_requested(&mut self, cx: &mut Context<Self>) {
        let options = self.start_options(cx);
        let gate = match self.process_state.lock() {
            Ok(state) if state.launching || state.hub.is_some() => StartGate::Busy,
            Ok(mut state) => {
                state.launching = true;
                StartGate::Start
            }
            Err(_) => StartGate::Poisoned,
        };

        match gate {
            StartGate::Start => {}
            StartGate::Busy => {
                self.append_log("lumen-hub is already starting or running");
                return;
            }
            StartGate::Poisoned => {
                self.append_log("error: hub state lock was poisoned");
                return;
            }
        }

        self.can_start = false;
        self.can_stop = false;
        self.set_status(i18n::status_preparing(), StatusKind::Preparing);
        let tx = self.tx.clone();
        let process_state = Arc::clone(&self.process_state);
        thread::spawn(move || start_hub(tx, process_state, options));
    }

    fn stop_requested(&mut self) {
        let outcome = match self.process_state.lock() {
            Ok(mut state) => {
                if let Some(hub) = state.hub.as_mut() {
                    match hub.kill() {
                        Ok(()) => StopOutcome::Stopped,
                        Err(error) => StopOutcome::Error(error.to_string()),
                    }
                } else if state.launching {
                    StopOutcome::Preparing
                } else {
                    StopOutcome::NotRunning
                }
            }
            Err(_) => StopOutcome::Poisoned,
        };

        match outcome {
            StopOutcome::Stopped => {
                self.set_status(i18n::status_stopping(), StatusKind::Stopping);
                self.append_log("sent stop signal to lumen-hub");
            }
            StopOutcome::Preparing => {
                self.append_log("lumen-hub is still preparing; stop after it starts");
            }
            StopOutcome::NotRunning => self.append_log("lumen-hub is not running"),
            StopOutcome::Error(error) => {
                self.append_log(&format!("error: failed to stop lumen-hub: {error}"));
            }
            StopOutcome::Poisoned => self.append_log("error: hub state lock was poisoned"),
        }
    }

    fn create_setup_requested(&mut self, cx: &mut Context<Self>) {
        match self.build_setup_request(cx) {
            Ok(request) => {
                self.set_status(i18n::status_creating_setup(), StatusKind::Creating);
                self.can_start = false;
                self.can_stop = false;
                let tx = self.tx.clone();
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
                self.append_log(&format!("setup required: {error}"));
                self.set_setup_required(Some(error), cx);
            }
        }
    }

    fn browse_path(&mut self, target: PickerTarget, window: &mut Window, cx: &mut Context<Self>) {
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
                let input = match target {
                    PickerTarget::CacheDir => InputTarget::Cache,
                    PickerTarget::ConfigPath => InputTarget::Config,
                    PickerTarget::BootstrapPath => InputTarget::Bootstrap,
                };
                self.set_input_value(input, value, window, cx);
                self.sync_wizard(cx);
            }
            Ok(None) => {}
            Err(error) => self.append_log(&format!("picker error: {error}")),
        }
    }

    fn apply_message(&mut self, message: UiMessage) {
        match message {
            UiMessage::Status { text, kind } => self.set_status(text, kind),
            UiMessage::ConfigPath(path) => self.config_path = path,
            UiMessage::Profile(profile) => self.profile_text = profile,
            UiMessage::Controls {
                can_start,
                can_stop,
            } => {
                self.can_start = can_start;
                self.can_stop = can_stop;
            }
            UiMessage::SetupRequired(reason) => {
                self.set_status(i18n::status_setup_required(), StatusKind::Setup);
                self.setup_visible = true;
                self.can_start = false;
                self.can_stop = false;
                self.setup_wizard_cursor = 0;
                self.setup_warning = reason.clone();
                self.append_log(&format!("setup required: {reason}"));
            }
            UiMessage::SetupReady(summary) => {
                self.set_status(i18n::status_ready(), StatusKind::Ready);
                self.config_path = summary.config_path.display().to_string();
                self.profile_text = summary.profile_text;
                self.setup_visible = false;
                self.can_start = true;
                self.can_stop = false;
                for line in summary.log_lines {
                    self.append_log(&line);
                }
            }
            UiMessage::Log(line) => self.append_log(&line),
            UiMessage::Error(error) => {
                self.set_status(i18n::status_error(), StatusKind::Error);
                self.can_stop = false;
                self.append_log(&format!("error: {error}"));
            }
        }
    }

    fn append_log(&mut self, line: &str) {
        self.log_lines.push(line.to_owned());
        if self.log_lines.len() > MAX_LOG_LINES {
            let keep_from = self.log_lines.len() - MAX_LOG_LINES;
            self.log_lines.drain(0..keep_from);
        }
    }

    /// Rebuild strings derived from the active locale after a language switch.
    fn refresh_localized(&mut self, cx: &mut Context<Self>) {
        let system = setup::detect_system();
        let memory = setup::detect_memory();
        let ram = match memory.total_gb {
            Some(total_gb) => i18n::ram_known(total_gb),
            None => i18n::ram_unknown().to_owned(),
        };
        self.detected_text =
            i18n::detected_system(system.os_label(), &system.arch, &ram).to_string();
        self.refresh_profile_text(cx);
        self.sync_wizard(cx);
    }

    fn refresh_profile_text(&mut self, _cx: &App) {
        if self.setup_visible {
            return;
        }
        if let Ok(plan) = resolve_start_plan(StartOptions::default()) {
            self.config_path = plan.config_path.display().to_string();
            self.profile_text = profile_label(&plan);
        }
    }

    fn step_copy(step: SetupStep) -> (SharedString, SharedString) {
        match step {
            SetupStep::Welcome => (i18n::welcome_title(), i18n::welcome_detail()),
            SetupStep::Conflict => (i18n::conflict_step_title(), i18n::conflict_step_detail()),
            SetupStep::Region => (i18n::region_step_title(), i18n::region_step_detail()),
            SetupStep::Preset => (i18n::preset_step_title(), i18n::preset_step_detail()),
            SetupStep::Backend => (i18n::backend_step_title(), i18n::backend_step_detail()),
            SetupStep::Paths => (i18n::paths_step_title(), i18n::paths_step_detail()),
            SetupStep::Review => (i18n::review_step_title(), i18n::review_step_detail()),
        }
    }
}

impl Render for ShellApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (background, foreground, font_family) = {
            let theme = cx.theme();
            (
                theme.background,
                theme.foreground,
                theme.font_family.clone(),
            )
        };
        let content = if self.setup_visible {
            self.render_setup(window, cx)
        } else {
            self.render_dashboard(window, cx)
        };

        div()
            .id("lumen-shell")
            .size_full()
            .bg(background)
            .text_color(foreground)
            .font_family(font_family)
            .text_sm()
            .child(content)
            .children(Root::render_sheet_layer(window, cx))
            .children(Root::render_dialog_layer(window, cx))
            .children(Root::render_notification_layer(window, cx))
    }
}

impl ShellApp {
    fn render_setup(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let sequence = self.step_sequence();
        let total = sequence.len().max(1);
        let step = self.current_step();
        let (title, detail) = Self::step_copy(step);
        let can_back = self.setup_wizard_cursor > 0;
        let can_next = self.can_advance(step, cx);
        let is_review = step == SetupStep::Review;

        let sidebar = v_flex()
            .w(px(264.))
            .h_full()
            .flex_shrink_0()
            .gap_6()
            .p_5()
            .bg(cx.theme().sidebar)
            .border_r_1()
            .border_color(cx.theme().sidebar_border)
            .text_color(cx.theme().sidebar_foreground)
            .child(
                v_flex()
                    .gap_1()
                    .child(div().text_lg().font_bold().child("Lumen Hub"))
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(i18n::app_subtitle()),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .mt_2()
                            .child(self.render_locale_toggle(cx))
                            .child(self.render_theme_toggle(cx)),
                    ),
            )
            .child(
                v_flex()
                    .gap_1()
                    .children(
                        sequence
                            .iter()
                            .enumerate()
                            .map(|(index, item)| self.render_step_item(*item, index, cx)),
                    ),
            )
            .child(div().flex_1())
            .child(self.render_system_card(cx));

        h_flex()
            .id("setup")
            .size_full()
            .child(sidebar)
            .child(
                v_flex()
                    .flex_1()
                    .h_full()
                    .overflow_hidden()
                    .child(
                        h_flex()
                            .justify_between()
                            .items_start()
                            .gap_4()
                            .p_6()
                            .border_b_1()
                            .border_color(cx.theme().border)
                            .child(
                                v_flex()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_xs()
                                            .font_semibold()
                                            .text_color(cx.theme().primary)
                                            .child(i18n::step_progress(
                                                self.setup_wizard_cursor as i32 + 1,
                                                total as i32,
                                            )),
                                    )
                                    .child(div().text_xl().font_bold().child(title))
                                    .child(
                                        div()
                                            .max_w(px(720.))
                                            .text_color(cx.theme().muted_foreground)
                                            .child(detail),
                                    ),
                            )
                            .child(self.render_status_pill(cx)),
                    )
                    .child(
                        div()
                            .flex_1()
                            .overflow_y_scrollbar()
                            .p_6()
                            .child(
                                v_flex()
                                    .gap_4()
                                    .child(self.render_setup_content(step, window, cx)),
                            ),
                    )
                    .child(
                        h_flex()
                            .justify_between()
                            .items_center()
                            .p_6()
                            .border_t_1()
                            .border_color(cx.theme().border)
                            .child(
                                Button::new("setup-back")
                                    .label(i18n::back())
                                    .disabled(!can_back)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.go_back(cx);
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("setup-next")
                                    .primary()
                                    .icon(if is_review {
                                        IconName::Check
                                    } else {
                                        IconName::ArrowRight
                                    })
                                    .label(if is_review {
                                        i18n::create_setup()
                                    } else {
                                        i18n::next()
                                    })
                                    .disabled(!can_next)
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        if is_review {
                                            this.create_setup_requested(cx);
                                        } else {
                                            this.go_next(cx);
                                        }
                                        cx.notify();
                                    })),
                            ),
                    ),
            )
            .into_any_element()
    }

    fn render_step_item(&self, step: SetupStep, index: usize, cx: &App) -> AnyElement {
        let theme = cx.theme();
        let active = self.current_step() == step;
        let done = index < self.setup_wizard_cursor;
        let (title, _) = Self::step_copy(step);
        let badge_bg = if active || done {
            theme.primary
        } else {
            theme.muted
        };
        let badge_fg = if active || done {
            theme.primary_foreground
        } else {
            theme.muted_foreground
        };
        let title_color = if active {
            theme.foreground
        } else {
            theme.muted_foreground
        };
        let weight = if active {
            FontWeight::SEMIBOLD
        } else {
            FontWeight::NORMAL
        };
        let accent = theme.sidebar_accent;
        let radius = theme.radius;

        h_flex()
            .id(("step", index))
            .gap_3()
            .items_center()
            .rounded(radius)
            .px_3()
            .py_2()
            .when(active, move |this| this.bg(accent))
            .child(
                div()
                    .size(px(22.))
                    .rounded(px(11.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_xs()
                    .font_bold()
                    .bg(badge_bg)
                    .text_color(badge_fg)
                    .child(if done {
                        SharedString::from("✓")
                    } else {
                        SharedString::from((index + 1).to_string())
                    }),
            )
            .child(
                div()
                    .text_sm()
                    .font_weight(weight)
                    .text_color(title_color)
                    .child(title),
            )
            .into_any_element()
    }

    fn render_system_card(&self, cx: &App) -> AnyElement {
        let theme = cx.theme();
        let muted_fg = theme.muted_foreground;
        let radius = theme.radius;
        let accent = theme.sidebar_accent;
        v_flex()
            .gap_2()
            .p_3()
            .rounded(radius)
            .bg(accent)
            .child(
                h_flex()
                    .gap_1p5()
                    .items_center()
                    .text_color(muted_fg)
                    .child(Icon::new(IconName::Info).size_3())
                    .child(div().text_xs().child(i18n::label_system())),
            )
            .child(
                div()
                    .text_sm()
                    .child(SharedString::from(self.detected_text.clone())),
            )
            .into_any_element()
    }

    fn render_setup_content(
        &mut self,
        step: SetupStep,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        match step {
            SetupStep::Welcome => v_flex()
                .gap_3()
                .child(self.surface_card(
                    cx,
                    v_flex()
                        .gap_2()
                        .child(self.section_title(cx, i18n::label_detected_env()))
                        .child(
                            div()
                                .text_color(cx.theme().muted_foreground)
                                .child(SharedString::from(self.detected_text.clone())),
                        ),
                ))
                .child(self.surface_card(
                    cx,
                    v_flex()
                        .gap_2()
                        .child(self.section_title(cx, i18n::label_what_next()))
                        .child(
                            div()
                                .text_color(cx.theme().muted_foreground)
                                .child(i18n::label_what_next_detail()),
                        ),
                ))
                .into_any_element(),
            SetupStep::Conflict => v_flex()
                .gap_3()
                .child(
                    RadioGroup::vertical("conflict")
                        .selected_index(Some(self.selected_existing_action))
                        .on_click(cx.listener(|this, index: &usize, window, cx| {
                            this.selected_existing_action = *index;
                            this.apply_conflict_action(window, cx);
                            cx.notify();
                        }))
                        .child(
                            Radio::new("conflict-overwrite")
                                .label(i18n::conflict_overwrite_title())
                                .child(self.option_detail(cx, i18n::conflict_overwrite_detail())),
                        )
                        .child(
                            Radio::new("conflict-next")
                                .label(i18n::conflict_next_title())
                                .child(self.option_detail(cx, i18n::conflict_next_detail())),
                        )
                        .child(
                            Radio::new("conflict-cancel")
                                .label(i18n::conflict_cancel_title())
                                .child(self.option_detail(cx, i18n::conflict_cancel_detail())),
                        ),
                )
                .into_any_element(),
            SetupStep::Region => v_flex()
                .gap_3()
                .child(
                    RadioGroup::vertical("region")
                        .selected_index(Some(self.selected_region_index))
                        .on_click(cx.listener(|this, index: &usize, _, cx| {
                            this.selected_region_index = *index;
                            this.sync_wizard(cx);
                            cx.notify();
                        }))
                        .child(
                            Radio::new("region-other")
                                .label(i18n::region_other_title())
                                .child(self.option_detail(cx, i18n::region_other_detail())),
                        )
                        .child(
                            Radio::new("region-cn")
                                .label(i18n::region_cn_title())
                                .child(self.option_detail(cx, i18n::region_cn_detail())),
                        ),
                )
                .into_any_element(),
            SetupStep::Preset => v_flex()
                .gap_3()
                .child(
                    RadioGroup::vertical("preset")
                        .selected_index(Some(self.selected_preset_index))
                        .on_click(cx.listener(|this, index: &usize, _, cx| {
                            this.selected_preset_index = *index;
                            this.sync_wizard(cx);
                            cx.notify();
                        }))
                        .child(
                            Radio::new("preset-minimal")
                                .label(i18n::preset_minimal_title())
                                .child(self.option_detail(cx, i18n::preset_minimal_detail())),
                        )
                        .child(
                            Radio::new("preset-basic")
                                .label(i18n::preset_basic_title())
                                .child(self.option_detail(cx, i18n::preset_basic_detail())),
                        )
                        .child(
                            Radio::new("preset-brave")
                                .label(i18n::preset_brave_title())
                                .child(self.option_detail(cx, i18n::preset_brave_detail())),
                        ),
                )
                .into_any_element(),
            SetupStep::Backend => v_flex()
                .gap_3()
                .child(
                    RadioGroup::vertical("backend")
                        .selected_index(Some(self.selected_backend_index))
                        .on_click(cx.listener(|this, index: &usize, _, cx| {
                            if this
                                .backend_choices
                                .get(*index)
                                .map(|choice| choice.enabled)
                                .unwrap_or(false)
                            {
                                this.selected_backend_index = *index;
                                this.sync_wizard(cx);
                                cx.notify();
                            }
                        }))
                        .children(self.backend_choices.iter().enumerate().map(|(index, choice)| {
                            let title = SharedString::from(choice.title.clone());
                            let detail = SharedString::from(choice.detail.clone());
                            let detail_warning = detail.clone();
                            let enabled = choice.enabled;
                            Radio::new(("backend-radio", index))
                                .label(title)
                                .disabled(!enabled)
                                .when(!enabled, |this| {
                                    this.child(self.option_detail_warning(cx, detail_warning))
                                })
                                .when(enabled, |this| this.child(self.option_detail(cx, detail)))
                        })),
                )
                .when(self.backend_choices.is_empty(), |this| {
                    this.child(self.surface_card(
                        cx,
                        div()
                            .text_color(cx.theme().danger_foreground)
                            .child(SharedString::from(self.setup_warning.clone())),
                    ))
                })
                .into_any_element(),
            SetupStep::Paths => {
                let cache_row = self.field_row(
                    "cache-dir",
                    i18n::cache_label(),
                    self.cache_input.clone(),
                    PickerTarget::CacheDir,
                    window,
                    cx,
                );
                v_flex()
                    .gap_4()
                    .child(cache_row)
                    .child(self.render_advanced_settings(window, cx))
                    .when(!self.setup_warning.is_empty(), |this| {
                        this.child(self.warning_block(cx))
                    })
                    .into_any_element()
            }
            SetupStep::Review => {
                let primary = cx.theme().primary;
                let fg = cx.theme().foreground;
                v_flex()
                    .gap_4()
                    .child(self.surface_card(
                        cx,
                        v_flex()
                            .gap_2()
                            .children(self.setup_review_text.lines().map(|line| {
                                h_flex()
                                    .gap_2()
                                    .items_start()
                                    .child(
                                        div()
                                            .mt(px(6.))
                                            .size(px(6.))
                                            .rounded(px(3.))
                                            .bg(primary),
                                    )
                                    .child(
                                        div()
                                            .text_color(fg)
                                            .child(SharedString::from(line.to_owned())),
                                    )
                            })),
                    ))
                    .when(!self.setup_warning.is_empty(), |this| {
                        this.child(self.warning_block(cx))
                    })
                    .into_any_element()
            }
        }
    }

    fn render_advanced_settings(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let config_row = self.field_row(
            "config-path",
            i18n::config_field_label(),
            self.config_input.clone(),
            PickerTarget::ConfigPath,
            window,
            cx,
        );
        let bootstrap_row = self.field_row(
            "bootstrap-path",
            i18n::bootstrap_label(),
            self.bootstrap_input.clone(),
            PickerTarget::BootstrapPath,
            window,
            cx,
        );
        let muted_fg = cx.theme().muted_foreground;
        let open = self.advanced_settings_open;

        self.surface_card(
            cx,
            v_flex()
                .gap_3()
                .child(
                    h_flex()
                        .id("advanced-toggle")
                        .justify_between()
                        .items_center()
                        .cursor_pointer()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.advanced_settings_open = !this.advanced_settings_open;
                            this.sync_wizard(cx);
                            cx.notify();
                        }))
                        .child(
                            h_flex()
                                .gap_2()
                                .items_center()
                                .child(
                                    Icon::new(IconName::Settings2)
                                        .size_4()
                                        .text_color(muted_fg),
                                )
                                .child(
                                    div().font_semibold().child(i18n::advanced_settings_label()),
                                ),
                        )
                        .child(
                            Icon::new(if open {
                                IconName::ChevronDown
                            } else {
                                IconName::ChevronRight
                            })
                            .size_4()
                            .text_color(muted_fg),
                        ),
                )
                .when(open, |this| {
                    this.child(
                        div()
                            .text_color(cx.theme().muted_foreground)
                            .child(i18n::advanced_settings_hint()),
                    )
                    .child(config_row)
                    .child(bootstrap_row)
                }),
        )
    }

    fn render_dashboard(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let border = theme.border;
        let radius = theme.radius;
        let muted_fg = theme.muted_foreground;
        let group_box = theme.group_box;
        let fg = theme.foreground;
        let mono_font = theme.mono_font_family.clone();
        let mono_size = theme.mono_font_size;

        let header = h_flex()
            .justify_between()
            .items_center()
            .gap_4()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(border)
            .child(
                v_flex()
                    .child(div().text_lg().font_bold().child("Lumen Hub"))
                    .child(
                        div()
                            .text_xs()
                            .text_color(muted_fg)
                            .child(i18n::app_subtitle()),
                    ),
            )
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(self.render_locale_toggle(cx))
                    .child(self.render_theme_toggle(cx))
                    .child(
                        Button::new("start")
                            .primary()
                            .large()
                            .label(i18n::start())
                            .disabled(!self.can_start)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.start_requested(cx);
                                cx.notify();
                            })),
                    )
                    .child(
                        Button::new("stop")
                            .danger()
                            .large()
                            .label(i18n::stop())
                            .disabled(!self.can_stop)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.stop_requested();
                                cx.notify();
                            })),
                    ),
            );

        let status_card = self.surface_card(
            cx,
            v_flex()
                .gap_4()
                .child(
                    h_flex()
                        .justify_between()
                        .items_center()
                        .child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().muted_foreground)
                                .child(i18n::label_status()),
                        )
                        .child(self.render_status_pill(cx)),
                )
                .child(self.kv_row(cx, i18n::label_config(), &self.config_path))
                .child(self.kv_row(cx, i18n::label_profile(), &self.profile_text)),
        );

        let info_card = self.surface_card(
            cx,
            v_flex()
                .gap_2()
                .child(self.section_title(cx, i18n::label_runtime()))
                .child(
                    div()
                        .text_color(cx.theme().muted_foreground)
                        .child(i18n::label_runtime_detail()),
                ),
        );

        let log_panel = v_flex()
            .flex_1()
            .h_full()
            .overflow_hidden()
            .rounded(radius)
            .border_1()
            .border_color(border)
            .bg(group_box)
            .child(
                h_flex()
                    .justify_between()
                    .items_center()
                    .px_4()
                    .py_2p5()
                    .border_b_1()
                    .border_color(border)
                    .child(
                        h_flex()
                            .gap_2()
                            .items_center()
                            .text_color(muted_fg)
                            .child(Icon::new(IconName::SquareTerminal).size_4())
                            .child(div().text_sm().font_semibold().child(i18n::label_log())),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(muted_fg)
                            .child(i18n::log_line_count(self.log_lines.len())),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .overflow_y_scrollbar()
                    .p_4()
                    .child(
                        v_flex().gap_1().children(
                            self.log_lines.iter().enumerate().map(|(index, line)| {
                                div()
                                    .id(("log", index))
                                    .text_color(fg)
                                    .font_family(mono_font.clone())
                                    .text_size(mono_size)
                                    .child(SharedString::from(line.clone()))
                            }),
                        ),
                    ),
            );

        v_flex()
            .id("dashboard")
            .size_full()
            .overflow_hidden()
            .child(header)
            .child(
                h_flex()
                    .flex_1()
                    .overflow_hidden()
                    .gap_5()
                    .p_6()
                    .child(
                        v_flex()
                            .w(px(320.))
                            .flex_shrink_0()
                            .gap_4()
                            .child(status_card)
                            .child(info_card),
                    )
                    .child(log_panel),
            )
            .into_any_element()
    }

    fn render_status_pill(&self, cx: &App) -> AnyElement {
        let theme = cx.theme();
        let (bg, fg, show_dot) = match self.status_kind {
            StatusKind::Running => (theme.success, theme.success_foreground, true),
            StatusKind::Error => (theme.danger, theme.danger_foreground, false),
            StatusKind::Setup | StatusKind::Stopping => {
                (theme.warning, theme.warning_foreground, false)
            }
            StatusKind::Preparing | StatusKind::Creating | StatusKind::Checking => {
                (theme.info, theme.info_foreground, true)
            }
            StatusKind::Ready | StatusKind::Stopped => (theme.muted, theme.muted_foreground, false),
        };

        h_flex()
            .gap_1p5()
            .items_center()
            .rounded(px(999.))
            .bg(bg)
            .text_color(fg)
            .px_3()
            .py_1()
            .text_xs()
            .font_semibold()
            .when(show_dot, |this| {
                this.child(div().size(px(6.)).rounded_full().bg(fg.opacity(0.9)))
            })
            .child(SharedString::from(self.status_text.clone()))
            .into_any_element()
    }

    fn render_locale_toggle(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let next = i18n::Locale::current().toggle();
        Button::new("locale-toggle")
            .ghost()
            .small()
            .label(next.code())
            .on_click(cx.listener(|this, _, _, cx| {
                let next = i18n::Locale::current().toggle();
                i18n::set_locale(next);
                this.refresh_localized(cx);
                cx.notify();
            }))
    }

    fn render_theme_toggle(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let is_dark = cx.theme().is_dark();
        Button::new("theme-toggle")
            .ghost()
            .small()
            .icon(if is_dark { IconName::Sun } else { IconName::Moon })
            .on_click(cx.listener(|_, _, window, cx| {
                let is_dark = cx.theme().is_dark();
                let next = if is_dark { ThemeMode::Light } else { ThemeMode::Dark };
                Theme::change(next, Some(window), cx);
            }))
    }

    fn surface_card(&self, cx: &App, body: impl IntoElement) -> AnyElement {
        let theme = cx.theme();
        v_flex()
            .w_full()
            .rounded(theme.radius)
            .border_1()
            .border_color(theme.border)
            .bg(theme.title_bar)
            .p_4()
            .child(body)
            .into_any_element()
    }

    fn section_title(&self, cx: &App, title: impl Into<SharedString>) -> AnyElement {
        let theme = cx.theme();
        div()
            .text_sm()
            .font_semibold()
            .text_color(theme.foreground)
            .child(title.into())
            .into_any_element()
    }

    fn option_detail(&self, cx: &App, detail: SharedString) -> AnyElement {
        let theme = cx.theme();
        div()
            .text_sm()
            .text_color(theme.muted_foreground)
            .child(detail)
            .into_any_element()
    }

    fn option_detail_warning(&self, cx: &App, detail: SharedString) -> AnyElement {
        let theme = cx.theme();
        div()
            .text_sm()
            .text_color(theme.danger)
            .child(detail)
            .into_any_element()
    }

    fn field_row(
        &self,
        id: &'static str,
        label: SharedString,
        input: Entity<InputState>,
        target: PickerTarget,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let label_color = cx.theme().foreground;
        v_flex()
            .id(id)
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .font_semibold()
                    .text_color(label_color)
                    .child(label),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(Input::new(&input).cleanable(true).flex_1())
                    .child(
                        Button::new(match target {
                            PickerTarget::CacheDir => "browse-cache-dir",
                            PickerTarget::ConfigPath => "browse-config-path",
                            PickerTarget::BootstrapPath => "browse-bootstrap-path",
                        })
                        .icon(IconName::FolderOpen)
                        .label(i18n::browse())
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.browse_path(target, window, cx);
                            cx.notify();
                        })),
                    ),
            )
            .into_any_element()
    }

    fn warning_block(&self, _cx: &App) -> AnyElement {
        gpui_component::alert::Alert::warning(
            "setup-warning",
            SharedString::from(self.setup_warning.clone()),
        )
        .title(i18n::label_warning().to_string())
        .into_any_element()
    }

    fn kv_row(&self, cx: &App, label: impl Into<SharedString>, value: &str) -> AnyElement {
        let theme = cx.theme();
        let label = label.into();
        v_flex()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(label),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.foreground)
                    .child(SharedString::from(value.to_owned())),
            )
            .into_any_element()
    }
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

fn profile_label(plan: &lumen_launcher::StartPlan) -> String {
    match &plan.bootstrap {
        Some(bootstrap) => i18n::profile_summary(
            &bootstrap.preset,
            &bootstrap.backend,
            &bootstrap.release_profile,
        )
        .to_string(),
        None => i18n::profile_only(&plan.profile).to_string(),
    }
}

fn non_empty_path(value: &str, label: &str) -> Result<PathBuf, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} is required"));
    }
    Ok(setup::expand_tilde(trimmed))
}

fn optional_path(value: &str) -> Option<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(setup::expand_tilde(trimmed))
    }
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
    let _ = tx.send(UiMessage::Profile(profile_label(&plan)));

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

    let _ = tx.send(UiMessage::Status {
        text: i18n::status_running().to_string(),
        kind: StatusKind::Running,
    });
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
                let _ = tx.send(UiMessage::Status {
                    text: i18n::status_stopped().to_string(),
                    kind: StatusKind::Stopped,
                });
                let _ = tx.send(UiMessage::Controls {
                    can_start: true,
                    can_stop: false,
                });
                let _ = tx.send(UiMessage::Log("lumen-hub exited".to_owned()));
                return;
            }
            Some(Ok(status)) => {
                let _ = tx.send(UiMessage::Status {
                    text: i18n::status_exited_with_error().to_string(),
                    kind: StatusKind::Error,
                });
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
    conflict_action: usize,
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

#[derive(Debug, Clone, Copy)]
enum InputTarget {
    Cache,
    Config,
    Bootstrap,
}

#[derive(Debug, Clone, Copy)]
enum StartGate {
    Start,
    Busy,
    Poisoned,
}

#[derive(Debug)]
enum StopOutcome {
    Stopped,
    Preparing,
    NotRunning,
    Error(String),
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetupStep {
    Welcome,
    Conflict,
    Region,
    Preset,
    Backend,
    Paths,
    Review,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatusKind {
    Checking,
    Ready,
    Preparing,
    Setup,
    Creating,
    Running,
    Stopping,
    Stopped,
    Error,
}

#[derive(Debug)]
enum UiMessage {
    Status { text: String, kind: StatusKind },
    ConfigPath(String),
    Profile(String),
    Controls { can_start: bool, can_stop: bool },
    SetupRequired(String),
    SetupReady(SetupSummary),
    Log(String),
    Error(String),
}

#[derive(Debug, Clone)]
struct BackendChoiceView {
    title: String,
    detail: String,
    enabled: bool,
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
        let _ = self.tx.send(UiMessage::Status {
            text: i18n::status_fetching_manifest().to_string(),
            kind: StatusKind::Preparing,
        });
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
        let _ = self.tx.send(UiMessage::Status {
            text: i18n::status_downloading_hub().to_string(),
            kind: StatusKind::Preparing,
        });
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
        let status = match (i18n::Locale::current(), total) {
            (i18n::Locale::Zh, Some(total)) => format!(
                "下载 {file_name}：{} / {}",
                format_bytes(written),
                format_bytes(total)
            ),
            (i18n::Locale::Zh, None) => {
                format!("下载 {file_name}：{}", format_bytes(written))
            }
            (i18n::Locale::En, Some(total)) => format!(
                "Downloading {file_name}: {} / {}",
                format_bytes(written),
                format_bytes(total)
            ),
            (i18n::Locale::En, None) => {
                format!("Downloading {file_name}: {}", format_bytes(written))
            }
        };
        let _ = self.tx.send(UiMessage::Status {
            text: status,
            kind: StatusKind::Preparing,
        });
    }

    fn download_finished(&mut self, file_name: &str, written: u64) {
        let _ = self.tx.send(UiMessage::Log(format!(
            "downloaded {file_name} ({})",
            format_bytes(written)
        )));
    }

    fn verify_started(&mut self, path: &Path) {
        let _ = self.tx.send(UiMessage::Status {
            text: i18n::status_verifying_hub().to_string(),
            kind: StatusKind::Preparing,
        });
        let _ = self
            .tx
            .send(UiMessage::Log(format!("verifying {}", path.display())));
    }

    fn verify_finished(&mut self, _path: &Path) {
        let _ = self.tx.send(UiMessage::Log("checksum ok".to_owned()));
    }

    fn extract_started(&mut self, path: &Path) {
        let _ = self.tx.send(UiMessage::Status {
            text: i18n::status_extracting_hub().to_string(),
            kind: StatusKind::Preparing,
        });
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
        let _ = self.tx.send(UiMessage::Status {
            text: i18n::status_starting_hub().to_string(),
            kind: StatusKind::Preparing,
        });
        let _ = self
            .tx
            .send(UiMessage::Log(format!("starting {}", hub_path.display())));
    }
}
