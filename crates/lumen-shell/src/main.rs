#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    io::{BufRead, BufReader, Read},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use lumen_launcher::{
    HubStdio, LaunchObserver, LauncherError, RunningHub, StartOptions, format_bytes, prepare_hub,
    resolve_start_plan, spawn_hub,
};
use slint::{ComponentHandle, SharedString, Timer, TimerMode};

slint::include_modules!();

const MAX_LOG_LINES: usize = 600;

fn main() -> Result<(), slint::PlatformError> {
    let app = AppWindow::new()?;
    let process_state = Arc::new(Mutex::new(ProcessState::default()));
    let (tx, rx) = mpsc::channel::<UiMessage>();

    app.set_status_text("Idle".into());
    app.set_profile_text("Profile: from ~/.lumen/bootstrap.json".into());
    app.set_config_path(default_config_label().into());
    app.set_log_text("Ready.\n".into());

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
    app.on_start_requested(move || {
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

        let tx = start_tx.clone();
        let process_state = Arc::clone(&start_process_state);
        thread::spawn(move || start_hub(tx, process_state));
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
                    let _ = stop_tx.send(UiMessage::Status("Stopping".to_owned()));
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

    app.run()
}

fn start_hub(tx: mpsc::Sender<UiMessage>, process_state: Arc<Mutex<ProcessState>>) {
    let mut observer = ShellObserver { tx: tx.clone() };
    let options = StartOptions::default();
    let plan = match resolve_start_plan(options) {
        Ok(plan) => plan,
        Err(error) => {
            let _ = tx.send(UiMessage::Error(error.to_string()));
            clear_launching(&process_state, &tx);
            return;
        }
    };

    let _ = tx.send(UiMessage::ConfigPath(
        plan.config_path.display().to_string(),
    ));
    let _ = tx.send(UiMessage::Profile(format!("Profile: {}", plan.profile)));

    let hub_path = match prepare_hub(&plan, &mut observer) {
        Ok(path) => path,
        Err(error) => {
            let _ = tx.send(UiMessage::Error(error.to_string()));
            clear_launching(&process_state, &tx);
            return;
        }
    };

    let mut hub = match spawn_hub(&plan, &hub_path, HubStdio::Piped, &mut observer) {
        Ok(hub) => hub,
        Err(error) => {
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

    let _ = tx.send(UiMessage::Status("Running".to_owned()));
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
                let _ = tx.send(UiMessage::Status("Stopped".to_owned()));
                let _ = tx.send(UiMessage::Log("lumen-hub exited".to_owned()));
                return;
            }
            Some(Ok(status)) => {
                let _ = tx.send(UiMessage::Status("Exited with error".to_owned()));
                let _ = tx.send(UiMessage::Error(format!(
                    "lumen-hub {}",
                    lumen_launcher::FormattedExitStatus(status)
                )));
                return;
            }
            Some(Err(error)) => {
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
        let reader = BufReader::new(reader);
        for line in reader.lines() {
            match line {
                Ok(line) => {
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

fn apply_message(app: &AppWindow, message: UiMessage) {
    match message {
        UiMessage::Status(status) => app.set_status_text(status.into()),
        UiMessage::ConfigPath(path) => app.set_config_path(path.into()),
        UiMessage::Profile(profile) => app.set_profile_text(profile.into()),
        UiMessage::Log(line) => append_log(app, &line),
        UiMessage::Error(error) => {
            app.set_status_text("Error".into());
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

fn default_config_label() -> String {
    match lumen_launcher::default_lumen_dir() {
        Ok(path) => path.join("config.yaml").display().to_string(),
        Err(_) => "~/.lumen/config.yaml".to_owned(),
    }
}

#[derive(Debug)]
enum UiMessage {
    Status(String),
    ConfigPath(String),
    Profile(String),
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
        let _ = self
            .tx
            .send(UiMessage::Status("Fetching manifest".to_owned()));
    }

    fn manifest_fetched(&mut self, version: &str) {
        let _ = self
            .tx
            .send(UiMessage::Log(format!("release manifest {version}")));
    }

    fn hub_already_installed(&mut self, hub_path: &std::path::Path) {
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
        let _ = self
            .tx
            .send(UiMessage::Status("Downloading hub".to_owned()));
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

    fn verify_started(&mut self, path: &std::path::Path) {
        let _ = self.tx.send(UiMessage::Status("Verifying hub".to_owned()));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("verifying {}", path.display())));
    }

    fn verify_finished(&mut self, _path: &std::path::Path) {
        let _ = self.tx.send(UiMessage::Log("checksum ok".to_owned()));
    }

    fn extract_started(&mut self, path: &std::path::Path) {
        let _ = self.tx.send(UiMessage::Status("Extracting hub".to_owned()));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("extracting {}", path.display())));
    }

    fn hub_installed(&mut self, hub_path: &std::path::Path) {
        let _ = self.tx.send(UiMessage::Log(format!(
            "lumen-hub ready: {}",
            hub_path.display()
        )));
    }

    fn hub_starting(&mut self, hub_path: &std::path::Path) {
        let _ = self
            .tx
            .send(UiMessage::Status("Starting lumen-hub".to_owned()));
        let _ = self
            .tx
            .send(UiMessage::Log(format!("starting {}", hub_path.display())));
    }
}
