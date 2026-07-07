use slint::SharedString;

use crate::{AppWindow, i18n};

pub const STEP_WELCOME: i32 = 0;
pub const STEP_CONFLICT: i32 = 1;
pub const STEP_REGION: i32 = 2;
pub const STEP_PRESET: i32 = 3;
pub const STEP_BACKEND: i32 = 4;
pub const STEP_PATHS: i32 = 5;
pub const STEP_REVIEW: i32 = 6;

pub fn step_sequence(existing: bool) -> Vec<i32> {
    let mut steps = vec![STEP_WELCOME];
    if existing {
        steps.push(STEP_CONFLICT);
    }
    steps.extend([
        STEP_REGION,
        STEP_PRESET,
        STEP_BACKEND,
        STEP_PATHS,
        STEP_REVIEW,
    ]);
    steps
}

pub fn reset_wizard(app: &AppWindow) {
    app.set_setup_wizard_cursor(0);
    sync_wizard(app);
}

pub fn sync_wizard(app: &AppWindow) {
    let sequence = step_sequence(app.get_existing_visible());
    let total = sequence.len().max(1) as i32;
    let cursor = app.get_setup_wizard_cursor().clamp(0, total - 1) as usize;
    let step_id = sequence[cursor];

    app.set_setup_wizard_cursor(cursor as i32);
    app.set_setup_step_id(step_id);
    app.set_setup_step_index((cursor + 1) as i32);
    app.set_setup_step_total(total);
    app.set_setup_progress_text(i18n::step_progress(cursor as i32 + 1, total));
    app.set_setup_finish_visible(step_id == STEP_REVIEW);
    app.set_setup_back_enabled(cursor > 0);

    let (title, detail) = step_copy(step_id);
    app.set_setup_step_title(title);
    app.set_setup_step_detail(detail);

    if step_id == STEP_REVIEW {
        app.set_setup_review_text(build_review_text(app).into());
    }

    app.set_setup_next_enabled(can_advance(app, step_id));
    update_setup_warning(app);
}

pub fn go_next(app: &AppWindow) -> Result<(), String> {
    let step_id = current_step_id(app);
    if !can_advance(app, step_id) {
        return Err(advance_block_reason(app, step_id));
    }

    let sequence = step_sequence(app.get_existing_visible());
    let total = sequence.len() as i32;
    let next = (app.get_setup_wizard_cursor() + 1).min(total - 1);
    app.set_setup_wizard_cursor(next);
    sync_wizard(app);
    Ok(())
}

pub fn go_back(app: &AppWindow) {
    let prev = (app.get_setup_wizard_cursor() - 1).max(0);
    app.set_setup_wizard_cursor(prev);
    sync_wizard(app);
}

fn current_step_id(app: &AppWindow) -> i32 {
    let sequence = step_sequence(app.get_existing_visible());
    let cursor = app.get_setup_wizard_cursor().clamp(0, sequence.len() as i32 - 1) as usize;
    sequence[cursor]
}

fn step_copy(step_id: i32) -> (SharedString, SharedString) {
    match step_id {
        STEP_WELCOME => (i18n::welcome_title(), i18n::welcome_detail()),
        STEP_CONFLICT => (i18n::conflict_step_title(), i18n::conflict_step_detail()),
        STEP_REGION => (i18n::region_step_title(), i18n::region_step_detail()),
        STEP_PRESET => (i18n::preset_step_title(), i18n::preset_step_detail()),
        STEP_BACKEND => (i18n::backend_step_title(), i18n::backend_step_detail()),
        STEP_PATHS => (i18n::paths_step_title(), i18n::paths_step_detail()),
        STEP_REVIEW => (i18n::review_step_title(), i18n::review_step_detail()),
        _ => (SharedString::default(), SharedString::default()),
    }
}

fn can_advance(app: &AppWindow, step_id: i32) -> bool {
    match step_id {
        STEP_CONFLICT => app.get_selected_existing_action() != 2,
        STEP_BACKEND => selected_backend(app).is_ok(),
        STEP_PATHS => paths_valid(app),
        STEP_REVIEW => {
            app.get_selected_existing_action() != 2
                && selected_backend(app).is_ok()
                && paths_valid(app)
        }
        _ => true,
    }
}

fn advance_block_reason(app: &AppWindow, step_id: i32) -> String {
    match step_id {
        STEP_CONFLICT if app.get_selected_existing_action() == 2 => {
            i18n::bi("Setup cancelled.", "已取消设置。").to_string()
        }
        STEP_BACKEND => selected_backend(app)
            .err()
            .unwrap_or_else(|| i18n::warning_backend_unavailable()),
        STEP_PATHS | STEP_REVIEW if !paths_valid(app) => {
            i18n::bi(
                "All storage paths are required.",
                "请填写所有存储路径。",
            )
            .to_string()
        }
        _ => i18n::bi("Complete this step first.", "请先完成当前步骤。").to_string(),
    }
}

fn paths_valid(app: &AppWindow) -> bool {
    !app.get_cache_dir_input().is_empty()
        && !app.get_config_path_input().is_empty()
        && !app.get_bootstrap_path_input().is_empty()
}

fn build_review_text(app: &AppWindow) -> SharedString {
    let preset = selected_preset(app);
    let region = if app.get_selected_region_index() == 1 {
        i18n::review_region_cn()
    } else {
        i18n::review_region_other()
    };
    let backend = selected_backend(app)
        .map(|backend| i18n::review_backend(backend.name, backend.release_profile))
        .unwrap_or_else(|_| i18n::bi("Backend: unavailable", "后端：不可用"));

    let lines = vec![
        region,
        i18n::review_preset(preset.display_title()),
        backend,
        i18n::review_path(
            "Cache directory",
            "缓存目录",
            &app.get_cache_dir_input().to_string(),
        ),
    ];

    let mut lines = lines;
    if app.get_advanced_settings_open() {
        lines.push(i18n::review_path(
            "Config file",
            "配置文件",
            &app.get_config_path_input().to_string(),
        ));
        lines.push(i18n::review_path(
            "Bootstrap file",
            "引导文件",
            &app.get_bootstrap_path_input().to_string(),
        ));
    }

    SharedString::from(lines.join("\n"))
}

pub fn selected_preset(app: &AppWindow) -> lumen_launcher::setup::Preset {
    let presets = lumen_launcher::setup::Preset::all();
    let index = app
        .get_selected_preset_index()
        .clamp(0, (presets.len() - 1) as i32) as usize;
    presets[index]
}

pub fn selected_backend(app: &AppWindow) -> Result<lumen_launcher::setup::Backend, String> {
    let platform =
        lumen_launcher::setup::current_platform_profile().map_err(|error| error.to_string())?;
    let choices = lumen_launcher::setup::backend_choices(platform);
    let index = app.get_selected_backend_index().max(0) as usize;
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

pub fn update_setup_warning(app: &AppWindow) {
    let mut warnings = Vec::new();
    let preset = selected_preset(app);
    let memory = lumen_launcher::setup::detect_memory();
    if let Some(total_gb) = memory.total_gb
        && total_gb < preset.min_ram_gb as f64
    {
        warnings.push(i18n::warning_ram(
            preset.display_title(),
            preset.min_ram_gb,
            total_gb,
        ));
    }

    if let Ok(platform) = lumen_launcher::setup::current_platform_profile() {
        let choices = lumen_launcher::setup::backend_choices(platform);
        let index = app.get_selected_backend_index().max(0) as usize;
        if let Some(choice) = choices.get(index)
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

    let cache_dir =
        lumen_launcher::setup::expand_tilde(app.get_cache_dir_input().to_string().trim());
    if cache_dir.exists()
        && let Some(free_gb) = lumen_launcher::setup::free_disk_gb(&cache_dir)
        && free_gb < preset.min_disk_gb as f64
    {
        warnings.push(i18n::warning_disk(
            &cache_dir.display().to_string(),
            preset.display_title(),
            free_gb,
            preset.min_disk_gb,
        ));
    }

    app.set_setup_warning(warnings.join("\n").into());
    if app.get_setup_step_id() == STEP_REVIEW {
        app.set_setup_review_text(build_review_text(app));
    }
}
