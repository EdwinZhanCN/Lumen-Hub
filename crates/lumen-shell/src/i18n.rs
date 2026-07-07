use slint::{ComponentHandle, SharedString};

use crate::AppWindow;

pub fn apply_static_labels(app: &AppWindow) {
    let i18n = app.global::<crate::I18n>();
    i18n.set_app_subtitle(bi("Local model runtime", "本地模型运行时"));
    i18n.set_start(bi("Start", "启动"));
    i18n.set_stop(bi("Stop", "停止"));
    i18n.set_status_label(bi("Status", "状态"));
    i18n.set_config_label(bi("Config file", "配置文件"));
    i18n.set_log_label(bi("Log", "日志"));
    i18n.set_browse(bi("Browse", "浏览"));
    i18n.set_back(bi("Back", "上一步"));
    i18n.set_next(bi("Next", "下一步"));
    i18n.set_create_setup(bi("Create setup", "创建配置"));
    i18n.set_cache_label(bi("Cache directory", "缓存目录"));
    i18n.set_config_field_label(bi("Config file", "配置文件"));
    i18n.set_bootstrap_label(bi("Bootstrap file", "引导文件"));
    i18n.set_advanced_settings_label(bi("Advanced settings", "高级设置"));
    i18n.set_advanced_settings_hint(bi(
        "Optional paths for config and bootstrap files. Defaults are recommended for most users.",
        "可选的配置文件与引导文件路径。大多数用户使用默认位置即可。",
    ));
    i18n.set_region_other_title(bi("International", "国际"));
    i18n.set_region_other_detail(bi("Hugging Face", "Hugging Face 官方源"));
    i18n.set_region_cn_title(bi("China", "中国"));
    i18n.set_region_cn_detail(bi("hf-mirror.com mirror", "hf-mirror.com 镜像"));
    i18n.set_preset_0_title(bi("minimal", "最小"));
    i18n.set_preset_0_detail(bi(
        "SigLIP + face · RAM 4 GB, GPU 2 GB",
        "SigLIP + 人脸 · 内存 4 GB，GPU 2 GB",
    ));
    i18n.set_preset_1_title(bi("basic", "基础"));
    i18n.set_preset_1_detail(bi(
        "SigLIP, face, OCR, BioCLIP core · RAM 6 GB",
        "SigLIP、人脸、OCR、BioCLIP 核心 · 内存 6 GB",
    ));
    i18n.set_preset_2_title(bi("brave", "激进"));
    i18n.set_preset_2_detail(bi(
        "Larger SigLIP and full BioCLIP catalog · RAM 8 GB",
        "更大 SigLIP 与完整 BioCLIP 目录 · 内存 8 GB",
    ));
    i18n.set_conflict_overwrite_title(bi("Overwrite", "覆盖"));
    i18n.set_conflict_overwrite_detail(bi(
        "Replace the default config and bootstrap files.",
        "覆盖默认的配置与引导文件。",
    ));
    i18n.set_conflict_next_title(bi("Create next to it", "另存为新文件"));
    i18n.set_conflict_next_detail(bi(
        "Save new config and bootstrap files alongside the existing ones.",
        "在现有文件旁另存新的配置与引导文件。",
    ));
    i18n.set_conflict_cancel_title(bi("Cancel setup", "取消设置"));
    i18n.set_conflict_cancel_detail(bi(
        "Keep existing files unchanged.",
        "保留现有文件，不进行设置。",
    ));
}

pub fn bi(en: &str, zh: &str) -> SharedString {
    format!("{en} / {zh}").into()
}

pub fn step_progress(current: i32, total: i32) -> SharedString {
    bi(
        &format!("Step {current} of {total}"),
        &format!("第 {current} 步，共 {total} 步"),
    )
}

pub fn welcome_title() -> SharedString {
    bi("Welcome to Lumen Hub", "欢迎使用 Lumen Hub")
}

pub fn welcome_detail() -> SharedString {
    bi(
        "This guided setup prepares Lumen Hub on your machine. Choose a preset, backend, and cache directory for downloaded models.",
        "本向导将在本机完成 Lumen Hub 初始设置。请选择预设、后端以及用于存放下载模型的缓存目录。",
    )
}

pub fn region_step_title() -> SharedString {
    bi("Choose model download region", "选择模型下载区域")
}

pub fn region_step_detail() -> SharedString {
    bi(
        "Pick where Lumen Hub should download model weights.",
        "选择 Lumen Hub 下载模型权重的来源。",
    )
}

pub fn preset_step_title() -> SharedString {
    bi("Choose a preset", "选择预设")
}

pub fn preset_step_detail() -> SharedString {
    bi(
        "Presets bundle services and resource guidance for your machine.",
        "预设包含服务组合以及针对本机的资源建议。",
    )
}

pub fn backend_step_title() -> SharedString {
    bi("Choose compute backend", "选择计算后端")
}

pub fn backend_step_detail() -> SharedString {
    bi(
        "Select the GPU/CPU backend for this installation.",
        "选择本次安装使用的 GPU/CPU 后端。",
    )
}

pub fn paths_step_title() -> SharedString {
    bi("Choose cache directory", "选择缓存目录")
}

pub fn paths_step_detail() -> SharedString {
    bi(
        "Downloaded model weights are stored here on first use.",
        "模型权重将在首次使用时下载并保存在此目录。",
    )
}

pub fn conflict_step_title() -> SharedString {
    bi("Existing setup found", "发现已有配置")
}

pub fn conflict_step_detail() -> SharedString {
    bi(
        "Config or bootstrap files already exist. Choose how to continue.",
        "已存在配置或引导文件，请选择如何处理。",
    )
}

pub fn review_step_title() -> SharedString {
    bi("Review and create", "确认并创建")
}

pub fn review_step_detail() -> SharedString {
    bi(
        "Check your choices, then create the Lumen setup.",
        "确认选择无误后，创建 Lumen 配置。",
    )
}

pub fn detected_system(os: &str, arch: &str, ram: &str) -> SharedString {
    bi(
        &format!("Detected: {os} / {arch} | {ram}"),
        &format!("检测到：{os} / {arch} | {ram}"),
    )
}

pub fn ram_known(gb: f64) -> String {
    format!("RAM {gb:.1} GB")
}

pub fn ram_known_zh(gb: f64) -> String {
    format!("内存 {gb:.1} GB")
}

pub fn ram_unknown_en() -> &'static str {
    "RAM unknown"
}

pub fn ram_unknown_zh() -> &'static str {
    "内存未知"
}

pub fn status_checking_setup() -> SharedString {
    bi("Checking setup", "正在检查配置")
}

pub fn status_setup_required() -> SharedString {
    bi("Setup required", "需要完成设置")
}

pub fn status_ready() -> SharedString {
    bi("Ready", "就绪")
}

pub fn status_preparing() -> SharedString {
    bi("Preparing", "准备中")
}

pub fn status_creating_setup() -> SharedString {
    bi("Creating setup", "正在创建配置")
}

pub fn status_running() -> SharedString {
    bi("Running", "运行中")
}

pub fn status_stopping() -> SharedString {
    bi("Stopping", "正在停止")
}

pub fn status_stopped() -> SharedString {
    bi("Stopped", "已停止")
}

pub fn status_error() -> SharedString {
    bi("Error", "错误")
}

pub fn status_exited_with_error() -> SharedString {
    bi("Exited with error", "异常退出")
}

pub fn status_fetching_manifest() -> SharedString {
    bi("Fetching manifest", "正在获取清单")
}

pub fn status_downloading_hub() -> SharedString {
    bi("Downloading hub", "正在下载 Hub")
}

pub fn status_verifying_hub() -> SharedString {
    bi("Verifying hub", "正在校验 Hub")
}

pub fn status_extracting_hub() -> SharedString {
    bi("Extracting hub", "正在解压 Hub")
}

pub fn status_starting_hub() -> SharedString {
    bi("Starting lumen-hub", "正在启动 lumen-hub")
}

pub fn bootstrap_missing(path: &str) -> SharedString {
    bi(
        &format!("Bootstrap file: {path} (missing)"),
        &format!("引导文件：{path}（缺失）"),
    )
}

pub fn profile_summary(preset: &str, backend: &str, profile: &str) -> SharedString {
    bi(
        &format!("Preset: {preset} | Backend: {backend} | Profile: {profile}"),
        &format!("预设：{preset} | 后端：{backend} | 配置：{profile}"),
    )
}

pub fn profile_only(profile: &str) -> SharedString {
    bi(
        &format!("Profile: {profile}"),
        &format!("配置：{profile}"),
    )
}

pub fn review_region_other() -> SharedString {
    bi("Region: International (Hugging Face)", "区域：国际（Hugging Face）")
}

pub fn review_region_cn() -> SharedString {
    bi("Region: China (hf-mirror.com)", "区域：中国（hf-mirror.com）")
}

pub fn review_preset(preset: &str) -> SharedString {
    bi(
        &format!("Preset: {preset}"),
        &format!("预设：{preset}"),
    )
}

pub fn review_backend(name: &str, profile: &str) -> SharedString {
    bi(
        &format!("Backend: {name} ({profile})"),
        &format!("后端：{name}（{profile}）"),
    )
}

pub fn review_path(label_en: &str, label_zh: &str, path: &str) -> SharedString {
    bi(
        &format!("{label_en}: {path}"),
        &format!("{label_zh}：{path}"),
    )
}

pub fn picker_cache_dir() -> &'static str {
    "Select cache directory / 选择缓存目录"
}

pub fn picker_config_path() -> &'static str {
    "Select config file / 选择配置文件"
}

pub fn picker_bootstrap_path() -> &'static str {
    "Select bootstrap file / 选择引导文件"
}

pub fn log_ready() -> SharedString {
    bi("Ready.", "就绪。")
}

pub fn warning_ram(preset: &str, min_gb: u64, detected_gb: f64) -> String {
    format!(
        "`{preset}` recommends at least {min_gb} GB RAM; detected {detected_gb:.1} GB. / \
         `{preset}` 建议至少 {min_gb} GB 内存；检测到 {detected_gb:.1} GB。"
    )
}

pub fn warning_disk(cache: &str, preset: &str, free_gb: f64, min_gb: u64) -> String {
    format!(
        "`{cache}` has {free_gb:.1} GB free; `{preset}` recommends at least {min_gb} GB. / \
         `{cache}` 剩余 {free_gb:.1} GB；`{preset}` 建议至少 {min_gb} GB。"
    )
}

pub fn warning_backend_unavailable() -> String {
    bi(
        "Selected backend is unavailable.",
        "所选后端不可用。",
    )
    .to_string()
}
