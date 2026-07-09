use std::sync::atomic::{AtomicU8, Ordering};

use gpui::SharedString;

/// The active UI locale. Backed by a process-global atomic so that background
/// threads (e.g. the launch observer) localize strings the same way the UI does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locale {
    Zh,
    En,
}

/// 0 = Zh, 1 = En.
static LOCALE: AtomicU8 = AtomicU8::new(0);

impl Locale {
    pub fn current() -> Self {
        match LOCALE.load(Ordering::Relaxed) {
            1 => Locale::En,
            _ => Locale::Zh,
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Locale::Zh => Locale::En,
            Locale::En => Locale::Zh,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Locale::Zh => "中文",
            Locale::En => "EN",
        }
    }
}

pub fn set_locale(locale: Locale) {
    LOCALE.store(
        match locale {
            Locale::Zh => 0,
            Locale::En => 1,
        },
        Ordering::Relaxed,
    );
}

/// Pick a string for the current locale.
pub fn t(zh: &'static str, en: &'static str) -> SharedString {
    match Locale::current() {
        Locale::Zh => zh.into(),
        Locale::En => en.into(),
    }
}

/// Pick a string for the current locale from owned values (for formatting).
fn ts(zh: String, en: String) -> SharedString {
    match Locale::current() {
        Locale::Zh => zh.into(),
        Locale::En => en.into(),
    }
}

pub fn app_subtitle() -> SharedString {
    t("本地模型运行时", "Local model runtime")
}

pub fn start() -> SharedString {
    t("启动", "Start")
}

pub fn stop() -> SharedString {
    t("停止", "Stop")
}

pub fn browse() -> SharedString {
    t("浏览…", "Browse…")
}

pub fn back() -> SharedString {
    t("上一步", "Back")
}

pub fn next() -> SharedString {
    t("下一步", "Next")
}

pub fn create_setup() -> SharedString {
    t("创建配置", "Create setup")
}

pub fn cache_label() -> SharedString {
    t("缓存目录", "Cache directory")
}

pub fn config_field_label() -> SharedString {
    t("配置文件", "Config file")
}

pub fn bootstrap_label() -> SharedString {
    t("引导文件", "Bootstrap file")
}

pub fn advanced_settings_label() -> SharedString {
    t("高级设置", "Advanced settings")
}

pub fn advanced_settings_hint() -> SharedString {
    t(
        "可选的配置文件与引导文件路径。大多数用户使用默认位置即可。",
        "Optional paths for config and bootstrap files. Defaults are recommended for most users.",
    )
}

pub fn step_progress(current: i32, total: i32) -> SharedString {
    match Locale::current() {
        Locale::Zh => format!("第 {current} / {total} 步").into(),
        Locale::En => format!("Step {current} of {total}").into(),
    }
}

pub fn welcome_title() -> SharedString {
    t("欢迎使用 Lumen Hub", "Welcome to Lumen Hub")
}

pub fn welcome_detail() -> SharedString {
    t(
        "本向导将在本机完成 Lumen Hub 的初始设置：选择下载区域、模型预设与计算后端，并写入本机配置与引导文件。",
        "This guided setup prepares Lumen Hub on your machine: choose a download region, a model preset and a compute backend, then write the local config and bootstrap files.",
    )
}

pub fn region_step_title() -> SharedString {
    t("选择下载区域", "Choose model download region")
}

pub fn region_step_detail() -> SharedString {
    t(
        "选择 Lumen Hub 下载模型权重的来源。",
        "Pick where Lumen Hub should download model weights.",
    )
}

pub fn preset_step_title() -> SharedString {
    t("选择预设", "Choose a preset")
}

pub fn preset_step_detail() -> SharedString {
    t(
        "预设包含服务组合以及针对本机的资源建议。",
        "Presets bundle services and resource guidance for your machine.",
    )
}

pub fn backend_step_title() -> SharedString {
    t("选择计算后端", "Choose compute backend")
}

pub fn backend_step_detail() -> SharedString {
    t(
        "选择本次安装使用的 GPU / CPU 后端。",
        "Select the GPU / CPU backend for this installation.",
    )
}

pub fn paths_step_title() -> SharedString {
    t("选择缓存目录", "Choose cache directory")
}

pub fn paths_step_detail() -> SharedString {
    t(
        "模型权重将在首次使用时下载并保存在此目录。",
        "Downloaded model weights are stored here on first use.",
    )
}

pub fn conflict_step_title() -> SharedString {
    t("发现已有配置", "Existing setup found")
}

pub fn conflict_step_detail() -> SharedString {
    t(
        "已存在配置或引导文件，请选择如何处理。",
        "Config or bootstrap files already exist. Choose how to continue.",
    )
}

pub fn review_step_title() -> SharedString {
    t("确认并创建", "Review and create")
}

pub fn review_step_detail() -> SharedString {
    t(
        "确认选择无误后，创建 Lumen 配置。",
        "Check your choices, then create the Lumen setup.",
    )
}

pub fn detected_system(os: &str, arch: &str, ram: &str) -> SharedString {
    format!("{os} · {arch} · {ram}").into()
}

pub fn ram_known(gb: f64) -> String {
    format!("{gb:.1} GB")
}

pub fn ram_unknown() -> &'static str {
    match Locale::current() {
        Locale::Zh => "内存未知",
        Locale::En => "RAM unknown",
    }
}

pub fn status_checking_setup() -> SharedString {
    t("正在检查配置", "Checking setup")
}

pub fn status_setup_required() -> SharedString {
    t("需要完成设置", "Setup required")
}

pub fn status_ready() -> SharedString {
    t("就绪", "Ready")
}

pub fn status_preparing() -> SharedString {
    t("准备中", "Preparing")
}

pub fn status_creating_setup() -> SharedString {
    t("正在创建配置", "Creating setup")
}

pub fn status_running() -> SharedString {
    t("运行中", "Running")
}

pub fn status_stopping() -> SharedString {
    t("正在停止", "Stopping")
}

pub fn status_stopped() -> SharedString {
    t("已停止", "Stopped")
}

pub fn status_error() -> SharedString {
    t("错误", "Error")
}

pub fn status_exited_with_error() -> SharedString {
    t("异常退出", "Exited with error")
}

pub fn status_fetching_manifest() -> SharedString {
    t("正在获取清单", "Fetching manifest")
}

pub fn status_downloading_hub() -> SharedString {
    t("正在下载 Hub", "Downloading Hub")
}

pub fn status_verifying_hub() -> SharedString {
    t("正在校验 Hub", "Verifying Hub")
}

pub fn status_extracting_hub() -> SharedString {
    t("正在解压 Hub", "Extracting Hub")
}

pub fn status_starting_hub() -> SharedString {
    t("正在启动 lumen-hub", "Starting lumen-hub")
}

pub fn bootstrap_missing(path: &str) -> SharedString {
    ts(
        format!("引导文件缺失：{path}"),
        format!("Bootstrap file missing: {path}"),
    )
}

pub fn profile_summary(preset: &str, backend: &str, profile: &str) -> SharedString {
    ts(
        format!("预设 {preset} · 后端 {backend} · {profile}"),
        format!("Preset: {preset} | Backend: {backend} | {profile}"),
    )
}

pub fn profile_only(profile: &str) -> SharedString {
    ts(
        format!("配置：{profile}"),
        format!("Profile: {profile}"),
    )
}

pub fn review_region_other() -> SharedString {
    t(
        "下载区域：国际（Hugging Face）",
        "Region: International (Hugging Face)",
    )
}

pub fn review_region_cn() -> SharedString {
    t(
        "下载区域：中国（hf-mirror.com）",
        "Region: China (hf-mirror.com)",
    )
}

pub fn review_preset(preset: &str) -> SharedString {
    ts(
        format!("预设：{preset}"),
        format!("Preset: {preset}"),
    )
}

pub fn review_backend(name: &str, profile: &str) -> SharedString {
    ts(
        format!("后端：{name}（{profile}）"),
        format!("Backend: {name} ({profile})"),
    )
}

pub fn review_path(label: SharedString, path: &str) -> SharedString {
    ts(
        format!("{label}：{path}"),
        format!("{label}: {path}"),
    )
}

pub fn region_other_title() -> SharedString {
    t("国际", "International")
}

pub fn region_other_detail() -> SharedString {
    t("Hugging Face 官方源", "Hugging Face")
}

pub fn region_cn_title() -> SharedString {
    t("中国", "China")
}

pub fn region_cn_detail() -> SharedString {
    t("hf-mirror.com 镜像", "hf-mirror.com mirror")
}

pub fn preset_minimal_title() -> SharedString {
    t("最小", "Minimal")
}

pub fn preset_minimal_detail() -> SharedString {
    t(
        "SigLIP + 人脸 · 内存 4 GB · GPU 2 GB",
        "SigLIP + face · RAM 4 GB · GPU 2 GB",
    )
}

pub fn preset_basic_title() -> SharedString {
    t("基础", "Basic")
}

pub fn preset_basic_detail() -> SharedString {
    t(
        "SigLIP、人脸、OCR、BioCLIP 核心 · 内存 6 GB",
        "SigLIP, face, OCR, BioCLIP core · RAM 6 GB",
    )
}

pub fn preset_brave_title() -> SharedString {
    t("激进", "Brave")
}

pub fn preset_brave_detail() -> SharedString {
    t(
        "更大 SigLIP 与完整 BioCLIP 目录 · 内存 8 GB",
        "Larger SigLIP and full BioCLIP catalog · RAM 8 GB",
    )
}

pub fn conflict_overwrite_title() -> SharedString {
    t("覆盖", "Overwrite")
}

pub fn conflict_overwrite_detail() -> SharedString {
    t(
        "覆盖默认的配置与引导文件。",
        "Replace the default config and bootstrap files.",
    )
}

pub fn conflict_next_title() -> SharedString {
    t("另存为新文件", "Create next to it")
}

pub fn conflict_next_detail() -> SharedString {
    t(
        "在现有文件旁另存新的配置与引导文件。",
        "Save new config and bootstrap files alongside the existing ones.",
    )
}

pub fn conflict_cancel_title() -> SharedString {
    t("取消设置", "Cancel setup")
}

pub fn conflict_cancel_detail() -> SharedString {
    t("保留现有文件，不进行设置。", "Keep existing files unchanged.")
}

pub fn picker_cache_dir() -> &'static str {
    match Locale::current() {
        Locale::Zh => "选择缓存目录",
        Locale::En => "Select cache directory",
    }
}

pub fn picker_config_path() -> &'static str {
    match Locale::current() {
        Locale::Zh => "选择配置文件",
        Locale::En => "Select config file",
    }
}

pub fn picker_bootstrap_path() -> &'static str {
    match Locale::current() {
        Locale::Zh => "选择引导文件",
        Locale::En => "Select bootstrap file",
    }
}

pub fn log_ready() -> SharedString {
    t("就绪。", "Ready.")
}

pub fn warning_ram(preset: &str, min_gb: u64, detected_gb: f64) -> String {
    match Locale::current() {
        Locale::Zh => format!(
            "预设 `{preset}` 建议至少 {min_gb} GB 内存；检测到 {detected_gb:.1} GB。"
        ),
        Locale::En => format!(
            "`{preset}` recommends at least {min_gb} GB RAM; detected {detected_gb:.1} GB."
        ),
    }
}

pub fn warning_disk(cache: &str, preset: &str, free_gb: f64, min_gb: u64) -> String {
    match Locale::current() {
        Locale::Zh => format!(
            "`{cache}` 剩余 {free_gb:.1} GB；预设 `{preset}` 建议至少 {min_gb} GB。"
        ),
        Locale::En => format!(
            "`{cache}` has {free_gb:.1} GB free; `{preset}` recommends at least {min_gb} GB."
        ),
    }
}

pub fn warning_backend_unavailable() -> String {
    t("所选后端不可用。", "Selected backend is unavailable.").to_string()
}

/// UI labels that are rendered inline in main.rs.
pub fn label_system() -> SharedString {
    t("系统环境", "System")
}

pub fn label_detected_env() -> SharedString {
    t("检测到的环境", "Detected environment")
}

pub fn label_what_next() -> SharedString {
    t("接下来会做什么", "What happens next")
}

pub fn label_what_next_detail() -> SharedString {
    t(
        "依次选择下载区域、模型预设与计算后端，最后写入本机的配置与引导文件。",
        "Choose a download region, a model preset and a compute backend, then write the local config and bootstrap files.",
    )
}

pub fn label_status() -> SharedString {
    t("状态", "Status")
}

pub fn label_config() -> SharedString {
    t("配置文件", "Config")
}

pub fn label_profile() -> SharedString {
    t("运行配置", "Profile")
}

pub fn label_runtime() -> SharedString {
    t("运行说明", "Runtime")
}

pub fn label_runtime_detail() -> SharedString {
    t(
        "Shell 会准备发布产物，按所选配置启动 lumen-hub，并在此处显示标准输出与错误日志。",
        "The shell prepares the release artifact, starts lumen-hub with the selected config, and shows stdout/stderr here.",
    )
}

pub fn label_log() -> SharedString {
    t("日志", "Log")
}

pub fn log_line_count(n: usize) -> SharedString {
    match Locale::current() {
        Locale::Zh => format!("{n} 行").into(),
        Locale::En => format!("{n} lines").into(),
    }
}

pub fn label_warning() -> SharedString {
    t("警告", "Warning")
}

pub fn block_setup_cancelled() -> SharedString {
    t("已取消设置。", "Setup cancelled.")
}

pub fn block_paths_required() -> SharedString {
    t("请填写所有存储路径。", "All storage paths are required.")
}

pub fn block_complete_step() -> SharedString {
    t("请先完成当前步骤。", "Complete this step first.")
}

pub fn block_setup_creation_cancelled() -> SharedString {
    t("已取消创建设置", "Setup creation was cancelled")
}

pub fn backend_unavailable_label() -> SharedString {
    t("后端：不可用", "Backend: unavailable")
}
