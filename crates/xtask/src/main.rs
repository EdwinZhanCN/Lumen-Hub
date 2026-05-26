use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use sha2::{Digest, Sha256};

const OPENVINO_WHEEL_URL: &str = "https://files.pythonhosted.org/packages/99/16/69ca742f0b65c40d4de3ff44bb6abc23c47b23e932bc901116176ae69922/onnxruntime_openvino-1.24.1-cp311-cp311-manylinux_2_28_x86_64.whl";
const OPENVINO_WHEEL_SHA256: &str =
    "3007c803634cc69c6d52af1dea7ce729d9bb62b9a11070fd2f959119199007a8";
const OPENVINO_WHEEL_FILE: &str =
    "onnxruntime_openvino-1.24.1-cp311-cp311-manylinux_2_28_x86_64.whl";
const ORT_VERSION: &str = "1.24.1";
const ORT_CPU_LINUX_AARCH64_WHEEL_URL: &str = "https://files.pythonhosted.org/packages/7b/61/b3305c39144e19dbe8791802076b29b4b592b09de03d0e340c1314bfd408/onnxruntime-1.24.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
const ORT_CPU_LINUX_AARCH64_WHEEL_SHA256: &str =
    "86bc43e922b1f581b3de26a3dc402149c70e5542fceb5bec6b3a85542dbeb164";
const ORT_CPU_LINUX_AARCH64_WHEEL_FILE: &str =
    "onnxruntime-1.24.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl";
const ORT_GPU_LINUX_X64_WHEEL_URL: &str = "https://files.pythonhosted.org/packages/ca/c7/07d06175f1124fc89e8b7da30d70eb8e0e1400d90961ae1cbea9da69e69b/onnxruntime_gpu-1.24.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
const ORT_GPU_LINUX_X64_WHEEL_SHA256: &str =
    "ac4bfc90c376516b13d709764ab257e4e3d78639bf6a2ccfc826e9db4a5c7ddf";
const ORT_GPU_LINUX_X64_WHEEL_FILE: &str =
    "onnxruntime_gpu-1.24.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
const MNN_PREBUILT_VERSION: &str = "3.5.0-lumnn.dyn.2";
const DEFAULT_RELEASE_VERSION: &str = "0.1.0-beta.10";
const DEFAULT_RELEASE_BASE_URL: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download";

#[derive(Clone, Copy)]
enum OrtBundle {
    CpuLinuxAarch64,
    GpuLinuxX64,
}

struct OrtBundleSpec {
    cache_key: &'static str,
    package: &'static str,
    wheel_file: &'static str,
    wheel_url: &'static str,
    wheel_sha256: &'static str,
    platform: &'static str,
}

impl OrtBundle {
    fn spec(self) -> OrtBundleSpec {
        match self {
            OrtBundle::CpuLinuxAarch64 => OrtBundleSpec {
                cache_key: "cpu-linux-aarch64",
                package: "onnxruntime",
                wheel_file: ORT_CPU_LINUX_AARCH64_WHEEL_FILE,
                wheel_url: ORT_CPU_LINUX_AARCH64_WHEEL_URL,
                wheel_sha256: ORT_CPU_LINUX_AARCH64_WHEEL_SHA256,
                platform: "Linux aarch64, manylinux_2_27 / glibc 2.27+",
            },
            OrtBundle::GpuLinuxX64 => OrtBundleSpec {
                cache_key: "gpu-linux-x64",
                package: "onnxruntime-gpu",
                wheel_file: ORT_GPU_LINUX_X64_WHEEL_FILE,
                wheel_url: ORT_GPU_LINUX_X64_WHEEL_URL,
                wheel_sha256: ORT_GPU_LINUX_X64_WHEEL_SHA256,
                platform: "Linux x64 CUDA, manylinux_2_27 / glibc 2.27+",
            },
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("dist") => dist(args.collect()),
        Some("release-metadata") => release_metadata(args.collect()),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(format!("unknown command `{other}`")),
    }
}

fn print_help() {
    println!(
        "Usage:\n  cargo xtask dist --profile <profile>\n  cargo xtask release-metadata [--assets-dir <dir>]\n\nProfiles:\n  {}",
        PROFILES
            .iter()
            .map(|profile| profile.name)
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

fn dist(args: Vec<String>) -> Result<(), String> {
    let profile_name = parse_profile_name(args)?;
    let profile = PROFILES
        .iter()
        .find(|profile| profile.name == profile_name)
        .ok_or_else(|| {
            format!(
                "unknown dist profile `{profile_name}`; expected one of: {}",
                PROFILES
                    .iter()
                    .map(|profile| profile.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let root = workspace_root()?;
    let archive_dir = root.join("dist").join(profile.archive_name);

    if archive_dir.exists() {
        fs::remove_dir_all(&archive_dir).map_err(|err| {
            format!(
                "failed to remove existing dist directory `{}`: {err}",
                archive_dir.display()
            )
        })?;
    }
    fs::create_dir_all(archive_dir.join("bin")).map_err(|err| {
        format!(
            "failed to create dist bin directory `{}`: {err}",
            archive_dir.join("bin").display()
        )
    })?;
    fs::create_dir_all(archive_dir.join("lib")).map_err(|err| {
        format!(
            "failed to create dist lib directory `{}`: {err}",
            archive_dir.join("lib").display()
        )
    })?;

    build_profile(profile, &root)?;
    copy_binary(profile, &root, &archive_dir)?;
    copy_cli_binary(profile, &root, &archive_dir)?;
    copy_warmup_assets(&root, &archive_dir)?;
    prepare_licenses(&root, &archive_dir)?;

    if profile.openvino_bundle {
        download_openvino_bundle(&root, &archive_dir)?;
    }

    if let Some(bundle) = profile.ort_bundle {
        download_ort_bundle(&root, &archive_dir, bundle)?;
    }

    if profile.mnn_bundle {
        bundle_mnn_libraries(&root, profile, &archive_dir)?;
    }

    write_readme(profile, &archive_dir)?;
    write_checksums(&archive_dir)?;
    let archive_path = package_archive(profile, &root, &archive_dir)?;
    println!("dist artifact prepared at {}", archive_dir.display());
    println!("release archive prepared at {}", archive_path.display());
    Ok(())
}

fn release_metadata(args: Vec<String>) -> Result<(), String> {
    let root = workspace_root()?;
    let assets_dir = parse_assets_dir(args, &root)?;
    fs::create_dir_all(&assets_dir).map_err(|err| {
        format!(
            "failed to create release assets directory `{}`: {err}",
            assets_dir.display()
        )
    })?;

    let base_url = release_base_url();
    write_release_manifest(&assets_dir, &base_url)?;
    write_top_level_checksums(&assets_dir)?;
    println!("release metadata prepared at {}", assets_dir.display());
    Ok(())
}

fn parse_profile_name(args: Vec<String>) -> Result<String, String> {
    match args.as_slice() {
        [flag, value] if flag == "--profile" => Ok(value.clone()),
        [value] => Ok(value.clone()),
        _ => Err("expected `dist --profile <profile>`".to_owned()),
    }
}

fn parse_assets_dir(args: Vec<String>, root: &Path) -> Result<PathBuf, String> {
    match args.as_slice() {
        [] => Ok(root.join("release-assets")),
        [flag, value] if flag == "--assets-dir" => Ok(PathBuf::from(value)),
        [value] => Ok(PathBuf::from(value)),
        _ => Err("expected `release-metadata [--assets-dir <dir>]`".to_owned()),
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output()
        .map_err(|err| format!("failed to run `cargo locate-project`: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo locate-project` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("`cargo locate-project` output was not UTF-8: {err}"))?;
    let manifest_path = PathBuf::from(stdout.trim());
    manifest_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "workspace manifest path has no parent".to_owned())
}

fn build_profile(profile: &DistProfile, root: &Path) -> Result<(), String> {
    let features = profile.features.join(",");
    let mut hub_command = Command::new("cargo");
    hub_command.current_dir(root).args([
        "build",
        "-p",
        "lumen-hub",
        "--release",
        "--no-default-features",
        "--target",
        profile.target,
        "--features",
        &features,
    ]);
    configure_profile_build_env(profile, &mut hub_command);
    let hub_status = hub_command
        .status()
        .map_err(|err| format!("failed to spawn lumen-hub cargo build: {err}"))?;
    if !hub_status.success() {
        return Err(format!(
            "lumen-hub cargo build failed for profile `{}` with status {hub_status}",
            profile.name
        ));
    }

    let mut cli_command = Command::new("cargo");
    cli_command.current_dir(root).args([
        "build",
        "-p",
        "lumen-cli",
        "--release",
        "--target",
        profile.target,
    ]);
    configure_profile_build_env(profile, &mut cli_command);
    let cli_status = cli_command
        .status()
        .map_err(|err| format!("failed to spawn lumen-cli cargo build: {err}"))?;
    if !cli_status.success() {
        return Err(format!(
            "lumen-cli cargo build failed for profile `{}` with status {cli_status}",
            profile.name
        ));
    }

    Ok(())
}

fn configure_profile_build_env(profile: &DistProfile, command: &mut Command) {
    command.env("RUSTC_BOOTSTRAP", "1");
    if profile.target.contains("windows") {
        command.env("RUSTFLAGS", rustflags_without_static_crt());
        command.env("CFLAGS", "/MD");
        command.env("CXXFLAGS", "/MD");
    }
}

fn rustflags_without_static_crt() -> String {
    let mut flags = env::var("RUSTFLAGS").unwrap_or_default();
    if !flags
        .split_whitespace()
        .any(|flag| flag == "-C" || flag.contains("target-feature=-crt-static"))
    {
        if !flags.is_empty() {
            flags.push(' ');
        }
        flags.push_str("-C target-feature=-crt-static");
        return flags;
    }
    if !flags.contains("target-feature=-crt-static") {
        if !flags.is_empty() {
            flags.push(' ');
        }
        flags.push_str("-C target-feature=-crt-static");
    }
    flags
}

fn copy_binary(profile: &DistProfile, root: &Path, archive_dir: &Path) -> Result<(), String> {
    let exe_name = if profile.target.contains("windows") {
        "lumen-hub.exe"
    } else {
        "lumen-hub"
    };
    let src = root
        .join("target")
        .join(profile.target)
        .join("release")
        .join(exe_name);

    if needs_runtime_launcher(profile) && !profile.target.contains("windows") {
        let bin_dst = archive_dir.join("bin").join("lumen-hub-bin");
        fs::copy(&src, &bin_dst).map_err(|err| {
            format!(
                "failed to copy binary `{}` to `{}`: {err}",
                src.display(),
                bin_dst.display()
            )
        })?;
        make_executable(&bin_dst)?;
        write_unix_launcher(&archive_dir.join("bin").join("lumen-hub"), profile)?;
        return Ok(());
    }

    let dst = archive_dir.join("bin").join(exe_name);
    fs::copy(&src, &dst).map_err(|err| {
        format!(
            "failed to copy binary `{}` to `{}`: {err}",
            src.display(),
            dst.display()
        )
    })?;
    make_executable(&dst)?;
    Ok(())
}

fn copy_cli_binary(profile: &DistProfile, root: &Path, archive_dir: &Path) -> Result<(), String> {
    let exe_name = if profile.target.contains("windows") {
        "lumen-cli.exe"
    } else {
        "lumen-cli"
    };
    let src = root
        .join("target")
        .join(profile.target)
        .join("release")
        .join(exe_name);
    let dst = archive_dir.join("bin").join(exe_name);

    fs::copy(&src, &dst).map_err(|err| {
        format!(
            "failed to copy CLI binary `{}` to `{}`: {err}",
            src.display(),
            dst.display()
        )
    })?;
    make_executable(&dst)?;
    Ok(())
}

fn copy_warmup_assets(root: &Path, archive_dir: &Path) -> Result<(), String> {
    let src = root.join("crates").join("lumen-hub").join("warmup");
    let dst = archive_dir.join("warmup");
    copy_dir_filtered(&src, &dst).map_err(|err| {
        format!(
            "failed to copy warmup assets from `{}`: {err}",
            src.display()
        )
    })
}

fn copy_dir_filtered(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_str().is_some_and(|name| name.starts_with('.')) {
            continue;
        }

        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if entry.file_type()?.is_dir() {
            copy_dir_filtered(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

fn needs_runtime_launcher(profile: &DistProfile) -> bool {
    profile.openvino_bundle
        || profile.ort_bundle.is_some()
        || profile.mnn_bundle
        || profile.jetson_dynamic_ort
}

fn write_unix_launcher(path: &Path, profile: &DistProfile) -> Result<(), String> {
    let mut lines = vec![
        "#!/usr/bin/env bash".to_owned(),
        "set -euo pipefail".to_owned(),
        r#"APP_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)""#.to_owned(),
    ];

    if profile.openvino_bundle || profile.ort_bundle.is_some() {
        lines.push(r#"export LUMNN_ORT_DYLIB_PATH="$APP_HOME/lib/libonnxruntime.so""#.to_owned());
    }

    if profile.jetson_dynamic_ort {
        lines.push(r#"if [[ -z "${LUMNN_ORT_DYLIB_PATH:-}" ]]; then"#.to_owned());
        lines.push(r#"  for candidate in "$APP_HOME/lib/libonnxruntime.so" /usr/local/lib/libonnxruntime.so /usr/lib/aarch64-linux-gnu/libonnxruntime.so /usr/local/lib/python*/dist-packages/onnxruntime/capi/libonnxruntime.so /usr/lib/python*/dist-packages/onnxruntime/capi/libonnxruntime.so "${VIRTUAL_ENV:-}"/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so "$HOME"/.local/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so; do"#.to_owned());
        lines.push(r#"    if [[ -f "$candidate" ]]; then"#.to_owned());
        lines.push(r#"      export LUMNN_ORT_DYLIB_PATH="$candidate""#.to_owned());
        lines.push(r#"      break"#.to_owned());
        lines.push(r#"    fi"#.to_owned());
        lines.push(r#"  done"#.to_owned());
        lines.push(r#"fi"#.to_owned());
        lines.push(r#"if [[ -n "${LUMNN_ORT_DYLIB_PATH:-}" ]]; then"#.to_owned());
        lines.push(r#"  ORT_HOME="$(cd "$(dirname "$LUMNN_ORT_DYLIB_PATH")" && pwd)""#.to_owned());
        lines.push(r#"  export LD_LIBRARY_PATH="$ORT_HOME:${LD_LIBRARY_PATH:-}""#.to_owned());
        lines.push(r#"fi"#.to_owned());
    }

    if profile.openvino_bundle || profile.ort_bundle.is_some() || profile.mnn_bundle {
        if profile.target.contains("apple") {
            lines.push(
                r#"export DYLD_LIBRARY_PATH="$APP_HOME/lib:${DYLD_LIBRARY_PATH:-}""#.to_owned(),
            );
        } else {
            lines.push(r#"export LD_LIBRARY_PATH="$APP_HOME/lib:${LD_LIBRARY_PATH:-}""#.to_owned());
        }
    }

    if profile.jetson_dynamic_ort {
        lines.push(r#"export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}""#.to_owned());
    }

    lines.push(r#"exec "$APP_HOME/bin/lumen-hub-bin" "$@""#.to_owned());

    fs::write(path, lines.join("\n") + "\n").map_err(|err| {
        format!(
            "failed to write runtime launcher `{}`: {err}",
            path.display()
        )
    })?;
    make_executable(path)
}

fn mnn_prebuilt_suffix(target: &str) -> Result<&'static str, String> {
    match target {
        "aarch64-apple-darwin" | "x86_64-apple-darwin" => Ok("macos-universal"),
        "x86_64-unknown-linux-gnu" => Ok("linux-x86_64"),
        "aarch64-unknown-linux-gnu" => Ok("linux-aarch64"),
        "x86_64-pc-windows-msvc" => Ok("windows-x86_64"),
        other => Err(format!(
            "no MNN prebuilt mapping for target `{other}`; expected a supported desktop target"
        )),
    }
}

fn mnn_prebuilt_lib_dir(root: &Path, target: &str) -> Result<PathBuf, String> {
    let suffix = mnn_prebuilt_suffix(target)?;
    let asset = format!("mnn-{MNN_PREBUILT_VERSION}-{suffix}");
    let lib_dir = root
        .join("crates")
        .join("lumnn-mnn-sys")
        .join("3rd_party")
        .join("prebuilt")
        .join(asset)
        .join("lib");
    if !lib_dir.is_dir() {
        return Err(format!(
            "MNN prebuilt lib directory not found at `{}`; build lumen-hub for `{target}` first so lumnn-mnn-sys can download prebuilts",
            lib_dir.display()
        ));
    }
    Ok(lib_dir)
}

fn bundle_mnn_libraries(
    root: &Path,
    profile: &DistProfile,
    archive_dir: &Path,
) -> Result<(), String> {
    let src_lib = mnn_prebuilt_lib_dir(root, profile.target)?;
    if profile.target.contains("windows") {
        copy_mnn_windows_runtime(&src_lib, &archive_dir.join("bin"))?;
    } else {
        copy_mnn_unix_runtime(&src_lib, &archive_dir.join("lib"))?;
    }
    write_mnn_notice(archive_dir, profile)
}

fn is_mnn_runtime_library(name: &str) -> bool {
    name == "libMNN.dylib"
        || name == "libMNN.so"
        || name == "MNN.dll"
        || name.starts_with("libllm.")
        || name.starts_with("libMNN_Express.")
        || name.starts_with("llm.")
        || name.starts_with("MNN_Express.")
}

fn copy_mnn_unix_runtime(src_lib: &Path, dst_lib: &Path) -> Result<(), String> {
    fs::create_dir_all(dst_lib).map_err(|err| {
        format!(
            "failed to create MNN lib directory `{}`: {err}",
            dst_lib.display()
        )
    })?;

    let mut copied_primary = false;
    for entry in fs::read_dir(src_lib)
        .map_err(|err| format!("failed to read `{}`: {err}", src_lib.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read MNN lib entry: {err}"))?;
        if !entry
            .file_type()
            .map_err(|err| format!("failed to read file type: {err}"))?
            .is_file()
        {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !is_mnn_runtime_library(&name) {
            continue;
        }

        let dst = dst_lib.join(name.as_ref());
        fs::copy(entry.path(), &dst).map_err(|err| {
            format!(
                "failed to copy MNN runtime `{}` to `{}`: {err}",
                entry.path().display(),
                dst.display()
            )
        })?;
        make_executable(&dst)?;
        if name == "libMNN.dylib" || name == "libMNN.so" {
            copied_primary = true;
        }
    }

    if !copied_primary {
        return Err(format!(
            "MNN prebuilt at `{}` did not contain libMNN.dylib or libMNN.so",
            src_lib.display()
        ));
    }

    Ok(())
}

fn copy_mnn_windows_runtime(src_lib: &Path, dst_bin: &Path) -> Result<(), String> {
    fs::create_dir_all(dst_bin).map_err(|err| {
        format!(
            "failed to create MNN bin directory `{}`: {err}",
            dst_bin.display()
        )
    })?;

    let mut copied_primary = false;
    for entry in fs::read_dir(src_lib)
        .map_err(|err| format!("failed to read `{}`: {err}", src_lib.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read MNN lib entry: {err}"))?;
        if !entry
            .file_type()
            .map_err(|err| format!("failed to read file type: {err}"))?
            .is_file()
        {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".dll") || !is_mnn_runtime_library(&name) {
            continue;
        }

        let dst = dst_bin.join(name.as_ref());
        fs::copy(entry.path(), &dst).map_err(|err| {
            format!(
                "failed to copy MNN runtime `{}` to `{}`: {err}",
                entry.path().display(),
                dst.display()
            )
        })?;
        if name == "MNN.dll" {
            copied_primary = true;
        }
    }

    if !copied_primary {
        return Err(format!(
            "MNN prebuilt at `{}` did not contain MNN.dll",
            src_lib.display()
        ));
    }

    Ok(())
}

fn write_mnn_notice(archive_dir: &Path, profile: &DistProfile) -> Result<(), String> {
    let runtime_location = if profile.target.contains("windows") {
        "`bin/*.dll`"
    } else {
        "`lib/`"
    };
    let launcher_note = if needs_runtime_launcher(profile) && !profile.target.contains("windows") {
        "Use `bin/lumen-hub`, which sets the bundled library path before launching the real binary.\n"
    } else {
        ""
    };
    let notice = format!(
        "This package bundles dynamic MNN `{MNN_PREBUILT_VERSION}` prebuilts for `{}`.\nRuntime libraries are packaged under {runtime_location}.\n{launcher_note}",
        profile.target
    );
    let path = archive_dir.join("licenses").join("mnn").join("README.md");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create MNN licenses directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    fs::write(&path, notice).map_err(|err| {
        format!(
            "failed to write MNN package notice `{}`: {err}",
            path.display()
        )
    })
}

fn download_ort_bundle(root: &Path, archive_dir: &Path, bundle: OrtBundle) -> Result<(), String> {
    let spec = bundle.spec();
    let cache_dir = root
        .join("target")
        .join("xtask-cache")
        .join("onnxruntime")
        .join(spec.cache_key);
    fs::create_dir_all(&cache_dir).map_err(|err| {
        format!(
            "failed to create ONNX Runtime wheel cache `{}`: {err}",
            cache_dir.display()
        )
    })?;
    let wheel_path = cache_dir.join(spec.wheel_file);
    ensure_ort_wheel(&wheel_path, &spec)?;
    extract_ort_wheel(&wheel_path, archive_dir)?;
    write_ort_notice(archive_dir, &spec)
}

fn ensure_ort_wheel(wheel_path: &Path, spec: &OrtBundleSpec) -> Result<(), String> {
    if wheel_path.is_file() {
        let digest = sha256_file(wheel_path)?;
        if digest == spec.wheel_sha256 {
            return Ok(());
        }
    }

    let tmp = wheel_path.with_extension("whl.tmp");
    if tmp.exists() {
        fs::remove_file(&tmp)
            .map_err(|err| format!("failed to remove stale `{}`: {err}", tmp.display()))?;
    }

    let mut response = ureq::get(spec.wheel_url)
        .call()
        .map_err(|err| format!("failed to download ONNX Runtime wheel: {err}"))?;
    let mut file = fs::File::create(&tmp)
        .map_err(|err| format!("failed to create `{}`: {err}", tmp.display()))?;
    io::copy(&mut response.body_mut().as_reader(), &mut file)
        .map_err(|err| format!("failed to write `{}`: {err}", tmp.display()))?;
    file.flush()
        .map_err(|err| format!("failed to flush `{}`: {err}", tmp.display()))?;

    let digest = sha256_file(&tmp)?;
    if digest != spec.wheel_sha256 {
        return Err(format!(
            "ONNX Runtime wheel SHA256 mismatch: expected {}, got {digest}",
            spec.wheel_sha256
        ));
    }

    fs::rename(&tmp, wheel_path).map_err(|err| {
        format!(
            "failed to move `{}` to `{}`: {err}",
            tmp.display(),
            wheel_path.display()
        )
    })
}

fn extract_ort_wheel(wheel_path: &Path, archive_dir: &Path) -> Result<(), String> {
    let file = fs::File::open(wheel_path)
        .map_err(|err| format!("failed to open `{}`: {err}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| {
        format!(
            "failed to read ONNX Runtime wheel `{}`: {err}",
            wheel_path.display()
        )
    })?;
    let lib_dir = archive_dir.join("lib");
    let license_dir = archive_dir.join("licenses").join("onnxruntime");
    fs::create_dir_all(&license_dir).map_err(|err| {
        format!(
            "failed to create ONNX Runtime licenses directory `{}`: {err}",
            license_dir.display()
        )
    })?;

    let mut extracted_libs = Vec::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|err| format!("failed to read wheel entry {index}: {err}"))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_owned();

        if is_shared_object(&name) {
            let base_name = archive_base_name(&name).ok_or_else(|| {
                format!("ONNX Runtime wheel entry `{name}` did not have a file name")
            })?;
            let target = lib_dir.join(base_name);
            let mut output = fs::File::create(&target)
                .map_err(|err| format!("failed to create `{}`: {err}", target.display()))?;
            io::copy(&mut entry, &mut output)
                .map_err(|err| format!("failed to extract `{name}`: {err}"))?;
            make_executable(&target)?;
            extracted_libs.push(target);
        } else if is_license_entry(&name) {
            let target = license_dir.join(sanitize_archive_name(&name));
            let mut output = fs::File::create(&target)
                .map_err(|err| format!("failed to create `{}`: {err}", target.display()))?;
            io::copy(&mut entry, &mut output)
                .map_err(|err| format!("failed to extract `{name}`: {err}"))?;
        }
    }

    ensure_unversioned_onnxruntime(&lib_dir, &extracted_libs, "ONNX Runtime wheel")
}

fn ensure_unversioned_onnxruntime(
    lib_dir: &Path,
    extracted_libs: &[PathBuf],
    source_name: &str,
) -> Result<(), String> {
    let exact_ort = lib_dir.join("libonnxruntime.so");
    if exact_ort.is_file() {
        return Ok(());
    }

    let versioned = extracted_libs
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libonnxruntime.so."))
        })
        .ok_or_else(|| {
            format!("{source_name} did not contain libonnxruntime.so or libonnxruntime.so.*")
        })?;
    fs::copy(versioned, &exact_ort).map_err(|err| {
        format!(
            "failed to create `{}` from `{}`: {err}",
            exact_ort.display(),
            versioned.display()
        )
    })?;
    make_executable(&exact_ort)
}

fn write_ort_notice(archive_dir: &Path, spec: &OrtBundleSpec) -> Result<(), String> {
    let notice = format!(
        "This package bundles ONNX Runtime from PyPI package `{}=={ORT_VERSION}`.\nWheel: `{}`\nSHA256: {}\nPlatform: {}.\n",
        spec.package, spec.wheel_file, spec.wheel_sha256, spec.platform
    );
    let path = archive_dir
        .join("licenses")
        .join("onnxruntime")
        .join("README.md");
    fs::write(&path, notice).map_err(|err| {
        format!(
            "failed to write ONNX Runtime package notice `{}`: {err}",
            path.display()
        )
    })
}

fn download_openvino_bundle(root: &Path, archive_dir: &Path) -> Result<(), String> {
    let cache_dir = root.join("target").join("xtask-cache").join("openvino");
    fs::create_dir_all(&cache_dir).map_err(|err| {
        format!(
            "failed to create OpenVINO wheel cache `{}`: {err}",
            cache_dir.display()
        )
    })?;
    let wheel_path = cache_dir.join(OPENVINO_WHEEL_FILE);
    ensure_openvino_wheel(&wheel_path)?;
    extract_openvino_wheel(&wheel_path, archive_dir)?;
    write_openvino_notice(archive_dir)
}

fn ensure_openvino_wheel(wheel_path: &Path) -> Result<(), String> {
    if wheel_path.is_file() {
        let digest = sha256_file(wheel_path)?;
        if digest == OPENVINO_WHEEL_SHA256 {
            return Ok(());
        }
    }

    let tmp = wheel_path.with_extension("whl.tmp");
    if tmp.exists() {
        fs::remove_file(&tmp)
            .map_err(|err| format!("failed to remove stale `{}`: {err}", tmp.display()))?;
    }

    let mut response = ureq::get(OPENVINO_WHEEL_URL)
        .call()
        .map_err(|err| format!("failed to download OpenVINO wheel: {err}"))?;
    let mut file = fs::File::create(&tmp)
        .map_err(|err| format!("failed to create `{}`: {err}", tmp.display()))?;
    io::copy(&mut response.body_mut().as_reader(), &mut file)
        .map_err(|err| format!("failed to write `{}`: {err}", tmp.display()))?;
    file.flush()
        .map_err(|err| format!("failed to flush `{}`: {err}", tmp.display()))?;

    let digest = sha256_file(&tmp)?;
    if digest != OPENVINO_WHEEL_SHA256 {
        return Err(format!(
            "OpenVINO wheel SHA256 mismatch: expected {OPENVINO_WHEEL_SHA256}, got {digest}"
        ));
    }

    fs::rename(&tmp, wheel_path).map_err(|err| {
        format!(
            "failed to move `{}` to `{}`: {err}",
            tmp.display(),
            wheel_path.display()
        )
    })
}

fn extract_openvino_wheel(wheel_path: &Path, archive_dir: &Path) -> Result<(), String> {
    let file = fs::File::open(wheel_path)
        .map_err(|err| format!("failed to open `{}`: {err}", wheel_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| {
        format!(
            "failed to read OpenVINO wheel `{}`: {err}",
            wheel_path.display()
        )
    })?;
    let lib_dir = archive_dir.join("lib");
    let license_dir = archive_dir.join("licenses").join("openvino");
    fs::create_dir_all(&license_dir).map_err(|err| {
        format!(
            "failed to create OpenVINO licenses directory `{}`: {err}",
            license_dir.display()
        )
    })?;

    let mut extracted_libs = Vec::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|err| format!("failed to read wheel entry {index}: {err}"))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_owned();

        if is_shared_object(&name) {
            let base_name = archive_base_name(&name)
                .ok_or_else(|| format!("OpenVINO wheel entry `{name}` did not have a file name"))?;
            let target = lib_dir.join(base_name);
            let mut output = fs::File::create(&target)
                .map_err(|err| format!("failed to create `{}`: {err}", target.display()))?;
            io::copy(&mut entry, &mut output)
                .map_err(|err| format!("failed to extract `{name}`: {err}"))?;
            make_executable(&target)?;
            extracted_libs.push(target);
        } else if is_license_entry(&name) {
            let target = license_dir.join(sanitize_archive_name(&name));
            let mut output = fs::File::create(&target)
                .map_err(|err| format!("failed to create `{}`: {err}", target.display()))?;
            io::copy(&mut entry, &mut output)
                .map_err(|err| format!("failed to extract `{name}`: {err}"))?;
        }
    }

    let exact_ort = lib_dir.join("libonnxruntime.so");
    if !exact_ort.is_file() {
        let versioned = extracted_libs
            .iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("libonnxruntime.so."))
            })
            .ok_or_else(|| {
                "OpenVINO wheel did not contain libonnxruntime.so or libonnxruntime.so.*"
                    .to_string()
            })?;
        fs::copy(versioned, &exact_ort).map_err(|err| {
            format!(
                "failed to create `{}` from `{}`: {err}",
                exact_ort.display(),
                versioned.display()
            )
        })?;
        make_executable(&exact_ort)?;
    }

    Ok(())
}

fn write_openvino_notice(archive_dir: &Path) -> Result<(), String> {
    let notice = format!(
        "This package bundles ONNX Runtime OpenVINO from PyPI package `onnxruntime-openvino==1.24.1`.\nWheel: `{OPENVINO_WHEEL_FILE}`\nSHA256: `{OPENVINO_WHEEL_SHA256}`\nPlatform: Linux x64, manylinux_2_28 / glibc 2.28+.\n"
    );
    let path = archive_dir
        .join("licenses")
        .join("openvino")
        .join("README.md");
    fs::write(&path, notice).map_err(|err| {
        format!(
            "failed to write OpenVINO package notice `{}`: {err}",
            path.display()
        )
    })
}

fn write_readme(profile: &DistProfile, archive_dir: &Path) -> Result<(), String> {
    let mut runtime_notes: Vec<String> = Vec::new();
    if profile.openvino_bundle {
        runtime_notes.push(
            "OpenVINO runtime libraries are bundled in `lib/`. `bin/lumen-hub` sets `LUMNN_ORT_DYLIB_PATH` and the bundled library path before launching the real binary.".to_owned(),
        );
    }
    if let Some(bundle) = profile.ort_bundle {
        let spec = bundle.spec();
        runtime_notes.push(format!(
            "ONNX Runtime dynamic libraries (`{}=={ORT_VERSION}`) are bundled in `lib/`. `bin/lumen-hub` sets `LUMNN_ORT_DYLIB_PATH` and the bundled library path before launching the real binary.",
            spec.package
        ));
    }
    if profile.mnn_bundle {
        let location = if profile.target.contains("windows") {
            "`bin/*.dll`"
        } else {
            "`lib/`"
        };
        runtime_notes.push(format!(
            "MNN dynamic runtime libraries (`{MNN_PREBUILT_VERSION}`) are bundled in {location}."
        ));
        if needs_runtime_launcher(profile) && !profile.target.contains("windows") {
            runtime_notes.push(
                "On Unix, use `bin/lumen-hub`; it configures the bundled library path before launching `bin/lumen-hub-bin`."
                    .to_owned(),
            );
        }
    }
    if profile.jetson_dynamic_ort {
        runtime_notes.push(
            "Jetson profile uses dynamic ONNX Runtime. Install a JetPack 6+ compatible onnxruntime-gpu package, or set `LUMNN_ORT_DYLIB_PATH` to its `libonnxruntime.so`.".to_owned(),
        );
        runtime_notes.push(
            "On Unix, use `bin/lumen-hub`; it discovers common Jetson/Python wheel runtime paths and configures `LD_LIBRARY_PATH` before launching `bin/lumen-hub-bin`."
                .to_owned(),
        );
    }
    let runtime_note = if runtime_notes.is_empty() {
        String::new()
    } else {
        format!("\n{}\n", runtime_notes.join("\n"))
    };
    let body = format!(
        "# {}\n\nProfile: `{}`\nTarget: `{}`\nFeatures: `{}`\n\nLayout:\n- `bin/`: executable launcher and binary\n- `lib/`: bundled runtime libraries, when needed\n- `licenses/`: license and third-party notices\n\nStart with:\n\n```bash\n./bin/lumen-hub --config /path/to/lumen-config.json\n```\n{runtime_note}",
        profile.archive_name,
        profile.name,
        profile.target,
        profile.features.join(",")
    );
    fs::write(archive_dir.join("README.md"), body)
        .map_err(|err| format!("failed to write README.md: {err}"))
}

fn prepare_licenses(root: &Path, archive_dir: &Path) -> Result<(), String> {
    let licenses_dir = archive_dir.join("licenses");
    fs::create_dir_all(&licenses_dir)
        .map_err(|err| format!("failed to create licenses directory: {err}"))?;

    let mut copied = false;
    for entry in
        fs::read_dir(root).map_err(|err| format!("failed to read workspace root: {err}"))?
    {
        let entry = entry.map_err(|err| format!("failed to read workspace entry: {err}"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let lower = name.to_ascii_lowercase();
        if lower.starts_with("license") || lower.starts_with("notice") {
            fs::copy(entry.path(), licenses_dir.join(name.as_ref()))
                .map_err(|err| format!("failed to copy `{name}` into licenses: {err}"))?;
            copied = true;
        }
    }

    if !copied {
        fs::write(
            licenses_dir.join("README.md"),
            "No workspace LICENSE/NOTICE file was present when this artifact was generated.\n",
        )
        .map_err(|err| format!("failed to write licenses README: {err}"))?;
    }

    Ok(())
}

fn write_checksums(archive_dir: &Path) -> Result<(), String> {
    let mut lines = Vec::new();
    let mut files = Vec::new();
    collect_files(archive_dir, archive_dir, &mut files)
        .map_err(|err| format!("failed to collect dist files: {err}"))?;
    files.sort();

    for relative in files {
        let path = archive_dir.join(&relative);
        let digest = sha256_file(&path)?;
        lines.push(format!("{digest}  {}", relative.display()));
    }

    fs::write(archive_dir.join("checksums.txt"), lines.join("\n") + "\n")
        .map_err(|err| format!("failed to write checksums.txt: {err}"))
}

fn package_archive(
    profile: &DistProfile,
    root: &Path,
    archive_dir: &Path,
) -> Result<PathBuf, String> {
    let dist_dir = root.join("dist");
    let archive_file = if profile.target.contains("windows") {
        dist_dir.join(format!("{}.zip", profile.archive_name))
    } else {
        dist_dir.join(format!("{}.tar.gz", profile.archive_name))
    };
    if archive_file.exists() {
        fs::remove_file(&archive_file).map_err(|err| {
            format!(
                "failed to remove existing archive `{}`: {err}",
                archive_file.display()
            )
        })?;
    }

    if profile.target.contains("windows") {
        zip_directory(archive_dir, &archive_file)?;
    } else {
        let status = Command::new("tar")
            .current_dir(&dist_dir)
            .args(["-czf"])
            .arg(&archive_file)
            .arg(profile.archive_name)
            .status()
            .map_err(|err| format!("failed to spawn `tar`: {err}"))?;
        if !status.success() {
            return Err(format!(
                "failed to create `{}` with tar",
                archive_file.display()
            ));
        }
    }
    Ok(archive_file)
}

fn zip_directory(src_dir: &Path, zip_path: &Path) -> Result<(), String> {
    let file = fs::File::create(zip_path)
        .map_err(|err| format!("failed to create `{}`: {err}", zip_path.display()))?;
    let mut writer = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o755);
    let mut files = Vec::new();
    collect_files(src_dir, src_dir, &mut files)
        .map_err(|err| format!("failed to collect files for zip: {err}"))?;
    files.sort();

    for relative in files {
        let path = src_dir.join(&relative);
        let archive_name = zip_archive_name(src_dir, &relative)?;
        writer
            .start_file(&archive_name, options)
            .map_err(|err| format!("failed to add `{}` to zip: {err}", relative.display()))?;
        let mut input = fs::File::open(&path)
            .map_err(|err| format!("failed to open `{}`: {err}", path.display()))?;
        io::copy(&mut input, &mut writer)
            .map_err(|err| format!("failed to write `{}` to zip: {err}", relative.display()))?;
    }
    writer
        .finish()
        .map_err(|err| format!("failed to finish `{}`: {err}", zip_path.display()))?;
    Ok(())
}

fn zip_archive_name(src_dir: &Path, relative: &Path) -> Result<String, String> {
    let root = src_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid archive directory `{}`", src_dir.display()))?;
    let mut parts = vec![root.to_owned()];
    for component in relative.components() {
        let std::path::Component::Normal(part) = component else {
            return Err(format!("invalid zip path `{}`", relative.display()));
        };
        let part = part
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 zip path `{}`", relative.display()))?;
        if part.contains('/') || part.contains('\\') {
            return Err(format!("invalid zip path component `{part}`"));
        }
        parts.push(part.to_owned());
    }
    Ok(parts.join("/"))
}

fn write_release_manifest(assets_dir: &Path, base_url: &str) -> Result<(), String> {
    let version = release_version();
    let mut hub = Vec::new();

    for profile in PROFILES {
        let file_name = if profile.target.contains("windows") {
            format!("{}.zip", profile.archive_name)
        } else {
            format!("{}.tar.gz", profile.archive_name)
        };
        let path = assets_dir.join(&file_name);
        if !path.is_file() {
            continue;
        }
        let sha256 = sha256_file(&path)?;
        hub.push(serde_json::json!({
            "profile": profile.name,
            "file_name": file_name,
            "url": format!("{}/{}", base_url.trim_end_matches('/'), file_name),
            "sha256": sha256,
        }));
    }

    let manifest = serde_json::json!({
        "version": version,
        "hub": hub,
    });
    let body = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to render release manifest: {err}"))?;
    fs::write(assets_dir.join("manifest.json"), body + "\n")
        .map_err(|err| format!("failed to write release manifest: {err}"))
}

fn write_top_level_checksums(assets_dir: &Path) -> Result<(), String> {
    let mut names = Vec::new();
    for entry in fs::read_dir(assets_dir)
        .map_err(|err| format!("failed to read `{}`: {err}", assets_dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read assets entry: {err}"))?;
        if !entry
            .file_type()
            .map_err(|err| format!("failed to read file type: {err}"))?
            .is_file()
        {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_internal_release_metadata(&name) {
            continue;
        }
        names.push(name);
    }
    names.sort();

    let mut lines = Vec::new();
    for name in names {
        let path = assets_dir.join(&name);
        let digest = sha256_file(&path)?;
        lines.push(format!("{digest}  {name}"));
    }
    fs::write(assets_dir.join("checksums.txt"), lines.join("\n") + "\n")
        .map_err(|err| format!("failed to write top-level checksums.txt: {err}"))
}

fn is_internal_release_metadata(name: &str) -> bool {
    name == "checksums.txt"
        || name == "dist-manifest.json"
        || name == "global-dist-manifest.json"
        || name == "plan-dist-manifest.json"
        || name.ends_with("-dist-manifest.json")
}

fn release_version() -> String {
    env::var("LUMEN_RELEASE_VERSION")
        .unwrap_or_else(|_| DEFAULT_RELEASE_VERSION.to_owned())
        .trim_start_matches('v')
        .to_owned()
}

fn release_base_url() -> String {
    env::var("LUMEN_RELEASE_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_RELEASE_BASE_URL.to_owned())
        .trim_end_matches('/')
        .to_owned()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("failed to open `{}`: {err}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read `{}`: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_files(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, files)?;
        } else if path.file_name().is_some_and(|name| name != "checksums.txt") {
            files.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
        }
    }
    Ok(())
}

fn is_shared_object(name: &str) -> bool {
    archive_base_name(name)
        .is_some_and(|base_name| base_name.ends_with(".so") || base_name.contains(".so."))
}

fn is_license_entry(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("license") || lower.contains("notice") || lower.contains("third-party")
}

fn archive_base_name(name: &str) -> Option<&str> {
    name.rsplit('/').next().filter(|part| !part.is_empty())
}

fn sanitize_archive_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)
        .map_err(|err| format!("failed to read permissions for `{}`: {err}", path.display()))?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    fs::set_permissions(path, permissions).map_err(|err| {
        format!(
            "failed to set executable bit on `{}`: {err}",
            path.display()
        )
    })
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

struct DistProfile {
    name: &'static str,
    archive_name: &'static str,
    target: &'static str,
    features: &'static [&'static str],
    openvino_bundle: bool,
    ort_bundle: Option<OrtBundle>,
    mnn_bundle: bool,
    jetson_dynamic_ort: bool,
}

const PROFILES: &[DistProfile] = &[
    DistProfile {
        name: "universal-cpu",
        archive_name: "lumen-hub-universal-cpu",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-universal-cpu"],
        openvino_bundle: false,
        ort_bundle: None,
        mnn_bundle: false,
        jetson_dynamic_ort: false,
    },
    DistProfile {
        name: "darwin-arm64",
        archive_name: "lumen-hub-darwin-arm64",
        target: "aarch64-apple-darwin",
        features: &["profile-darwin-arm64"],
        openvino_bundle: false,
        ort_bundle: None,
        mnn_bundle: true,
        jetson_dynamic_ort: false,
    },
    DistProfile {
        name: "linux-x64-cuda",
        archive_name: "lumen-hub-linux-x64-cuda",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-linux-x64-cuda"],
        openvino_bundle: false,
        ort_bundle: Some(OrtBundle::GpuLinuxX64),
        mnn_bundle: false,
        jetson_dynamic_ort: false,
    },
    DistProfile {
        name: "linux-arm64",
        archive_name: "lumen-hub-linux-arm64",
        target: "aarch64-unknown-linux-gnu",
        features: &["profile-linux-arm64"],
        openvino_bundle: false,
        ort_bundle: Some(OrtBundle::CpuLinuxAarch64),
        mnn_bundle: true,
        jetson_dynamic_ort: false,
    },
    DistProfile {
        name: "linux-arm64-jetson",
        archive_name: "lumen-hub-linux-arm64-jetson",
        target: "aarch64-unknown-linux-gnu",
        features: &["profile-linux-arm64-jetson"],
        openvino_bundle: false,
        ort_bundle: None,
        mnn_bundle: false,
        jetson_dynamic_ort: true,
    },
    DistProfile {
        name: "windows-x64-dml",
        archive_name: "lumen-hub-windows-x64-dml",
        target: "x86_64-pc-windows-msvc",
        features: &["profile-windows-x64-dml"],
        openvino_bundle: false,
        ort_bundle: None,
        mnn_bundle: false,
        jetson_dynamic_ort: false,
    },
    DistProfile {
        name: "linux-x64-openvino",
        archive_name: "lumen-hub-linux-x64-openvino",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-linux-x64-openvino"],
        openvino_bundle: true,
        ort_bundle: None,
        mnn_bundle: false,
        jetson_dynamic_ort: false,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_archive_names_use_forward_slashes() {
        let src_dir = Path::new("dist").join("lumen-cli-windows-x64");
        let relative = Path::new("bin").join("lumen-cli.exe");
        assert_eq!(
            zip_archive_name(&src_dir, &relative).unwrap(),
            "lumen-cli-windows-x64/bin/lumen-cli.exe"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn zip_archive_names_reject_backslash_components() {
        let src_dir = Path::new("dist").join("lumen-cli-windows-x64");
        assert!(zip_archive_name(&src_dir, Path::new(r"bin\lumen-cli.exe")).is_err());
    }

    #[test]
    fn mnn_prebuilt_suffix_maps_supported_targets() {
        assert_eq!(
            mnn_prebuilt_suffix("aarch64-apple-darwin").unwrap(),
            "macos-universal"
        );
        assert_eq!(
            mnn_prebuilt_suffix("x86_64-unknown-linux-gnu").unwrap(),
            "linux-x86_64"
        );
        assert_eq!(
            mnn_prebuilt_suffix("aarch64-unknown-linux-gnu").unwrap(),
            "linux-aarch64"
        );
        assert_eq!(
            mnn_prebuilt_suffix("x86_64-pc-windows-msvc").unwrap(),
            "windows-x86_64"
        );
    }

    #[test]
    fn top_level_checksums_ignore_cargo_dist_manifests() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!(
            "lumen-xtask-checksums-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("lumen-cli-installer.sh"), "installer").unwrap();
        fs::write(dir.join("aarch64-apple-darwin-dist-manifest.json"), "{}").unwrap();
        fs::write(dir.join("global-dist-manifest.json"), "{}").unwrap();

        write_top_level_checksums(&dir).unwrap();
        let checksums = fs::read_to_string(dir.join("checksums.txt")).unwrap();
        fs::remove_dir_all(&dir).unwrap();

        assert!(checksums.contains("lumen-cli-installer.sh"));
        assert!(!checksums.contains("dist-manifest"));
    }
}
