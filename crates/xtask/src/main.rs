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
const DEFAULT_RELEASE_VERSION: &str = "0.1.0-beta.2";
const DEFAULT_RELEASE_BASE_URL: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download";
const DEFAULT_MANIFEST_URL: &str =
    "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download/manifest.json";

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
        Some("cli-dist") => cli_dist(args.collect()),
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
        "Usage:\n  cargo xtask dist --profile <profile>\n  cargo xtask cli-dist --target <target>\n  cargo xtask release-metadata [--assets-dir <dir>]\n\nProfiles:\n  {}\n\nCLI targets:\n  {}",
        PROFILES
            .iter()
            .map(|profile| profile.name)
            .collect::<Vec<_>>()
            .join("\n  "),
        CLI_TARGETS
            .iter()
            .map(|target| target.name)
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

    write_readme(profile, &archive_dir)?;
    write_checksums(&archive_dir)?;
    let archive_path = package_archive(profile, &root, &archive_dir)?;
    println!("dist artifact prepared at {}", archive_dir.display());
    println!("release archive prepared at {}", archive_path.display());
    Ok(())
}

fn cli_dist(args: Vec<String>) -> Result<(), String> {
    let target_name = parse_target_name(args)?;
    let cli_target = CLI_TARGETS
        .iter()
        .find(|target| target.name == target_name)
        .ok_or_else(|| {
            format!(
                "unknown CLI target `{target_name}`; expected one of: {}",
                CLI_TARGETS
                    .iter()
                    .map(|target| target.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let root = workspace_root()?;
    let archive_dir = root.join("dist").join(cli_target.archive_name);

    if archive_dir.exists() {
        fs::remove_dir_all(&archive_dir).map_err(|err| {
            format!(
                "failed to remove existing CLI dist directory `{}`: {err}",
                archive_dir.display()
            )
        })?;
    }
    fs::create_dir_all(archive_dir.join("bin")).map_err(|err| {
        format!(
            "failed to create CLI dist bin directory `{}`: {err}",
            archive_dir.join("bin").display()
        )
    })?;

    build_cli_target(cli_target, &root)?;
    copy_cli_target_binary(cli_target, &root, &archive_dir)?;
    prepare_licenses(&root, &archive_dir)?;
    write_cli_readme(cli_target, &archive_dir)?;
    let archive_path = package_cli_archive(cli_target, &root, &archive_dir)?;
    println!("CLI dist artifact prepared at {}", archive_dir.display());
    println!("CLI release archive prepared at {}", archive_path.display());
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

    write_release_manifest(&assets_dir)?;
    write_install_sh(&assets_dir)?;
    write_install_ps1(&assets_dir)?;
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

fn parse_target_name(args: Vec<String>) -> Result<String, String> {
    match args.as_slice() {
        [flag, value] if flag == "--target" => Ok(value.clone()),
        [value] => Ok(value.clone()),
        _ => Err("expected `cli-dist --target <target>`".to_owned()),
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

fn build_cli_target(cli_target: &CliTarget, root: &Path) -> Result<(), String> {
    let mut command = Command::new("cargo");
    command.current_dir(root).args([
        "build",
        "-p",
        "lumen-cli",
        "--release",
        "--target",
        cli_target.rust_target,
    ]);
    if cli_target.rust_target.contains("windows") {
        command.env("RUSTFLAGS", rustflags_without_static_crt());
        command.env("CFLAGS", "/MD");
        command.env("CXXFLAGS", "/MD");
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to spawn lumen-cli cargo build: {err}"))?;
    if !status.success() {
        return Err(format!(
            "lumen-cli cargo build failed for target `{}` with status {status}",
            cli_target.name
        ));
    }
    Ok(())
}

fn configure_profile_build_env(profile: &DistProfile, command: &mut Command) {
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

    if profile.openvino_bundle && !profile.target.contains("windows") {
        let bin_dst = archive_dir.join("bin").join("lumen-hub-bin");
        fs::copy(&src, &bin_dst).map_err(|err| {
            format!(
                "failed to copy binary `{}` to `{}`: {err}",
                src.display(),
                bin_dst.display()
            )
        })?;
        make_executable(&bin_dst)?;
        write_openvino_wrapper(&archive_dir.join("bin").join("lumen-hub"))?;
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

fn copy_cli_target_binary(
    cli_target: &CliTarget,
    root: &Path,
    archive_dir: &Path,
) -> Result<(), String> {
    let src = root
        .join("target")
        .join(cli_target.rust_target)
        .join("release")
        .join(cli_target.exe_name);
    let dst = archive_dir.join("bin").join(cli_target.exe_name);

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

fn write_openvino_wrapper(path: &Path) -> Result<(), String> {
    let body = r#"#!/usr/bin/env bash
set -euo pipefail
APP_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LUMNN_ORT_DYLIB_PATH="$APP_HOME/lib/libonnxruntime.so"
export LD_LIBRARY_PATH="$APP_HOME/lib:${LD_LIBRARY_PATH:-}"
exec "$APP_HOME/bin/lumen-hub-bin" "$@"
"#;
    fs::write(path, body).map_err(|err| {
        format!(
            "failed to write OpenVINO wrapper `{}`: {err}",
            path.display()
        )
    })?;
    make_executable(path)
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
                format!("OpenVINO wheel did not contain libonnxruntime.so or libonnxruntime.so.*")
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
    let openvino_note = if profile.openvino_bundle {
        "\nOpenVINO runtime libraries are bundled in `lib/`. Use `bin/lumen-hub`, which sets `LUMNN_ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` before launching the real binary.\n"
    } else {
        ""
    };
    let body = format!(
        "# {}\n\nProfile: `{}`\nTarget: `{}`\nFeatures: `{}`\n\nLayout:\n- `bin/`: executable launcher and binary\n- `lib/`: bundled runtime libraries, when needed\n- `licenses/`: license and third-party notices\n\nStart with:\n\n```bash\n./bin/lumen-hub --config /path/to/lumen-config.json\n```\n{openvino_note}",
        profile.archive_name,
        profile.name,
        profile.target,
        profile.features.join(",")
    );
    fs::write(archive_dir.join("README.md"), body)
        .map_err(|err| format!("failed to write README.md: {err}"))
}

fn write_cli_readme(cli_target: &CliTarget, archive_dir: &Path) -> Result<(), String> {
    let body = format!(
        "# {}\n\nTarget: `{}`\nRust target: `{}`\n\nLayout:\n- `bin/`: lumen-cli executable\n- `licenses/`: license and third-party notices\n\nStart with:\n\n```bash\nlumen-cli init\nlumen-cli start\n```\n",
        cli_target.archive_name, cli_target.name, cli_target.rust_target
    );
    fs::write(archive_dir.join("README.md"), body)
        .map_err(|err| format!("failed to write CLI README.md: {err}"))
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

fn package_cli_archive(
    cli_target: &CliTarget,
    root: &Path,
    archive_dir: &Path,
) -> Result<PathBuf, String> {
    let dist_dir = root.join("dist");
    let archive_file = dist_dir.join(cli_target.file_name());
    if archive_file.exists() {
        fs::remove_file(&archive_file).map_err(|err| {
            format!(
                "failed to remove existing CLI archive `{}`: {err}",
                archive_file.display()
            )
        })?;
    }

    if cli_target.zip {
        zip_directory(archive_dir, &archive_file)?;
    } else {
        let status = Command::new("tar")
            .current_dir(&dist_dir)
            .args(["-czf"])
            .arg(&archive_file)
            .arg(cli_target.archive_name)
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
        let archive_name = Path::new(
            src_dir
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("invalid archive directory `{}`", src_dir.display()))?,
        )
        .join(&relative);
        writer
            .start_file(archive_name.to_string_lossy(), options)
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

fn write_release_manifest(assets_dir: &Path) -> Result<(), String> {
    let version = release_version();
    let base_url =
        env::var("LUMEN_RELEASE_BASE_URL").unwrap_or_else(|_| DEFAULT_RELEASE_BASE_URL.to_owned());
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

    let mut cli = Vec::new();
    for target in CLI_TARGETS {
        let file_name = target.file_name();
        let path = assets_dir.join(&file_name);
        if !path.is_file() {
            continue;
        }
        let sha256 = sha256_file(&path)?;
        cli.push(serde_json::json!({
            "target": target.name,
            "file_name": file_name,
            "url": format!("{}/{}", base_url.trim_end_matches('/'), file_name),
            "sha256": sha256,
        }));
    }

    let manifest = serde_json::json!({
        "version": version,
        "hub": hub,
        "cli": cli,
    });
    let body = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to render release manifest: {err}"))?;
    fs::write(assets_dir.join("manifest.json"), body + "\n")
        .map_err(|err| format!("failed to write release manifest: {err}"))
}

fn write_install_sh(assets_dir: &Path) -> Result<(), String> {
    let body = format!(
        r#"#!/usr/bin/env sh
set -eu

MANIFEST_URL="${{LUMEN_RELEASE_MANIFEST_URL:-{DEFAULT_MANIFEST_URL}}}"
INSTALL_DIR="${{LUMEN_INSTALL_DIR:-$HOME/.lumen/bin}}"

need() {{
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' was not found" >&2
    exit 1
  fi
}}

sha256_of() {{
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{{print $1}}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{{print $1}}'
  else
    echo "error: sha256sum or shasum is required" >&2
    exit 1
  fi
}}

json_field() {{
  manifest_path="$1"
  target="$2"
  field="$3"
  awk -v target="$target" -v field="$field" '
    $0 ~ "\"target\"" {{
      in_target = index($0, "\"" target "\"") > 0
    }}
    in_target && $0 ~ "\"" field "\"" {{
      value = $0
      sub(/^[^:]*:[[:space:]]*/, "", value)
      sub(/,[[:space:]]*$/, "", value)
      sub(/^"/, "", value)
      sub(/"$/, "", value)
      print value
      found = 1
      exit
    }}
    END {{
      if (!found) exit 1
    }}
  ' "$manifest_path"
}}

need curl
need awk
need tar

os="$(uname -s)"
arch="$(uname -m)"
case "$os:$arch" in
  Darwin:arm64|Darwin:aarch64)
    target="darwin-arm64"
    ;;
  Linux:x86_64|Linux:amd64)
    target="linux-x64"
    ;;
  *)
    echo "error: unsupported platform $os/$arch; supported: macOS arm64, Linux x64" >&2
    exit 1
    ;;
esac

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

manifest="$tmp_dir/manifest.json"
curl -fsSL "$MANIFEST_URL" -o "$manifest"

file_name="$(json_field "$manifest" "$target" file_name)"
url="$(json_field "$manifest" "$target" url)"
expected="$(json_field "$manifest" "$target" sha256)"
archive="$tmp_dir/$file_name"

curl -fsSL "$url" -o "$archive"
actual="$(sha256_of "$archive")"
if [ "$actual" != "$expected" ]; then
  echo "error: checksum mismatch for $file_name" >&2
  echo "expected: $expected" >&2
  echo "actual:   $actual" >&2
  exit 1
fi

extract_dir="$tmp_dir/extract"
mkdir -p "$extract_dir" "$INSTALL_DIR"
tar -xzf "$archive" -C "$extract_dir"
cli_path="$(find "$extract_dir" -path '*/bin/lumen-cli' -type f | head -n 1)"
if [ -z "$cli_path" ]; then
  echo "error: archive did not contain bin/lumen-cli" >&2
  exit 1
fi

cp "$cli_path" "$INSTALL_DIR/lumen-cli"
chmod 755 "$INSTALL_DIR/lumen-cli"

echo "lumen-cli installed to $INSTALL_DIR/lumen-cli"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    echo "Add this to your PATH:"
    echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
    ;;
esac
echo "Next:"
echo "  lumen-cli init"
echo "  lumen-cli start"
"#
    );
    fs::write(assets_dir.join("install.sh"), body)
        .map_err(|err| format!("failed to write install.sh: {err}"))?;
    make_executable(&assets_dir.join("install.sh"))
}

fn write_install_ps1(assets_dir: &Path) -> Result<(), String> {
    let body = format!(
        r#"$ErrorActionPreference = "Stop"

$ManifestUrl = if ($env:LUMEN_RELEASE_MANIFEST_URL) {{ $env:LUMEN_RELEASE_MANIFEST_URL }} else {{ "{DEFAULT_MANIFEST_URL}" }}
$InstallDir = if ($env:LUMEN_INSTALL_DIR) {{ $env:LUMEN_INSTALL_DIR }} else {{ Join-Path $env:LOCALAPPDATA "Lumen\bin" }}

if (-not [Environment]::Is64BitOperatingSystem) {{
    throw "unsupported platform: Windows x64 is required"
}}

$TempDir = Join-Path ([IO.Path]::GetTempPath()) ("lumen-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

try {{
    $ManifestPath = Join-Path $TempDir "manifest.json"
    Invoke-WebRequest -Uri $ManifestUrl -OutFile $ManifestPath -UseBasicParsing
    $Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
    $Artifact = $Manifest.cli | Where-Object {{ $_.target -eq "windows-x64" }} | Select-Object -First 1
    if (-not $Artifact) {{
        throw "manifest does not contain CLI artifact for windows-x64"
    }}

    $ArchivePath = Join-Path $TempDir $Artifact.file_name
    Invoke-WebRequest -Uri $Artifact.url -OutFile $ArchivePath -UseBasicParsing
    $Actual = (Get-FileHash -Algorithm SHA256 -Path $ArchivePath).Hash.ToLowerInvariant()
    $Expected = [string]$Artifact.sha256
    if ($Actual -ne $Expected.ToLowerInvariant()) {{
        throw "checksum mismatch for $($Artifact.file_name): expected $Expected, got $Actual"
    }}

    $ExtractDir = Join-Path $TempDir "extract"
    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force
    $CliPath = Get-ChildItem -Path $ExtractDir -Recurse -File -Filter "lumen-cli.exe" |
        Where-Object {{ $_.FullName -match "\\bin\\lumen-cli\.exe$" }} |
        Select-Object -First 1
    if (-not $CliPath) {{
        throw "archive did not contain bin\lumen-cli.exe"
    }}

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item -Path $CliPath.FullName -Destination (Join-Path $InstallDir "lumen-cli.exe") -Force

    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $PathParts = @()
    if ($UserPath) {{ $PathParts = $UserPath -split ";" | Where-Object {{ $_ }} }}
    if ($PathParts -notcontains $InstallDir) {{
        $NewPath = if ($UserPath) {{ "$UserPath;$InstallDir" }} else {{ $InstallDir }}
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    }}
    if (($env:Path -split ";") -notcontains $InstallDir) {{
        $env:Path = "$env:Path;$InstallDir"
    }}

    Write-Host "lumen-cli installed to $(Join-Path $InstallDir "lumen-cli.exe")"
    Write-Host "Next:"
    Write-Host "  lumen-cli init"
    Write-Host "  lumen-cli start"
}} finally {{
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}}
"#
    );
    fs::write(assets_dir.join("install.ps1"), body)
        .map_err(|err| format!("failed to write install.ps1: {err}"))
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
        if name == "checksums.txt" {
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

fn release_version() -> String {
    env::var("LUMEN_RELEASE_VERSION")
        .unwrap_or_else(|_| DEFAULT_RELEASE_VERSION.to_owned())
        .trim_start_matches('v')
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
}

struct CliTarget {
    name: &'static str,
    archive_name: &'static str,
    rust_target: &'static str,
    exe_name: &'static str,
    zip: bool,
}

impl CliTarget {
    fn file_name(&self) -> String {
        if self.zip {
            format!("{}.zip", self.archive_name)
        } else {
            format!("{}.tar.gz", self.archive_name)
        }
    }
}

const PROFILES: &[DistProfile] = &[
    DistProfile {
        name: "universal-cpu",
        archive_name: "lumen-hub-universal-cpu",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-universal-cpu"],
        openvino_bundle: false,
    },
    DistProfile {
        name: "darwin-arm64",
        archive_name: "lumen-hub-darwin-arm64",
        target: "aarch64-apple-darwin",
        features: &["profile-darwin-arm64"],
        openvino_bundle: false,
    },
    DistProfile {
        name: "linux-x64-cuda",
        archive_name: "lumen-hub-linux-x64-cuda",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-linux-x64-cuda"],
        openvino_bundle: false,
    },
    DistProfile {
        name: "windows-x64-dml",
        archive_name: "lumen-hub-windows-x64-dml",
        target: "x86_64-pc-windows-msvc",
        features: &["profile-windows-x64-dml"],
        openvino_bundle: false,
    },
    DistProfile {
        name: "linux-x64-openvino",
        archive_name: "lumen-hub-linux-x64-openvino",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-linux-x64-openvino"],
        openvino_bundle: true,
    },
];

const CLI_TARGETS: &[CliTarget] = &[
    CliTarget {
        name: "darwin-arm64",
        archive_name: "lumen-cli-darwin-arm64",
        rust_target: "aarch64-apple-darwin",
        exe_name: "lumen-cli",
        zip: false,
    },
    CliTarget {
        name: "linux-x64",
        archive_name: "lumen-cli-linux-x64",
        rust_target: "x86_64-unknown-linux-gnu",
        exe_name: "lumen-cli",
        zip: false,
    },
    CliTarget {
        name: "windows-x64",
        archive_name: "lumen-cli-windows-x64",
        rust_target: "x86_64-pc-windows-msvc",
        exe_name: "lumen-cli.exe",
        zip: true,
    },
];
