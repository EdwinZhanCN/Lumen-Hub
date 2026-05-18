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
const BETA_VERSION: &str = "0.1.0-beta.1";
const DEFAULT_RELEASE_BASE_URL: &str =
    "https://github.com/Lumilio-Photos/lumen-rs/releases/download/v0.1.0-beta.1";

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
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(format!("unknown command `{other}`")),
    }
}

fn print_help() {
    println!(
        "Usage:\n  cargo xtask dist --profile <profile>\n\nProfiles:\n  {}",
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

    write_readme(profile, &archive_dir)?;
    write_checksums(&archive_dir)?;
    let archive_path = package_archive(profile, &root, &archive_dir)?;
    write_release_manifest(&root)?;
    println!("dist artifact prepared at {}", archive_dir.display());
    println!("release archive prepared at {}", archive_path.display());
    Ok(())
}

fn parse_profile_name(args: Vec<String>) -> Result<String, String> {
    match args.as_slice() {
        [flag, value] if flag == "--profile" => Ok(value.clone()),
        [value] => Ok(value.clone()),
        _ => Err("expected `dist --profile <profile>`".to_owned()),
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
    let hub_status = Command::new("cargo")
        .current_dir(root)
        .args([
            "build",
            "-p",
            "lumen-hub",
            "--release",
            "--no-default-features",
            "--target",
            profile.target,
            "--features",
            &features,
        ])
        .status()
        .map_err(|err| format!("failed to spawn lumen-hub cargo build: {err}"))?;
    if !hub_status.success() {
        return Err(format!(
            "lumen-hub cargo build failed for profile `{}` with status {hub_status}",
            profile.name
        ));
    }

    let cli_status = Command::new("cargo")
        .current_dir(root)
        .args([
            "build",
            "-p",
            "lumen-cli",
            "--release",
            "--target",
            profile.target,
        ])
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

fn write_release_manifest(root: &Path) -> Result<(), String> {
    let dist_dir = root.join("dist");
    let base_url =
        env::var("LUMEN_RELEASE_BASE_URL").unwrap_or_else(|_| DEFAULT_RELEASE_BASE_URL.to_owned());
    let mut artifacts = Vec::new();

    for profile in PROFILES {
        let file_name = if profile.target.contains("windows") {
            format!("{}.zip", profile.archive_name)
        } else {
            format!("{}.tar.gz", profile.archive_name)
        };
        let path = dist_dir.join(&file_name);
        if !path.is_file() {
            continue;
        }
        let sha256 = sha256_file(&path)?;
        artifacts.push(serde_json::json!({
            "profile": profile.name,
            "file_name": file_name,
            "url": format!("{}/{}", base_url.trim_end_matches('/'), file_name),
            "sha256": sha256,
        }));
    }

    let manifest = serde_json::json!({
        "version": BETA_VERSION,
        "hub": artifacts,
    });
    let body = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to render release manifest: {err}"))?;
    fs::write(dist_dir.join("manifest.json"), body + "\n")
        .map_err(|err| format!("failed to write release manifest: {err}"))
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
