//! Build automation for Lumen Hub Burn distributions.
//!
//! Burn statically links its compute backends, so distributions are just the
//! built binaries plus warmup assets and licenses — there are no external
//! runtime libraries (ONNX Runtime, MNN, ...) to bundle anymore.
//!
//! Commands:
//!   cargo xtask dist --profile <profile>                Build + package one profile.
//!   cargo xtask release-metadata [--assets-dir <dir>]   Write manifest.json + checksums.txt.

use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use sha2::{Digest, Sha256};
use zip::{ZipWriter, write::SimpleFileOptions};

/// A distribution profile: an OS/arch target plus the Burn compute backend.
struct DistProfile {
    /// Profile id used on the CLI and in release artifact names.
    name: &'static str,
    /// Rust target triple.
    target: &'static str,
    /// lumen-hub backend + model cargo features.
    features: &'static [&'static str],
}

const MODEL_FEATURES: &[&str] = &["siglip", "ppocr", "insightface"];

// The `*-gpu` profiles use the `wgpu` backend, which targets Vulkan/GL/DX12 at
// runtime (so a single binary covers "vulkan + wgpu"). `cuda`/`rocm`/`jetson`
// are vendor-specific builds that require their toolkit in the build
// environment (installed by the release workflow); `jetson` is the arm64 CUDA
// build against the L4T/Tegra stack.
const PROFILES: &[DistProfile] = &[
    // macOS
    DistProfile {
        name: "darwin-arm64-metal",
        target: "aarch64-apple-darwin",
        features: &["metal", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "darwin-arm64-cpu",
        target: "aarch64-apple-darwin",
        features: &["cpu", "siglip", "ppocr", "insightface"],
    },
    // Windows
    DistProfile {
        name: "windows-x64-cpu",
        target: "x86_64-pc-windows-msvc",
        features: &["cpu", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "windows-x64-gpu",
        target: "x86_64-pc-windows-msvc",
        features: &["wgpu", "siglip", "ppocr", "insightface"],
    },
    // Linux x64
    DistProfile {
        name: "linux-x64-cpu",
        target: "x86_64-unknown-linux-gnu",
        features: &["cpu", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "linux-x64-gpu",
        target: "x86_64-unknown-linux-gnu",
        features: &["wgpu", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "linux-x64-cuda",
        target: "x86_64-unknown-linux-gnu",
        features: &["cuda", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "linux-x64-rocm",
        target: "x86_64-unknown-linux-gnu",
        features: &["rocm", "siglip", "ppocr", "insightface"],
    },
    // Linux arm64
    DistProfile {
        name: "linux-arm64-cpu",
        target: "aarch64-unknown-linux-gnu",
        features: &["cpu", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "linux-arm64-gpu",
        target: "aarch64-unknown-linux-gnu",
        features: &["wgpu", "siglip", "ppocr", "insightface"],
    },
    DistProfile {
        name: "linux-arm64-jetson",
        target: "aarch64-unknown-linux-gnu",
        features: &["cuda", "siglip", "ppocr", "insightface"],
    },
];

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
            .map(|p| p.name)
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

fn dist(args: Vec<String>) -> Result<(), String> {
    let profile_name = parse_named_arg(&args, "--profile")?
        .ok_or_else(|| "missing required argument `--profile <profile>`".to_owned())?;
    let profile = PROFILES
        .iter()
        .find(|p| p.name == profile_name)
        .ok_or_else(|| {
            format!(
                "unknown dist profile `{profile_name}`; expected one of: {}",
                PROFILES
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    let root = workspace_root()?;
    let archive_name = format!("lumen-hub-{}", profile.name);
    let staging = root.join("dist").join(&archive_name);
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|e| format!("clean {}: {e}", staging.display()))?;
    }
    fs::create_dir_all(staging.join("bin")).map_err(|e| format!("mkdir bin: {e}"))?;

    build_profile(profile, &root)?;
    copy_binary(&root, profile, "lumen-hub", &staging)?;
    copy_binary(&root, profile, "lumen-cli", &staging)?;
    copy_warmup_assets(&root, &staging)?;
    copy_licenses(&root, &staging)?;
    write_readme(profile, &staging)?;

    let archive = root.join("dist").join(format!("{archive_name}.zip"));
    zip_directory(&staging, &archive)?;
    println!("packaged {}", archive.display());
    Ok(())
}

fn build_profile(profile: &DistProfile, root: &Path) -> Result<(), String> {
    let features = profile.features.join(",");
    println!(
        "building lumen-hub + lumen-cli for {} (features: {features})",
        profile.target
    );
    let status = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()))
        .current_dir(root)
        .args([
            "build",
            "--release",
            "--target",
            profile.target,
            "-p",
            "lumen-hub",
            "-p",
            "lumen-cli",
            "--no-default-features",
            "--features",
            &features,
        ])
        .status()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if !status.success() {
        return Err(format!("cargo build failed for profile `{}`", profile.name));
    }
    Ok(())
}

fn copy_binary(
    root: &Path,
    profile: &DistProfile,
    bin: &str,
    staging: &Path,
) -> Result<(), String> {
    let exe = if profile.target.contains("windows") {
        format!("{bin}.exe")
    } else {
        bin.to_owned()
    };
    let src = root
        .join("target")
        .join(profile.target)
        .join("release")
        .join(&exe);
    let dst = staging.join("bin").join(&exe);
    fs::copy(&src, &dst)
        .map(|_| ())
        .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))
}

fn copy_warmup_assets(root: &Path, staging: &Path) -> Result<(), String> {
    // Warmup fixtures live with the hub crate (see warmup::default_warmup_dir).
    let src = root.join("crates").join("lumen-hub").join("warmup");
    if !src.is_dir() {
        return Ok(());
    }
    copy_dir_recursive(&src, &staging.join("warmup"))
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("mkdir {}: {e}", dst.display()))?;
    for entry in fs::read_dir(src).map_err(|e| format!("read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target)?;
        } else {
            fs::copy(&path, &target)
                .map(|_| ())
                .map_err(|e| format!("copy {}: {e}", path.display()))?;
        }
    }
    Ok(())
}

fn copy_licenses(root: &Path, staging: &Path) -> Result<(), String> {
    let license = root.join("LICENSE");
    if license.is_file() {
        fs::copy(&license, staging.join("LICENSE"))
            .map(|_| ())
            .map_err(|e| format!("copy LICENSE: {e}"))?;
    }
    Ok(())
}

fn write_readme(profile: &DistProfile, staging: &Path) -> Result<(), String> {
    let backend = profile
        .features
        .iter()
        .find(|f| !MODEL_FEATURES.contains(f))
        .copied()
        .unwrap_or("cpu");
    let body = format!(
        "# Lumen Hub ({name})\n\nBurn backend: {backend}\nTarget: {target}\n\nRun:\n  ./bin/lumen-hub --config <config.yaml>\n\nModels are downloaded on first start into the configured cache_dir.\n",
        name = profile.name,
        target = profile.target,
    );
    fs::write(staging.join("README.md"), body).map_err(|e| format!("write README: {e}"))
}

fn release_metadata(args: Vec<String>) -> Result<(), String> {
    let root = workspace_root()?;
    let assets_dir = parse_named_arg(&args, "--assets-dir")?
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("dist"));

    // Version + download base for the manifest the installer consumes. In CI the
    // release workflow sets these; locally they fall back to a dev placeholder.
    let version = env::var("LUMEN_RELEASE_VERSION").unwrap_or_else(|_| "0.0.0-dev".to_owned());
    let base_url = env::var("LUMEN_RELEASE_BASE_URL").unwrap_or_else(|_| {
        format!("https://github.com/EdwinZhanCN/Lumen-Hub/releases/download/{version}")
    });
    let base_url = base_url.trim_end_matches('/').to_owned();

    // 1. Hub manifest: one entry per lumen-hub-<profile>.zip.
    let mut hub = Vec::new();
    for entry in
        fs::read_dir(&assets_dir).map_err(|e| format!("read {}: {e}", assets_dir.display()))?
    {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(profile) = name
            .strip_prefix("lumen-hub-")
            .and_then(|rest| rest.strip_suffix(".zip"))
        {
            let sha = sha256_file(&path)?;
            hub.push((profile.to_owned(), name.clone(), sha));
        }
    }
    hub.sort();
    let hub_json = hub
        .iter()
        .map(|(profile, file_name, sha)| {
            format!(
                r#"{{"profile":"{profile}","file_name":"{file_name}","url":"{base_url}/{file_name}","sha256":"{sha}"}}"#
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let manifest = format!(r#"{{"version":"{version}","hub":[{hub_json}]}}"#);
    fs::write(assets_dir.join("manifest.json"), manifest + "\n")
        .map_err(|e| format!("write manifest.json: {e}"))?;
    println!("wrote {}", assets_dir.join("manifest.json").display());

    // 2. Top-level checksums over every asset (manifest included, self excluded).
    let mut lines = Vec::new();
    for entry in
        fs::read_dir(&assets_dir).map_err(|e| format!("read {}: {e}", assets_dir.display()))?
    {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "checksums.txt" {
            continue;
        }
        lines.push(format!("{}  {name}", sha256_file(&path)?));
    }
    lines.sort();
    fs::write(assets_dir.join("checksums.txt"), lines.join("\n") + "\n")
        .map_err(|e| format!("write checksums.txt: {e}"))?;
    println!("wrote {}", assets_dir.join("checksums.txt").display());
    Ok(())
}

fn zip_directory(src_dir: &Path, zip_path: &Path) -> Result<(), String> {
    let file = File::create(zip_path).map_err(|e| format!("create {}: {e}", zip_path.display()))?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default();
    let base = src_dir
        .parent()
        .ok_or_else(|| "archive dir has no parent".to_owned())?;
    zip_dir_into(&mut zip, src_dir, base, options)?;
    zip.finish().map_err(|e| format!("finish zip: {e}"))?;
    Ok(())
}

fn zip_dir_into(
    zip: &mut ZipWriter<File>,
    dir: &Path,
    base: &Path,
    options: SimpleFileOptions,
) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let path = entry.path();
        let rel = path
            .strip_prefix(base)
            .map_err(|e| format!("strip prefix: {e}"))?
            .to_string_lossy()
            .replace('\\', "/");
        if path.is_dir() {
            zip.add_directory(format!("{rel}/"), options)
                .map_err(|e| format!("zip dir {rel}: {e}"))?;
            zip_dir_into(zip, &path, base, options)?;
        } else {
            zip.start_file(rel.clone(), options)
                .map_err(|e| format!("zip start {rel}: {e}"))?;
            let mut buf = Vec::new();
            File::open(&path)
                .and_then(|mut f| f.read_to_end(&mut buf))
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            zip.write_all(&buf)
                .map_err(|e| format!("zip write {rel}: {e}"))?;
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn workspace_root() -> Result<PathBuf, String> {
    // crates/xtask/../.. => workspace root
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to resolve workspace root".to_owned())
}

fn parse_named_arg(args: &[String], flag: &str) -> Result<Option<String>, String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter
                .next()
                .cloned()
                .map(Some)
                .ok_or_else(|| format!("missing value for `{flag}`"));
        }
        if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            return Ok(Some(value.to_owned()));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_have_unique_names() {
        let mut names: Vec<_> = PROFILES.iter().map(|p| p.name).collect();
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len, "duplicate profile names");
    }

    #[test]
    fn every_profile_selects_a_backend_feature() {
        for profile in PROFILES {
            assert!(
                profile.features.iter().any(|f| !MODEL_FEATURES.contains(f)),
                "profile {} has no backend feature",
                profile.name
            );
        }
    }

    #[test]
    fn parses_named_args() {
        let args = vec!["--profile".to_owned(), "linux-x64-cpu".to_owned()];
        assert_eq!(
            parse_named_arg(&args, "--profile").unwrap(),
            Some("linux-x64-cpu".to_owned())
        );
        let eq = vec!["--profile=darwin-arm64-metal".to_owned()];
        assert_eq!(
            parse_named_arg(&eq, "--profile").unwrap(),
            Some("darwin-arm64-metal".to_owned())
        );
    }
}
