use std::{
    env, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use sha2::{Digest, Sha256};

const OPENVINO_BUNDLE_ENV: &str = "LUMNN_OPENVINO_BUNDLE_DIR";

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
    fs::create_dir_all(&archive_dir).map_err(|err| {
        format!(
            "failed to create dist directory `{}`: {err}",
            archive_dir.display()
        )
    })?;

    build_profile(profile, &root)?;
    copy_binary(profile, &root, &archive_dir)?;
    write_readme(profile, &archive_dir)?;
    prepare_licenses(&root, &archive_dir)?;

    if profile.openvino_bundle {
        copy_openvino_bundle(profile, &archive_dir)?;
    }

    write_checksums(&archive_dir)?;
    println!("dist artifact prepared at {}", archive_dir.display());
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
    let status = Command::new("cargo")
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
        .map_err(|err| format!("failed to spawn cargo build: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo build failed for profile `{}` with status {status}",
            profile.name
        ))
    }
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
    let dst = archive_dir.join(exe_name);
    fs::copy(&src, &dst).map_err(|err| {
        format!(
            "failed to copy binary `{}` to `{}`: {err}",
            src.display(),
            dst.display()
        )
    })?;
    Ok(())
}

fn copy_openvino_bundle(profile: &DistProfile, archive_dir: &Path) -> Result<(), String> {
    let bundle_dir = env::var_os(OPENVINO_BUNDLE_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "`{OPENVINO_BUNDLE_ENV}` must point to an OpenVINO-enabled ONNX Runtime bundle for profile `{}`",
                profile.name
            )
        })?;
    if !bundle_dir.is_dir() {
        return Err(format!(
            "`{OPENVINO_BUNDLE_ENV}` points to `{}`, which is not a directory",
            bundle_dir.display()
        ));
    }

    let ort_dylib = bundle_dir.join(default_ort_dylib_name(profile.target));
    if !ort_dylib.is_file() {
        return Err(format!(
            "OpenVINO bundle `{}` must contain `{}` at its root",
            bundle_dir.display(),
            default_ort_dylib_name(profile.target)
        ));
    }

    copy_dir_contents(&bundle_dir, archive_dir).map_err(|err| {
        format!(
            "failed to copy OpenVINO bundle `{}` into `{}`: {err}",
            bundle_dir.display(),
            archive_dir.display()
        )
    })
}

fn write_readme(profile: &DistProfile, archive_dir: &Path) -> Result<(), String> {
    let body = format!(
        "# {}\n\nProfile: `{}`\nTarget: `{}`\nFeatures: `{}`\n\nSet `LUMNN_ORT_OPENVINO_DEVICE=CPU` to force OpenVINO CPU execution in environments without Intel GPU/NPU.\n",
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

fn copy_dir_contents(src: &Path, dst: &Path) -> io::Result<()> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            fs::create_dir_all(&dst_path)?;
            copy_dir_contents(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

fn default_ort_dylib_name(target: &str) -> &'static str {
    if target.contains("windows") {
        "onnxruntime.dll"
    } else if target.contains("apple") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
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
        name: "cpu-portable",
        archive_name: "lumen-hub-cpu-portable",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-cpu-portable"],
        openvino_bundle: false,
    },
    DistProfile {
        name: "apple-arm64",
        archive_name: "lumen-hub-apple-arm64",
        target: "aarch64-apple-darwin",
        features: &["profile-apple-arm64"],
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
        name: "linux-x64-tensorrt",
        archive_name: "lumen-hub-linux-x64-tensorrt",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-linux-x64-tensorrt"],
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
    DistProfile {
        name: "windows-x64-openvino",
        archive_name: "lumen-hub-windows-x64-openvino",
        target: "x86_64-pc-windows-msvc",
        features: &["profile-windows-x64-openvino"],
        openvino_bundle: true,
    },
    DistProfile {
        name: "linux-x64-cpu-optimized",
        archive_name: "lumen-hub-linux-x64-cpu-optimized",
        target: "x86_64-unknown-linux-gnu",
        features: &["profile-linux-x64-cpu-optimized"],
        openvino_bundle: false,
    },
];
