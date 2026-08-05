//! Build automation for Lumen Hub Burn distributions.
//!
//! Burn statically links its compute backends, so distributions are just the
//! built binaries plus warmup assets and licenses — there are no external
//! runtime libraries (ONNX Runtime, MNN, ...) to bundle anymore.
//!
//! Commands:
//!   cargo xtask dist --profile <profile>               Build + package one profile.
//!   cargo xtask release-metadata [--assets-dir <dir>]  Write schema-2 manifest.json + SHA256SUMS.
//!   cargo xtask config-fixtures [--check]              Regenerate preset/custom config fixtures.
//!   cargo xtask contract-check                         Verify local proto provenance + buf lint.
//!   cargo xtask contract-verify                        Add remote SDK/baseline proof.
//!   cargo xtask contract-sync --sdk <tag>              Re-vendor the SDK-owned proto.

use std::{
    env,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use sha2::{Digest, Sha256};
use zip::{ZipWriter, write::SimpleFileOptions};

use lumen_schema::{ArtifactInfo, PlatformInfo};

/// The fixed current-major baseline tag for `buf breaking` (WIRE_JSON
/// policy). The data-plane and control-plane contracts are frozen within this
/// major; any wire-level break against this tag fails CI. Bump only on a
/// protocol-major release.
const CONTRACT_BASELINE_TAG: &str = "v0.1.1";

const ML_SERVICE_PROTO_REL: &str = "crates/lumen-hub/proto/ml_service.proto";
const PROVENANCE_REL: &str = "crates/lumen-hub/proto/provenance.json";
const SDK_REPOSITORY: &str = "https://github.com/EdwinZhanCN/Lumen-SDK.git";
const SDK_RAW_PREFIX: &str = "https://raw.githubusercontent.com/EdwinZhanCN/Lumen-SDK";

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Provenance {
    schema_version: u32,
    ml_service: ProvenanceSource,
}

#[derive(serde::Deserialize)]
struct ProvenanceSource {
    authority: String,
    tag: String,
    commit: String,
    sha256: String,
}

/// A distribution profile: an OS/arch target plus the Burn compute backend.
struct DistProfile {
    /// Profile id used on the CLI and in release artifact names.
    name: &'static str,
    /// Rust target triple.
    target: &'static str,
    /// lumen-hub backend + model cargo features.
    features: &'static [&'static str],
}

const MODEL_FEATURES: &[&str] = &["siglip", "ppocr", "insightface", "clip"];

// The `*-gpu` profiles use the `wgpu` backend, which targets Vulkan/GL/DX12 at
// runtime (so a single binary covers "vulkan + wgpu"). `cuda`/`rocm`/`jetson`
// are vendor-specific source-build recipes that require their toolkit in the
// build environment; the release workflow deliberately publishes only the
// subset it can continuously exercise. `jetson` targets the L4T/Tegra stack.
const PROFILES: &[DistProfile] = &[
    // macOS
    DistProfile {
        name: "darwin-arm64-metal",
        target: "aarch64-apple-darwin",
        features: &["metal", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "darwin-arm64-cpu",
        target: "aarch64-apple-darwin",
        features: &["cpu", "siglip", "ppocr", "insightface", "clip"],
    },
    // Windows
    DistProfile {
        name: "windows-x64-cpu",
        target: "x86_64-pc-windows-msvc",
        features: &["cpu", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "windows-x64-gpu",
        target: "x86_64-pc-windows-msvc",
        features: &["wgpu", "siglip", "ppocr", "insightface", "clip"],
    },
    // Linux x64
    DistProfile {
        name: "linux-x64-cpu",
        target: "x86_64-unknown-linux-gnu",
        features: &["cpu", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "linux-x64-gpu",
        target: "x86_64-unknown-linux-gnu",
        features: &["wgpu", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "linux-x64-cuda",
        target: "x86_64-unknown-linux-gnu",
        features: &["cuda", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "linux-x64-rocm",
        target: "x86_64-unknown-linux-gnu",
        features: &["rocm", "siglip", "ppocr", "insightface", "clip"],
    },
    // Linux arm64
    DistProfile {
        name: "linux-arm64-cpu",
        target: "aarch64-unknown-linux-gnu",
        features: &["cpu", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "linux-arm64-gpu",
        target: "aarch64-unknown-linux-gnu",
        features: &["wgpu", "siglip", "ppocr", "insightface", "clip"],
    },
    DistProfile {
        name: "linux-arm64-jetson",
        target: "aarch64-unknown-linux-gnu",
        features: &["cuda", "siglip", "ppocr", "insightface", "clip"],
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
        Some("config-fixtures") => config_fixtures(args.collect()),
        Some("contract-check") => contract_check(args.collect()),
        Some("contract-verify") => contract_verify(args.collect()),
        Some("contract-sync") => contract_sync(args.collect()),
        Some("golden") => golden(args.collect()),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(other) => Err(format!("unknown command `{other}`")),
    }
}

fn print_help() {
    println!(
        "Usage:\n  cargo xtask dist --profile <profile>\n  cargo xtask release-metadata [--assets-dir <dir>]\n  cargo xtask config-fixtures [--check]   Regenerate preset/custom config fixtures\n  cargo xtask contract-check              Verify committed provenance + buf lint (offline)\n  cargo xtask contract-verify             Verify SDK source + fixed-major baseline (network)\n  cargo xtask contract-sync --sdk <tag>    Re-vendor ml_service.proto from an SDK tag\n  cargo xtask golden [--models-dir <dir>]   Regenerate l1 golden embeddings\n\nProfiles:\n  {}",
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

    // 1. Hub artifacts: one entry per lumen-hub-<profile>.zip.
    let mut artifacts = Vec::new();
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
            artifacts.push(ArtifactInfo {
                profile: profile.to_owned(),
                file_name: name.clone(),
                sha256: sha,
            });
        }
    }

    if artifacts.is_empty() {
        return Err(format!(
            "no lumen-hub-<profile>.zip archives found in {}",
            assets_dir.display()
        ));
    }
    artifacts.sort_by(|left, right| left.profile.cmp(&right.profile));

    // 2. Protocol provenance: hash the proto sources at release time and
    //    verify the data-plane major still matches the schema constant.
    let ml_service_proto = root.join("crates/lumen-hub/proto/ml_service.proto");
    let control_proto = root.join("crates/lumen-hub/proto/control.proto");
    let ml_service_sha = sha256_file(&ml_service_proto)?;
    let control_sha = sha256_file(&control_proto)?;
    let data_plane_major = data_plane_major_from_proto(&ml_service_proto)?;
    if data_plane_major != lumen_schema::DATA_PLANE_MAJOR {
        return Err(format!(
            "ml_service.proto package major {data_plane_major} does not match lumen-schema DATA_PLANE_MAJOR {}",
            lumen_schema::DATA_PLANE_MAJOR
        ));
    }

    // 3. Dist platforms are derived from the archives that actually exist.
    // PROFILES may retain source-build recipes that are intentionally not part
    // of the supported release surface.
    let mut platforms = Vec::with_capacity(artifacts.len());
    for artifact in &artifacts {
        let profile = PROFILES
            .iter()
            .find(|candidate| candidate.name == artifact.profile)
            .ok_or_else(|| {
                format!(
                    "release archive {} names unknown profile {}",
                    artifact.file_name, artifact.profile
                )
            })?;
        platforms.push(PlatformInfo {
            name: profile.name.to_owned(),
            target: profile.target.to_owned(),
            backend: profile_backend(profile).to_owned(),
        });
    }

    let manifest = lumen_schema::HubManifest::build(
        &version,
        &base_url,
        &platforms,
        &artifacts,
        lumen_schema::ManifestProtocol {
            data_plane_major,
            ml_service: lumen_schema::ManifestProtocolFile {
                path: "crates/lumen-hub/proto/ml_service.proto".to_owned(),
                sha256: ml_service_sha,
            },
            control: lumen_schema::ManifestProtocolFile {
                path: "crates/lumen-hub/proto/control.proto".to_owned(),
                sha256: control_sha,
            },
        },
    );
    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| format!("serialize manifest: {e}"))?;
    fs::write(assets_dir.join("manifest.json"), manifest_json + "\n")
        .map_err(|e| format!("write manifest.json: {e}"))?;
    println!(
        "wrote {} (schemaVersion {})",
        assets_dir.join("manifest.json").display(),
        lumen_schema::MANIFEST_SCHEMA_VERSION
    );

    // 4. Top-level checksums over every asset (manifest included, self excluded).
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
        if name == "SHA256SUMS" {
            continue;
        }
        lines.push(format!("{}  {name}", sha256_file(&path)?));
    }
    lines.sort();
    fs::write(assets_dir.join("SHA256SUMS"), lines.join("\n") + "\n")
        .map_err(|e| format!("write SHA256SUMS: {e}"))?;
    println!("wrote {}", assets_dir.join("SHA256SUMS").display());
    Ok(())
}

/// The compute backend of a dist profile: the first feature that is not a
/// model feature (see PROFILES).
fn profile_backend(profile: &DistProfile) -> &'static str {
    profile
        .features
        .iter()
        .find(|feature| !MODEL_FEATURES.contains(feature))
        .copied()
        .unwrap_or("cpu")
}

/// Parse the `package <name>.vN;` major from a proto file and fail if the
/// file does not carry a versioned package.
fn data_plane_major_from_proto(path: &Path) -> Result<u32, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    data_plane_major_from_content(&content).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse the `package <name>.vN;` major from proto source bytes.
fn data_plane_major_from_content(content: &str) -> Result<u32, String> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(package) = line.strip_prefix("package ") {
            let package = package.trim_end_matches(';').trim();
            let major = package
                .rsplit('.')
                .next()
                .and_then(|part| part.strip_prefix('v'))
                .and_then(|digit| digit.parse::<u32>().ok())
                .ok_or_else(|| format!("package `{package}` has no parseable vN major"))?;
            return Ok(major);
        }
    }
    Err("no `package` line found".to_owned())
}

/// Regenerate the stable preset/custom config fixtures that every entry point
/// (launcher, Docker, managed applications) is verified against.
fn config_fixtures(args: Vec<String>) -> Result<(), String> {
    let check = args.iter().any(|argument| argument == "--check");
    let root = workspace_root()?;
    let fixtures_dir = root.join("fixtures").join("config");
    fs::create_dir_all(&fixtures_dir)
        .map_err(|e| format!("mkdir {}: {e}", fixtures_dir.display()))?;

    // Deterministic environment options: fixtures are machine-independent.
    let options = lumen_schema::RenderOptions {
        region: "other",
        cache_dir: "/var/lib/lumen/models",
        target: lumen_schema::ConfigTarget::Network,
    };

    let mut fixtures: Vec<(String, lumen_schema::LumenConfig)> = Vec::new();
    for preset in lumen_schema::Preset::all() {
        fixtures.push((
            preset.name.to_owned(),
            lumen_schema::preset_config(*preset, &options)?,
        ));
    }
    // Representative custom combinations covering the Docker/CLI custom path:
    // multi-capability with model/dataset overrides, non-default pairing, and
    // the smallest single-capability selection.
    fixtures.push((
        "custom-siglip-bioclip".to_owned(),
        lumen_schema::custom_config(
            &["siglip", "bioclip"],
            Some(lumen_schema::SIGLIP_BRAVE_MODEL),
            Some(lumen_schema::BIOCLIP_FULL_DATASET),
            &options,
        )?,
    ));
    fixtures.push((
        "custom-face-ocr".to_owned(),
        lumen_schema::custom_config(&["face", "ocr"], None, None, &options)?,
    ));
    fixtures.push((
        "custom-siglip".to_owned(),
        lumen_schema::custom_config(&["siglip"], None, None, &options)?,
    ));

    let mut drifted = Vec::new();
    for (name, config) in &fixtures {
        let mut content = format!(
            "# Generated by `cargo xtask config-fixtures`. Do not edit.\n# Selection: {name}\n"
        );
        content.push_str(
            &serde_yaml::to_string(config).map_err(|e| format!("serialize {name}: {e}"))?,
        );
        let path = fixtures_dir.join(format!("{name}.yaml"));
        if check {
            let existing =
                fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            if existing != content {
                drifted.push(name.clone());
            }
        } else {
            fs::write(&path, content).map_err(|e| format!("write {}: {e}", path.display()))?;
        }
    }

    if check {
        if drifted.is_empty() {
            println!("config fixtures are up to date ({} files)", fixtures.len());
        } else {
            return Err(format!(
                "config fixtures are out of date: {}; run `cargo xtask config-fixtures`",
                drifted.join(", ")
            ));
        }
    } else {
        println!(
            "wrote {} config fixtures to {}",
            fixtures.len(),
            fixtures_dir.display()
        );
    }
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
            zip.add_directory(format!("{rel}/"), options.unix_permissions(0o755))
                .map_err(|e| format!("zip dir {rel}: {e}"))?;
            zip_dir_into(zip, &path, base, options)?;
        } else {
            let mode = if path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                == Some("bin")
            {
                0o755
            } else {
                0o644
            };
            zip.start_file(rel.clone(), options.unix_permissions(mode))
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

/// Verifies the committed proto contract without consulting a remote
/// repository. This is the deterministic check used by normal CI.
fn contract_check(args: Vec<String>) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!(
            "contract-check does not accept arguments: {}",
            args.join(" ")
        ));
    }
    let root = workspace_root()?;
    let provenance = validate_local_contract(&root)?;
    run_buf(&root.join("crates/lumen-hub/proto"), &["lint"])?;
    println!(
        "contract-check ok — local provenance {}@{} and buf lint",
        provenance.ml_service.authority, provenance.ml_service.tag
    );
    Ok(())
}

/// Adds the two network-bound proofs that do not belong in every unrelated CI
/// run: the vendored SDK source/tag identity and WIRE_JSON compatibility with
/// the fixed Hub baseline tag.
fn contract_verify(args: Vec<String>) -> Result<(), String> {
    if !args.is_empty() {
        return Err(format!(
            "contract-verify does not accept arguments: {}",
            args.join(" ")
        ));
    }
    let root = workspace_root()?;
    let proto_dir = root.join("crates/lumen-hub/proto");
    let provenance = validate_local_contract(&root)?;
    run_buf(&proto_dir, &["lint"])?;
    verify_sdk_provenance(&provenance)?;

    let against = format!(
        "https://github.com/EdwinZhanCN/Lumen-Hub.git#ref={CONTRACT_BASELINE_TAG},subdir=crates/lumen-hub/proto"
    );
    run_buf(&proto_dir, &["breaking", "--against", &against])?;
    println!(
        "contract-verify ok — SDK source and WIRE_JSON baseline {CONTRACT_BASELINE_TAG} verified"
    );
    Ok(())
}

/// Re-vendors the SDK-owned data-plane proto from exactly one immutable tag.
fn contract_sync(args: Vec<String>) -> Result<(), String> {
    let tag = match args.as_slice() {
        [flag, tag] if flag == "--sdk" => tag.clone(),
        [arg] if arg.starts_with("--sdk=") => arg
            .strip_prefix("--sdk=")
            .expect("prefix checked")
            .to_owned(),
        _ => return Err("usage: cargo xtask contract-sync --sdk <tag>".to_owned()),
    };
    if tag.trim().is_empty() {
        return Err("SDK tag must not be empty".to_owned());
    }
    let root = workspace_root()?;
    sync_ml_service_from_sdk(
        &root.join(ML_SERVICE_PROTO_REL),
        &root.join(PROVENANCE_REL),
        &tag,
    )
}

fn validate_local_contract(root: &Path) -> Result<Provenance, String> {
    let ml_service_path = root.join(ML_SERVICE_PROTO_REL);
    let provenance_path = root.join(PROVENANCE_REL);
    let provenance_raw = fs::read(&provenance_path)
        .map_err(|e| format!("read {}: {e}", provenance_path.display()))?;
    let provenance: Provenance = serde_json::from_slice(&provenance_raw)
        .map_err(|e| format!("parse {}: {e}", provenance_path.display()))?;

    if provenance.schema_version != 1 {
        return Err(format!(
            "provenance schemaVersion {} is unsupported",
            provenance.schema_version
        ));
    }
    if provenance.ml_service.authority != "EdwinZhanCN/lumen-sdk" {
        return Err(format!(
            "unexpected ml_service authority {}",
            provenance.ml_service.authority
        ));
    }
    if !is_lower_hex(&provenance.ml_service.commit, 40) {
        return Err(
            "provenance ml_service commit must be a 40-character lowercase hex SHA".to_owned(),
        );
    }
    if !is_lower_hex(&provenance.ml_service.sha256, 64) {
        return Err("provenance ml_service sha256 must be 64-character lowercase hex".to_owned());
    }

    let vendored_sha = sha256_file(&ml_service_path)?;
    if vendored_sha != provenance.ml_service.sha256 {
        return Err(format!(
            "{ML_SERVICE_PROTO_REL} sha256 {vendored_sha} does not match provenance {}@{} ({}); run `cargo xtask contract-sync --sdk <tag>`",
            provenance.ml_service.authority,
            provenance.ml_service.tag,
            provenance.ml_service.sha256
        ));
    }
    let data_plane_major = data_plane_major_from_proto(&ml_service_path)?;
    if data_plane_major != lumen_schema::DATA_PLANE_MAJOR {
        return Err(format!(
            "data-plane major {data_plane_major} does not match lumen-schema DATA_PLANE_MAJOR {}",
            lumen_schema::DATA_PLANE_MAJOR
        ));
    }
    println!(
        "provenance ok — vendored ml_service.proto matches {}@{} (commit {}, sha {})",
        provenance.ml_service.authority,
        provenance.ml_service.tag,
        short_hash(&provenance.ml_service.commit),
        short_hash(&vendored_sha)
    );
    Ok(provenance)
}

fn verify_sdk_provenance(provenance: &Provenance) -> Result<(), String> {
    let commit = resolve_remote_tag(SDK_REPOSITORY, &provenance.ml_service.tag)?;
    if commit != provenance.ml_service.commit {
        return Err(format!(
            "SDK tag {} resolves to {}, provenance records {}",
            provenance.ml_service.tag, commit, provenance.ml_service.commit
        ));
    }

    let url = format!(
        "{SDK_RAW_PREFIX}/{}/proto/ml_service.proto",
        provenance.ml_service.tag
    );
    let bytes = fetch_url(&url)?;
    let remote_sha = sha256_bytes(&bytes);
    if remote_sha != provenance.ml_service.sha256 {
        return Err(format!(
            "SDK source {} hashes to {}, provenance records {}",
            url, remote_sha, provenance.ml_service.sha256
        ));
    }
    println!(
        "SDK provenance verified — {} @ {} ({})",
        provenance.ml_service.authority,
        provenance.ml_service.tag,
        short_hash(&commit)
    );
    Ok(())
}

/// Re-vendors `ml_service.proto` from a Lumen-SDK release tag.
fn sync_ml_service_from_sdk(
    ml_service_path: &Path,
    provenance_path: &Path,
    tag: &str,
) -> Result<(), String> {
    let commit = resolve_remote_tag(SDK_REPOSITORY, tag)?;

    let url = format!("{SDK_RAW_PREFIX}/{tag}/proto/ml_service.proto");
    let bytes = fetch_url(&url)?;
    let fetched_sha = sha256_bytes(&bytes);
    let content = String::from_utf8(bytes.clone())
        .map_err(|_| format!("fetched {url} is not valid UTF-8"))?;
    let major = data_plane_major_from_content(&content)?;
    if major != lumen_schema::DATA_PLANE_MAJOR {
        return Err(format!(
            "refusing to vendor {tag}: data-plane major {major} does not match lumen-schema DATA_PLANE_MAJOR {}",
            lumen_schema::DATA_PLANE_MAJOR
        ));
    }

    fs::write(ml_service_path, &bytes)
        .map_err(|e| format!("write {}: {e}", ml_service_path.display()))?;
    let provenance = serde_json::json!({
        "schemaVersion": 1,
        "mlService": {
            "authority": "EdwinZhanCN/lumen-sdk",
            "tag": tag,
            "commit": commit,
            "sha256": fetched_sha,
        },
    });
    fs::write(
        provenance_path,
        serde_json::to_string_pretty(&provenance).expect("serialize provenance") + "\n",
    )
    .map_err(|e| format!("write {}: {e}", provenance_path.display()))?;

    println!(
        "vendored {url} -> {} ({})",
        ml_service_path.display(),
        short_hash(&fetched_sha)
    );
    println!(
        "provenance updated — EdwinZhanCN/lumen-sdk @ {tag} ({})",
        short_hash(&commit)
    );
    println!("next: run `cargo build` and `cargo xtask contract-check`, then review the diff");
    Ok(())
}

fn resolve_remote_tag(repository: &str, tag: &str) -> Result<String, String> {
    let output = Command::new("git")
        .args([
            "ls-remote",
            repository,
            &format!("refs/tags/{tag}"),
            &format!("refs/tags/{tag}^{{}}"),
        ])
        .output()
        .map_err(|e| format!("git ls-remote failed: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-remote failed for {tag}: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let peeled = format!("refs/tags/{tag}^{{}}");
    let plain = format!("refs/tags/{tag}");
    let mut plain_commit = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let mut fields = line.split_whitespace();
        let Some(sha) = fields.next() else { continue };
        let Some(ref_name) = fields.next() else {
            continue;
        };
        if ref_name == peeled {
            return Ok(sha.to_owned());
        }
        if ref_name == plain {
            plain_commit = Some(sha.to_owned());
        }
    }
    plain_commit.ok_or_else(|| format!("tag {tag} not found in {repository}"))
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn short_hash(value: &str) -> &str {
    &value[..value.len().min(12)]
}

fn run_buf(dir: &Path, args: &[&str]) -> Result<(), String> {
    let status = Command::new("buf")
        .args(args)
        .current_dir(dir)
        .status()
        .map_err(|e| format!("failed to run `buf` (install from https://buf.build): {e}"))?;
    if !status.success() {
        return Err(format!("`buf {}` failed", args.join(" ")));
    }
    Ok(())
}

fn fetch_url(url: &str) -> Result<Vec<u8>, String> {
    let output = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .map_err(|e| format!("failed to run curl: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "fetch {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Regenerates the l1 golden embeddings from real weights by running the
/// l1_models suite with LUMEN_GOLDEN_WRITE=1. Review the resulting diff under
/// crates/lumen-hub/tests/golden/ before committing.
fn golden(args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let mut models_dir: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--models-dir" => {
                models_dir = Some(args.next().ok_or("missing value for --models-dir")?);
            }
            other => return Err(format!("unknown golden argument `{other}`")),
        }
    }

    let mut command = Command::new("cargo");
    command.args([
        "test",
        "-p",
        "lumen-hub",
        "--release",
        "--test",
        "l1_models",
        "--",
        "--test-threads=1",
    ]);
    command.env("LUMEN_GOLDEN_WRITE", "1");
    if let Some(dir) = models_dir {
        command.env("LUMEN_MODELS_DIR", dir);
    }
    let status = command
        .status()
        .map_err(|e| format!("failed to run cargo test: {e}"))?;
    if !status.success() {
        return Err("golden regeneration run failed".to_owned());
    }
    println!("golden files updated under crates/lumen-hub/tests/golden/ — review the diff");
    Ok(())
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

    #[test]
    fn parses_data_plane_major_from_package() {
        assert_eq!(
            data_plane_major_from_content("package home_native.v1;\n").unwrap(),
            1
        );
        assert_eq!(
            data_plane_major_from_content("package lumen.control.v2;\n").unwrap(),
            2
        );
        assert!(data_plane_major_from_content("package home_native;\n").is_err());
        assert!(data_plane_major_from_content("syntax = \"proto3\";\n").is_err());
    }

    #[test]
    fn parses_provenance_without_unknown_fields() {
        let raw = r#"{
            "schemaVersion": 1,
            "mlService": {
                "authority": "EdwinZhanCN/lumen-sdk",
                "tag": "v1.3.2",
                "commit": "9514d11c954abdaba8750acbc5054602cefb3eed",
                "sha256": "d5a2f6fe8322a453b2f97bc50123d3fbcea2ae2655321272d63b673b28290f3f"
            }
        }"#;
        let provenance: Provenance = serde_json::from_str(raw).expect("parse provenance");
        assert_eq!(provenance.schema_version, 1);
        assert_eq!(provenance.ml_service.tag, "v1.3.2");
        assert_eq!(provenance.ml_service.sha256.len(), 64);
    }

    #[test]
    fn rejects_unknown_provenance_shape() {
        let raw = r#"{"schemaVersion": 2, "mlService": {}}"#;
        assert!(serde_json::from_str::<Provenance>(raw).is_err());
    }

    #[test]
    fn zip_marks_binaries_executable() {
        let root = env::temp_dir().join(format!("lumen-xtask-zip-{}", std::process::id()));
        let staging = root.join("lumen-hub-test");
        fs::create_dir_all(staging.join("bin")).unwrap();
        fs::write(staging.join("bin/lumen-hub"), b"binary").unwrap();
        fs::write(staging.join("README.md"), b"readme").unwrap();

        let archive = root.join("lumen-hub-test.zip");
        zip_directory(&staging, &archive).unwrap();

        let file = File::open(&archive).unwrap();
        let mut zip = zip::ZipArchive::new(file).unwrap();
        let binary_mode = zip
            .by_name("lumen-hub-test/bin/lumen-hub")
            .unwrap()
            .unix_mode()
            .unwrap();
        let readme_mode = zip
            .by_name("lumen-hub-test/README.md")
            .unwrap()
            .unix_mode()
            .unwrap();

        assert_eq!(binary_mode & 0o777, 0o755);
        assert_eq!(readme_mode & 0o777, 0o644);
        fs::remove_dir_all(root).unwrap();
    }
}
