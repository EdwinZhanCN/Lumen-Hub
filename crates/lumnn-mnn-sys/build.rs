// Derived from rust-paddle-ocr
// Original project: https://github.com/zibo-chen/rust-paddle-ocr
// Licensed under the Apache License, Version 2.0
// Modifications made for this project.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// MNN prebuilt version to download from GitHub releases
const MNN_PREBUILT_VERSION: &str = "3.5.0-lumnn.dyn.2";
const MNN_PREBUILT_REPO: &str = "EdwinZhanCN/MNN";

fn main() {
    // 在 docs.rs 构建环境中，跳过所有 C++ 编译
    if env::var("DOCS_RS").is_ok() || env::var("CARGO_FEATURE_DOCSRS").is_ok() {
        println!("cargo:warning=Building for docs.rs, skipping C++ compilation");
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    let manifest_dir_path = PathBuf::from(&manifest_dir);

    let (mnn_include_dir, mnn_lib_dir) = if let Ok(lib_dir_str) = env::var("MNN_LIB_DIR") {
        let lib_dir = PathBuf::from(&lib_dir_str);
        if !lib_dir.exists() {
            panic!("MNN_LIB_DIR='{}' does not exist", lib_dir.display());
        }

        verify_dynamic_libraries(&lib_dir, &os);

        let include_dirs = get_mnn_include_dirs(&manifest_dir_path);

        println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
        println!("cargo:rerun-if-env-changed=MNN_INCLUDE_DIR");
        println!(
            "cargo:warning=Using MNN dynamic libraries from MNN_LIB_DIR: {}",
            lib_dir.display()
        );

        (include_dirs, vec![lib_dir])
    } else {
        let asset_name = get_prebuilt_asset_name(&os, &arch).unwrap_or_else(|| {
            panic!(
                "No prebuilt MNN available for {os}/{arch}.\n\
                 Set MNN_LIB_DIR to a directory containing dynamic MNN libraries."
            )
        });
        let prebuilt_dir = download_prebuilt_mnn(&manifest_dir_path, &asset_name, &os);

        let include_dir = prebuilt_dir.join("include");
        let lib_dir = prebuilt_dir.join("lib");

        if !include_dir.exists() {
            panic!(
                "Prebuilt MNN include directory not found: {}",
                include_dir.display()
            );
        }
        if !lib_dir.exists() {
            panic!(
                "Prebuilt MNN lib directory not found: {}",
                lib_dir.display()
            );
        }

        verify_dynamic_libraries(&lib_dir, &os);

        println!("cargo:warning=Using prebuilt MNN {MNN_PREBUILT_VERSION} for {os}/{arch}");

        (vec![include_dir], vec![lib_dir])
    };

    build_wrapper(&manifest_dir_path, &mnn_include_dir, &os);
    link_libraries(&mnn_lib_dir, &os);
    bind_gen(&manifest_dir_path, &mnn_include_dir, &os, &arch);
}

/// Get MNN include directories when using a custom MNN_LIB_DIR.
/// Priority:
/// 1. MNN_INCLUDE_DIR environment variable
/// 2. MNN_SOURCE_DIR/include (if MNN_SOURCE_DIR is set)
/// 3. Local 3rd_party/MNN/include
/// 4. Bundled prebuilt headers (when MNN_LIB_DIR is set)
fn get_mnn_include_dirs(manifest_dir: &Path) -> Vec<PathBuf> {
    if let Ok(include_dir) = env::var("MNN_INCLUDE_DIR") {
        let include_path = PathBuf::from(&include_dir);
        if include_path.exists() {
            println!(
                "cargo:warning=Using MNN headers from MNN_INCLUDE_DIR: {}",
                include_path.display()
            );
            return vec![include_path];
        }
        panic!(
            "MNN_INCLUDE_DIR='{}' does not exist",
            include_path.display()
        );
    }

    if let Ok(mnn_dir) = env::var("MNN_SOURCE_DIR") {
        let mnn_path = PathBuf::from(&mnn_dir);
        if mnn_path.exists() {
            println!(
                "cargo:warning=Using MNN headers from MNN_SOURCE_DIR: {}",
                mnn_path.display()
            );
            return mnn_source_include_dirs(manifest_dir, &mnn_path);
        }
        panic!("MNN_SOURCE_DIR='{}' does not exist", mnn_path.display());
    }

    let local_mnn = manifest_dir.join("3rd_party/MNN");
    let local_include = local_mnn.join("include");
    if local_include.exists() {
        println!(
            "cargo:warning=Using MNN headers from local source: {}",
            local_include.display()
        );
        return mnn_source_include_dirs(manifest_dir, &local_mnn);
    }

    if env::var("MNN_LIB_DIR").is_ok() {
        if let Some(prebuilt_include) = find_bundled_prebuilt_include(manifest_dir) {
            println!(
                "cargo:warning=Using bundled prebuilt MNN headers with custom MNN_LIB_DIR: {}",
                prebuilt_include.display()
            );
            return vec![prebuilt_include];
        }
    }

    panic!(
        "MNN headers not found. Please set one of:\n\
         - MNN_INCLUDE_DIR: path to directory containing MNN headers\n\
         - MNN_SOURCE_DIR: path to MNN source tree\n\
         Or ensure 3rd_party/MNN exists in the project root."
    );
}

fn find_bundled_prebuilt_include(manifest_dir: &Path) -> Option<PathBuf> {
    let prebuilt_root = manifest_dir.join("3rd_party/prebuilt");
    if !prebuilt_root.is_dir() {
        return None;
    }

    let mut candidates = Vec::new();
    if let Ok(entries) = fs::read_dir(&prebuilt_root) {
        for entry in entries.flatten() {
            let include_dir = entry.path().join("include");
            if include_dir.join("MNN/Interpreter.hpp").is_file() {
                candidates.push(include_dir);
            }
        }
    }

    candidates.sort();
    candidates.into_iter().next()
}

fn mnn_source_include_dirs(build_or_manifest_dir: &Path, mnn_source_dir: &Path) -> Vec<PathBuf> {
    let mut include_dirs = Vec::new();

    let build_include = build_or_manifest_dir.join("include");
    if build_include.exists() {
        include_dirs.push(build_include);
    }

    let source_include = mnn_source_dir.join("include");
    if source_include.exists() {
        include_dirs.push(source_include);
    }

    include_dirs
}

fn get_prebuilt_asset_name(os: &str, arch: &str) -> Option<String> {
    let suffix = match (os, arch) {
        ("linux", "x86_64") => "linux-x86_64",
        ("linux", "aarch64") => "linux-aarch64",
        ("windows", "x86_64") => "windows-x86_64",
        ("macos", _) => "macos-universal",
        _ => return None,
    };
    Some(format!("mnn-{MNN_PREBUILT_VERSION}-{suffix}"))
}

fn download_prebuilt_mnn(manifest_dir: &Path, asset_name: &str, os: &str) -> PathBuf {
    let cache_dir = manifest_dir.join("3rd_party").join("prebuilt");
    let extract_dir = cache_dir.join(asset_name);

    if extract_dir.join("lib").exists() && extract_dir.join("include").exists() {
        println!(
            "cargo:warning=Using cached prebuilt MNN from: {}",
            extract_dir.display()
        );
        remove_static_libraries(&extract_dir);
        return extract_dir;
    }

    fs::create_dir_all(&cache_dir).expect("Failed to create prebuilt cache directory");

    let (ext, url) = if os == "windows" {
        (
            "zip",
            format!(
                "https://github.com/{MNN_PREBUILT_REPO}/releases/download/{MNN_PREBUILT_VERSION}/{asset_name}.zip"
            ),
        )
    } else {
        (
            "tar.gz",
            format!(
                "https://github.com/{MNN_PREBUILT_REPO}/releases/download/{MNN_PREBUILT_VERSION}/{asset_name}.tar.gz"
            ),
        )
    };

    let archive_path = cache_dir.join(format!("{asset_name}.{ext}"));

    if !archive_path.exists() {
        println!("cargo:warning=Downloading prebuilt MNN from: {url}");
        download_file(&url, &archive_path);
    }

    println!(
        "cargo:warning=Extracting prebuilt MNN to: {}",
        extract_dir.display()
    );

    if os == "windows" {
        extract_zip(&archive_path, &cache_dir);
    } else {
        extract_tar_gz(&archive_path, &cache_dir);
    }

    if !extract_dir.join("lib").exists() {
        panic!(
            "Prebuilt MNN extraction failed: lib/ not found in {}",
            extract_dir.display()
        );
    }

    remove_static_libraries(&extract_dir);
    extract_dir
}

fn verify_dynamic_libraries(lib_dir: &Path, os: &str) {
    let required: &[&str] = match os {
        "macos" | "ios" => &["libMNN.dylib"],
        "linux" => &["libMNN.so"],
        "windows" => &["MNN.dll", "MNN.lib"],
        _ => &["libMNN.so"],
    };

    for name in required {
        let path = lib_dir.join(name);
        if !path.is_file() {
            panic!(
                "Required MNN dynamic library not found: {}.\n\
                 Expected dynamic-only prebuilts tagged `{}`.",
                path.display(),
                MNN_PREBUILT_VERSION
            );
        }
    }
}

fn remove_static_libraries(extract_dir: &Path) {
    let lib_dir = extract_dir.join("lib");
    let Ok(entries) = fs::read_dir(&lib_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        let is_static = name.ends_with(".a")
            || name.ends_with("_static.lib")
            || (name.ends_with(".lib") && !matches!(name, "MNN.lib" | "MNN_Express.lib"));

        if is_static {
            let _ = fs::remove_file(&path);
        }
    }
}

fn download_file(url: &str, dest: &Path) {
    let status = Command::new("curl")
        .args(["-L", "-f", "-s", "-o"])
        .arg(dest.to_str().unwrap())
        .arg(url)
        .status();

    if matches!(status, Ok(s) if s.success()) {
        return;
    }

    if cfg!(target_os = "windows") {
        let ps_cmd = format!(
            "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
            url,
            dest.to_str().unwrap()
        );
        let status = Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps_cmd])
            .status();
        if matches!(status, Ok(s) if s.success()) {
            return;
        }
    }

    panic!(
        "Failed to download {}. Please ensure curl is available, \
         or download manually to: {}",
        url,
        dest.display()
    );
}

fn extract_tar_gz(archive: &Path, dest_dir: &Path) {
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(archive.to_str().unwrap())
        .args(["-C"])
        .arg(dest_dir.to_str().unwrap())
        .status()
        .expect("Failed to run tar");

    if !status.success() {
        panic!("Failed to extract {}", archive.display());
    }
}

fn extract_zip(archive: &Path, dest_dir: &Path) {
    if cfg!(target_os = "windows") {
        let ps_cmd = format!(
            "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
            archive.to_str().unwrap(),
            dest_dir.to_str().unwrap()
        );
        let status = Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps_cmd])
            .status()
            .expect("Failed to run powershell");
        if !status.success() {
            panic!("Failed to extract {}", archive.display());
        }
    } else {
        let status = Command::new("unzip")
            .args(["-o", "-q"])
            .arg(archive.to_str().unwrap())
            .args(["-d"])
            .arg(dest_dir.to_str().unwrap())
            .status()
            .expect("Failed to run unzip");
        if !status.success() {
            panic!("Failed to extract {}", archive.display());
        }
    }
}

fn build_wrapper(manifest_dir: &Path, mnn_include_dirs: &[PathBuf], os: &str) {
    let wrapper_file = manifest_dir.join("cpp/src/mnn_wrapper.cpp");

    println!("cargo:rerun-if-changed=cpp/src/mnn_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/include/mnn_wrapper.h");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file(&wrapper_file)
        .include(manifest_dir.join("cpp/include"));

    for inc in mnn_include_dirs {
        build.include(inc);
    }

    if os == "windows" {
        build.flag("/std:c++14").flag("/EHsc").flag("/W3");
    } else {
        build.flag("-std=c++14").flag("-fvisibility=hidden");
    }

    build.compile("mnn_wrapper");
}

fn link_libraries(lib_dirs: &[PathBuf], os: &str) {
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    println!("cargo:rustc-link-lib=dylib=MNN");
    link_optional_dynamic_library(lib_dirs, os, "MNN_Express");

    match os {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        "android" => {
            println!("cargo:rustc-link-lib=c++_shared");
            println!("cargo:rustc-link-lib=log");
        }
        "windows" => {}
        _ => {}
    }
}

fn link_optional_dynamic_library(lib_dirs: &[PathBuf], os: &str, library: &str) {
    let has_library = lib_dirs
        .iter()
        .any(|dir| dynamic_library_path(dir, os, library).exists());

    if has_library {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}

fn dynamic_library_path(lib_dir: &Path, os: &str, library: &str) -> PathBuf {
    match os {
        "windows" => lib_dir.join(format!("{library}.lib")),
        "macos" | "ios" => lib_dir.join(format!("lib{library}.dylib")),
        _ => lib_dir.join(format!("lib{library}.so")),
    }
}

fn bind_gen(manifest_dir: &Path, mnn_include_dirs: &[PathBuf], os: &str, arch: &str) {
    let header_path = manifest_dir.join("cpp/include/mnn_wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .allowlist_function("mnnr_.*")
        .allowlist_type("MNN.*")
        .allowlist_type("MNNR.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false);

    for inc in mnn_include_dirs {
        builder = builder.clang_arg(format!("-I{}", inc.display()));
    }

    if os == "linux" {
        builder = add_linux_system_include_args(builder);
    }

    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .unwrap_or_default();

        let api_level = "21";
        let target = match arch {
            "aarch64" => "aarch64-linux-android",
            "arm" => "armv7-linux-androideabi",
            "x86_64" => "x86_64-linux-android",
            "x86" => "i686-linux-android",
            _ => "aarch64-linux-android",
        };
        builder = builder.clang_arg(format!("--target={target}{api_level}"));

        if !ndk.is_empty() {
            let host_tag = if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else {
                "linux-x86_64"
            };
            let sysroot = PathBuf::from(&ndk)
                .join("toolchains/llvm/prebuilt")
                .join(host_tag)
                .join("sysroot");
            if sysroot.exists() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot.display()));
            }
        }
    }

    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let clang_target = if rust_target == "aarch64-apple-ios-sim" {
            "arm64-apple-ios13.0-simulator".to_string()
        } else if rust_target == "aarch64-apple-ios" {
            "arm64-apple-ios13.0".to_string()
        } else if rust_target == "x86_64-apple-ios" {
            "x86_64-apple-ios13.0-simulator".to_string()
        } else {
            rust_target
        };
        builder = builder.clang_arg(format!("--target={clang_target}"));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_path.join("mnn_bindings.rs"), bindings.to_string())
        .expect("Couldn't write bindings!");
}

fn add_linux_system_include_args(mut builder: bindgen::Builder) -> bindgen::Builder {
    let mut include_dirs = Vec::new();
    let mut seen = HashSet::new();

    let compiler = cc::Build::new().get_compiler();
    let compiler_path = compiler.path();

    if let Some(include_dir) = command_path_output(compiler_path, &["-print-file-name=include"]) {
        push_unique_path(&mut include_dirs, &mut seen, PathBuf::from(include_dir));
    }

    let sysroot = command_path_output(compiler_path, &["-print-sysroot"])
        .filter(|value| !value.is_empty() && value != "/");

    let target_include = command_path_output(compiler_path, &["-dumpmachine"])
        .map(PathBuf::from)
        .or_else(|| env::var("TARGET").ok().map(PathBuf::from));

    if let Some(sysroot) = sysroot.as_ref() {
        let sysroot_path = PathBuf::from(sysroot);
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            sysroot_path.join("usr/local/include"),
        );
        if let Some(target) = target_include.as_ref() {
            push_unique_path(
                &mut include_dirs,
                &mut seen,
                sysroot_path.join("usr/include").join(target),
            );
        }
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            sysroot_path.join("usr/include"),
        );
    }

    push_unique_path(
        &mut include_dirs,
        &mut seen,
        PathBuf::from("/usr/local/include"),
    );
    if let Some(target) = target_include.as_ref() {
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            PathBuf::from("/usr/include").join(target),
        );
    }
    push_unique_path(&mut include_dirs, &mut seen, PathBuf::from("/usr/include"));

    for dir in include_dirs {
        println!(
            "cargo:warning=Adding Linux system include for bindgen: {}",
            dir.display()
        );
        builder = builder.clang_arg(format!("-isystem{}", dir.display()));
    }

    builder
}

fn command_path_output(program: &std::path::Path, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn push_unique_path(paths: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    if path.exists() && seen.insert(path.clone()) {
        paths.push(path);
    }
}
