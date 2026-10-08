//! Builds the IREE runtime + Lumen shim with CMake/Ninja and links it statically.
//!
//! Inputs:
//!   LUMEN_IREE_SOURCE_DIR  pinned IREE checkout (default: <workspace>/third_party/iree,
//!                          populated by `cargo xtask iree-fetch`)
//!   CC / CXX               honored by CMake as usual
//! Requirements: cmake >= 3.21 and ninja on PATH; driver-cuda additionally needs
//! the CUDA Toolkit >= 12 (headers only; libcuda is loaded at run time).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to spawn {command:?}: {error}"));
    assert!(status.success(), "command failed ({status}): {command:?}");
}

fn iree_source_dir(manifest_dir: &Path) -> PathBuf {
    println!("cargo:rerun-if-env-changed=LUMEN_IREE_SOURCE_DIR");
    let dir = env::var_os("LUMEN_IREE_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../third_party/iree"));
    let marker = dir.join("runtime/src/iree/runtime/api.h");
    let flatcc = dir.join("third_party/flatcc/include/flatcc/flatcc_verifier.h");
    if !marker.is_file() || !flatcc.is_file() {
        panic!(
            "IREE source not found at {} (need runtime/ and third_party/flatcc). \
             Run `cargo xtask iree-fetch` or set LUMEN_IREE_SOURCE_DIR.",
            dir.display()
        );
    }
    dir
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let iree = iree_source_dir(&manifest_dir);
    for file in ["CMakeLists.txt", "lumen_iree.c", "lumen_iree.h"] {
        println!("cargo:rerun-if-changed=csrc/{file}");
    }

    let mut drivers = vec!["local-task"];
    if env::var_os("CARGO_FEATURE_DRIVER_CUDA").is_some() {
        drivers.push("cuda");
    }
    if env::var_os("CARGO_FEATURE_DRIVER_METAL").is_some() {
        assert_eq!(
            target_os, "macos",
            "driver-metal is only supported on macOS"
        );
        drivers.push("metal");
    }

    let build_dir = out_dir.join("cmake");
    run(Command::new("cmake")
        .arg("-G")
        .arg("Ninja")
        .arg("-S")
        .arg(manifest_dir.join("csrc"))
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DIREE_SOURCE_DIR={}", iree.display()))
        .arg(format!("-DLUMEN_IREE_DRIVERS={}", drivers.join(";"))));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("lumen_iree"));

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=lumen_iree");
    println!("cargo:include={}", manifest_dir.join("csrc").display());

    // System dependencies reported by CMake (frameworks, -l flags, plain names).
    let libs = fs::read_to_string(build_dir.join("lumen_iree_link_libs.txt"))
        .expect("CMake did not write lumen_iree_link_libs.txt");
    for entry in libs.trim().split(';').map(str::trim) {
        if entry.is_empty()
            || entry.contains("::")
            || entry.starts_with("iree_")
            || entry.starts_with("flatcc")
        {
            continue;
        }
        if let Some(framework) = entry.strip_prefix("-framework ") {
            println!("cargo:rustc-link-lib=framework={}", framework.trim());
        } else if let Some(name) = entry.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={name}");
        } else {
            println!(
                "cargo:rustc-link-lib=dylib={}",
                entry.trim_end_matches(".lib")
            );
        }
    }
    match target_os.as_str() {
        "linux" => {
            for lib in ["dl", "pthread", "m", "rt"] {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
        }
        "macos" if drivers.contains(&"metal") => {
            for framework in ["Foundation", "Metal", "CoreGraphics"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
            println!("cargo:rustc-link-lib=dylib=objc");
        }
        _ => {}
    }
}
