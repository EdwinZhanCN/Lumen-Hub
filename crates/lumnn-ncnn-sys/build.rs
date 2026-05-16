use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn main() {
    println!("cargo:rerun-if-env-changed=NCNN_DIR");
    println!("cargo:rerun-if-env-changed=NCNN_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=Vulkan_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=Vulkan_LIBRARY");

    let vendored = feature_enabled("VENDORED");
    let system = feature_enabled("SYSTEM");
    let static_link = feature_enabled("STATIC");
    let shared_link = feature_enabled("SHARED");
    let vulkan = feature_enabled("VULKAN");

    if vendored && system {
        panic!("features `vendored` and `system` are mutually exclusive");
    }
    if static_link && shared_link {
        panic!("features `static` and `shared` are mutually exclusive");
    }

    if system {
        link_system(static_link, vulkan);
        return;
    }

    build_vendored(static_link, shared_link, vulkan);
}

fn build_vendored(static_link: bool, shared_link: bool, vulkan: bool) {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = env::var_os("NCNN_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("third_party").join("ncnn"));

    if !source_dir.join("CMakeLists.txt").exists() {
        panic!(
            "ncnn source tree not found at `{}`. Set NCNN_SOURCE_DIR to an ncnn checkout, \
             or add ncnn as crates/lumnn-ncnn-sys/third_party/ncnn.",
            source_dir.display()
        );
    }

    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("CMakeLists.txt").display()
    );

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let build_dir = out_dir.join("ncnn-build");
    let install_dir = out_dir.join("ncnn-install");
    let build_type = cmake_build_type();

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&source_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_BUILD_TYPE={build_type}"))
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()))
        .arg("-DNCNN_C_API=ON")
        .arg("-DNCNN_STRING=ON")
        .arg("-DNCNN_STDIO=ON")
        .arg("-DNCNN_BUILD_TOOLS=OFF")
        .arg("-DNCNN_BUILD_EXAMPLES=OFF")
        .arg("-DNCNN_BUILD_TESTS=OFF")
        .arg("-DNCNN_BENCHMARK=OFF")
        .arg("-DNCNN_INSTALL_SDK=ON")
        .arg(format!(
            "-DNCNN_SHARED_LIB={}",
            if static_link { "OFF" } else { "ON" }
        ))
        .arg(format!(
            "-DNCNN_OPENMP={}",
            if feature_enabled("OPENMP") {
                "ON"
            } else {
                "OFF"
            }
        ))
        .arg(format!(
            "-DNCNN_VULKAN={}",
            if vulkan { "ON" } else { "OFF" }
        ));

    if command_exists("ninja") {
        configure.arg("-GNinja");
    }

    if vulkan {
        configure_vulkan(&mut configure);
    }

    run(&mut configure, "configure vendored ncnn");

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg(&build_type)
        .arg("--target")
        .arg("install")
        .arg("--parallel");
    run(&mut build, "build vendored ncnn");

    let lib_dir = find_library_dir(&install_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!(
        "cargo:rustc-link-lib={}={}",
        if static_link { "static" } else { "dylib" },
        "ncnn"
    );

    if shared_link || !static_link {
        emit_runtime_search_paths(&lib_dir);
    }

    if vulkan {
        link_vulkan_runtime();
    }
}

fn link_system(static_link: bool, vulkan: bool) {
    let ncnn_dir = env::var_os("NCNN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("NCNN_DIR must be set when feature `system` is enabled"));

    let lib_dir = find_library_dir(&ncnn_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!(
        "cargo:rustc-link-lib={}={}",
        if static_link { "static" } else { "dylib" },
        "ncnn"
    );
    if !static_link {
        emit_runtime_search_paths(&lib_dir);
    }
    if vulkan {
        link_vulkan_runtime();
    }
}

fn configure_vulkan(configure: &mut Command) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    if let Some(sdk) = env::var_os("VULKAN_SDK").map(PathBuf::from) {
        let include = sdk.join("include");
        let library = sdk.join("lib").join("libMoltenVK.dylib");
        if !include.join("vulkan").join("vulkan.h").exists() {
            panic!(
                "VULKAN_SDK is set to `{}`, but include/vulkan/vulkan.h was not found",
                sdk.display()
            );
        }
        if !library.exists() {
            panic!(
                "VULKAN_SDK is set to `{}`, but lib/libMoltenVK.dylib was not found",
                sdk.display()
            );
        }

        configure
            .arg(format!("-DVulkan_INCLUDE_DIR={}", include.display()))
            .arg(format!("-DVulkan_LIBRARY={}", library.display()));
        return;
    }

    let include = env::var_os("Vulkan_INCLUDE_DIR").map(PathBuf::from);
    let library = env::var_os("Vulkan_LIBRARY").map(PathBuf::from);
    match (include, library) {
        (Some(include), Some(library)) => {
            configure
                .arg(format!("-DVulkan_INCLUDE_DIR={}", include.display()))
                .arg(format!("-DVulkan_LIBRARY={}", library.display()));
        }
        _ => {
            panic!(
                "macOS ncnn Vulkan builds require VULKAN_SDK, or both \
                 Vulkan_INCLUDE_DIR and Vulkan_LIBRARY"
            );
        }
    }
}

fn link_vulkan_runtime() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target_os.as_str() {
        "macos" => {
            let moltenvk_lib = if let Some(sdk) = env::var_os("VULKAN_SDK").map(PathBuf::from) {
                sdk.join("lib")
            } else if let Some(library) = env::var_os("Vulkan_LIBRARY").map(PathBuf::from) {
                library
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| panic!("Vulkan_LIBRARY must include a parent directory"))
            } else {
                panic!("macOS ncnn Vulkan builds require VULKAN_SDK or Vulkan_LIBRARY");
            };
            println!("cargo:rustc-link-search=native={}", moltenvk_lib.display());
            println!("cargo:rustc-link-lib=dylib=MoltenVK");
            emit_runtime_search_paths(&moltenvk_lib);
        }
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
        "windows" => {
            println!("cargo:rustc-link-lib=dylib=vulkan-1");
        }
        _ => {}
    }
}

fn emit_runtime_search_paths(path: &Path) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    }
}

fn find_library_dir(root: &Path) -> PathBuf {
    let candidates = [
        root.join("lib"),
        root.join("lib64"),
        root.join("build").join("lib"),
        root.to_path_buf(),
    ];

    candidates
        .into_iter()
        .find(|dir| dir.exists())
        .unwrap_or_else(|| panic!("no library directory found under `{}`", root.display()))
}

fn cmake_build_type() -> String {
    match env::var("PROFILE").as_deref() {
        Ok("debug") => "Debug".to_owned(),
        _ => "Release".to_owned(),
    }
}

fn command_exists(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run(command: &mut Command, label: &str) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to {label}: {err}"));
    if !status.success() {
        panic!("{label} failed with status {status}");
    }
}

fn feature_enabled(name: &str) -> bool {
    let mut key = OsString::from("CARGO_FEATURE_");
    key.push(name);
    env::var_os(key).is_some()
}
