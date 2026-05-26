use std::{env, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let proto_file = "proto/ml_service.proto";
    let proto_dir = "proto";
    let out_dir = "src/daemon";
    let protoc = protoc_bin_vendored::protoc_bin_path()?;

    println!("cargo:rerun-if-changed={proto_file}");
    println!("cargo:rerun-if-changed={proto_dir}");

    link_linux_ort_cxx_runtime();

    let mut prost_config = tonic_prost_build::Config::new();
    prost_config.protoc_executable(protoc);

    tonic_prost_build::configure()
        .out_dir(out_dir)
        .include_file("proto.rs")
        .compile_with_config(prost_config, &[proto_file], &[proto_dir])?;

    Ok(())
}

fn link_linux_ort_cxx_runtime() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").ok();
    if target_os.as_deref() != Some("linux") {
        return;
    }

    let has_ort = env::var_os("CARGO_FEATURE_ORT_DOWNLOAD_BINARIES").is_some()
        || env::var_os("CARGO_FEATURE_ORT_LOAD_DYNAMIC").is_some()
        || env::var_os("CARGO_FEATURE_ORT_PKG_CONFIG").is_some()
        || env::var_os("CARGO_FEATURE_ORT_CUDA").is_some()
        || env::var_os("CARGO_FEATURE_ORT_TENSORRT").is_some()
        || env::var_os("CARGO_FEATURE_ORT_OPENVINO").is_some()
        || env::var_os("CARGO_FEATURE_ORT_XNNPACK").is_some();

    if !has_ort {
        return;
    }

    println!("cargo:rustc-link-arg-bins=-lstdc++");
    println!("cargo:rustc-link-arg-bins=-lsupc++");
}
