use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let proto_file = "proto/ml_service.proto";
    let proto_dir = "proto";
    let out_dir = "src/daemon";
    let protoc = protoc_bin_vendored::protoc_bin_path()?;

    println!("cargo:rerun-if-changed={proto_file}");
    println!("cargo:rerun-if-changed={proto_dir}");

    let mut prost_config = tonic_prost_build::Config::new();
    prost_config.protoc_executable(protoc);

    tonic_prost_build::configure()
        .out_dir(out_dir)
        .include_file("proto.rs")
        .compile_with_config(prost_config, &[proto_file], &[proto_dir])?;

    Ok(())
}
