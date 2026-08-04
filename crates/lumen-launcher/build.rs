use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let proto_file = "../lumen-hub/proto/control.proto";
    let proto_dir = "../lumen-hub/proto";
    let protoc = protoc_bin_vendored::protoc_bin_path()?;

    println!("cargo:rerun-if-changed={proto_file}");

    let mut prost_config = tonic_prost_build::Config::new();
    prost_config.protoc_executable(protoc);

    tonic_prost_build::configure()
        .build_server(false)
        .compile_with_config(prost_config, &[proto_file], &[proto_dir])?;
    Ok(())
}
