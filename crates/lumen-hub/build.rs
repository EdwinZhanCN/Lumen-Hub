use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let proto_file = "proto/ml_service.proto";
    let proto_dir = "proto";
    let out_dir = "src/daemon";

    println!("cargo:rerun-if-changed={proto_file}");
    println!("cargo:rerun-if-changed={proto_dir}");

    tonic_prost_build::configure()
        .out_dir(out_dir)
        .include_file("proto.rs")
        .compile_protos(&[proto_file], &[proto_dir])?;

    Ok(())
}
