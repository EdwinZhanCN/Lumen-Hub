use std::path::Path;

use lumen_hub::docker_config::DockerConfigInput;
use lumen_launcher::setup::{Backend, REGION_OTHER, render_config};
use lumen_schema::{LumenConfig, Preset};

fn docker_base_config() -> LumenConfig {
    serde_yaml::from_str(include_str!(
        "../../../packaging/docker/config.default.yaml"
    ))
    .expect("Docker base config parses")
}

#[test]
fn cli_and_docker_render_the_same_canonical_presets() {
    for preset in Preset::all() {
        let cli = serde_yaml::from_str::<LumenConfig>(&render_config(
            *preset,
            REGION_OTHER,
            Backend::cpu("linux-x64-cpu"),
            Path::new("/tmp/lumen"),
        ))
        .expect("CLI preset config parses");

        let mut docker = docker_base_config();
        DockerConfigInput::for_preset(preset.name)
            .apply(&mut docker)
            .expect("Docker preset applies");

        assert_eq!(cli.deployment_service_names(), preset.components);
        assert_eq!(docker.deployment_service_names(), preset.components);

        for service in preset.components {
            let cli_model = &cli.services[*service].models["default"];
            let docker_model = &docker.services[*service].models["default"];
            assert_eq!(
                cli_model.model, docker_model.model,
                "{} preset model drift for {service}",
                preset.name
            );
            assert_eq!(
                cli_model.dataset, docker_model.dataset,
                "{} preset dataset drift for {service}",
                preset.name
            );

            // The committed fixture is the canonical render for this preset;
            // every entry point must agree with it on selected services.
            let fixture = serde_yaml::from_str::<LumenConfig>(match preset.name {
                "minimal" => include_str!("../../../fixtures/config/minimal.yaml"),
                "basic" => include_str!("../../../fixtures/config/basic.yaml"),
                "brave" => include_str!("../../../fixtures/config/brave.yaml"),
                other => panic!("unexpected preset {other}"),
            })
            .expect("fixture parses");
            let fixture_model = &fixture.services[*service].models["default"];
            assert_eq!(
                cli_model.model, fixture_model.model,
                "{} preset fixture model drift for {service}",
                preset.name
            );
            assert_eq!(
                cli_model.dataset, fixture_model.dataset,
                "{} preset fixture dataset drift for {service}",
                preset.name
            );
        }
    }
}
