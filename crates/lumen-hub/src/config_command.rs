use std::ffi::OsString;

use lumen_schema::{
    ConfigTarget, Preset, RenderOptions, custom_yaml, intern_services, preset_yaml,
};

pub fn is_render_command(args: &[OsString]) -> bool {
    args.get(1).and_then(|arg| arg.to_str()) == Some("config")
        && args.get(2).and_then(|arg| arg.to_str()) == Some("render")
}

pub fn run_render_command(args: Vec<OsString>) -> Result<String, String> {
    let mut target = None;
    let mut preset_name = None;
    let mut region = None;
    let mut cache_dir = None;
    let mut services = None;
    let mut siglip_model = None;
    let mut bioclip_dataset = None;
    let mut iter = args.into_iter().skip(3);
    while let Some(raw) = iter.next() {
        let flag = raw
            .into_string()
            .map_err(|_| "arguments must be valid UTF-8".to_owned())?;
        if flag == "-h" || flag == "--help" {
            return Err(usage().to_owned());
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for `{flag}`"))?
            .into_string()
            .map_err(|_| format!("value for `{flag}` must be valid UTF-8"))?;
        match flag.as_str() {
            "--target" => target = Some(ConfigTarget::parse(&value)?),
            "--preset" => preset_name = Some(value),
            "--region" => region = Some(value),
            "--cache-dir" => cache_dir = Some(value),
            "--services" => services = Some(value),
            "--siglip-model" => siglip_model = Some(value),
            "--bioclip-dataset" => bioclip_dataset = Some(value),
            _ => {
                return Err(format!(
                    "unknown config render option `{flag}`\n\n{}",
                    usage()
                ));
            }
        }
    }

    let target = target.ok_or_else(|| "missing `--target <desktop|network>`".to_owned())?;
    let preset_name =
        preset_name.ok_or_else(|| "missing `--preset <minimal|basic|brave|custom>`".to_owned())?;
    let region = region.ok_or_else(|| "missing `--region <other|cn>`".to_owned())?;
    let cache_dir = cache_dir.ok_or_else(|| "missing `--cache-dir <path>`".to_owned())?;
    let options = RenderOptions {
        region: &region,
        cache_dir: &cache_dir,
        target,
    };

    if preset_name == "custom" {
        let services = services.ok_or_else(|| {
            "missing `--services <siglip,face,ocr,bioclip>` for `--preset custom`".to_owned()
        })?;
        let services = intern_services(services.split(',').map(str::trim))?;
        return custom_yaml(
            &services,
            siglip_model.as_deref(),
            bioclip_dataset.as_deref(),
            &options,
        );
    }

    if services.is_some() || siglip_model.is_some() || bioclip_dataset.is_some() {
        return Err(
            "`--services`, `--siglip-model`, and `--bioclip-dataset` are only valid with `--preset custom`"
                .to_owned(),
        );
    }
    let preset = Preset::by_name(&preset_name).ok_or_else(|| {
        format!("unsupported preset `{preset_name}`; expected minimal, basic, brave, or custom")
    })?;
    preset_yaml(preset, &options)
}

pub fn usage() -> &'static str {
    "Usage:\n  lumen-hub config render --target <desktop|network> --preset <minimal|basic|brave|custom> --region <other|cn> --cache-dir <path> [--services <siglip,face,ocr,bioclip>] [--siglip-model <model>] [--bioclip-dataset <dataset>]"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<OsString> {
        parts.iter().map(OsString::from).collect()
    }

    #[test]
    fn desktop_render_is_loopback_only() {
        let yaml = run_render_command(args(&[
            "lumen-hub",
            "config",
            "render",
            "--target",
            "desktop",
            "--preset",
            "basic",
            "--region",
            "other",
            "--cache-dir",
            "/tmp/models",
        ]))
        .unwrap();
        let config: lumen_schema::LumenConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.metadata.cache_dir, "/tmp/models");
    }

    #[test]
    fn custom_render_selects_services_and_overrides() {
        let yaml = run_render_command(args(&[
            "lumen-hub",
            "config",
            "render",
            "--target",
            "network",
            "--preset",
            "custom",
            "--services",
            "bioclip,siglip",
            "--siglip-model",
            "siglip2-so400m-patch14-384",
            "--bioclip-dataset",
            "TreeOfLife200M",
            "--region",
            "cn",
            "--cache-dir",
            "/tmp/models",
        ]))
        .unwrap();
        let config: lumen_schema::LumenConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.deployment_service_names(), vec!["siglip", "bioclip"]);
        assert_eq!(
            config.services["siglip"].models["default"].model,
            "siglip2-so400m-patch14-384"
        );
        assert_eq!(
            config.services["bioclip"].models["default"]
                .dataset
                .as_deref(),
            Some("TreeOfLife200M")
        );
        assert_eq!(config.server.host, "0.0.0.0");
    }

    #[test]
    fn custom_flags_are_rejected_on_official_presets() {
        assert!(
            run_render_command(args(&[
                "lumen-hub",
                "config",
                "render",
                "--target",
                "network",
                "--preset",
                "basic",
                "--services",
                "siglip",
                "--region",
                "other",
                "--cache-dir",
                "/tmp/models",
            ]))
            .is_err()
        );
    }

    #[test]
    fn rejects_incomplete_or_unknown_intent() {
        assert!(run_render_command(args(&["lumen-hub", "config", "render"])).is_err());
    }
}
