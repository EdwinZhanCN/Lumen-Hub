use std::ffi::OsString;

use lumen_schema::{ConfigTarget, Preset, RenderOptions, preset_yaml};

pub fn is_render_command(args: &[OsString]) -> bool {
    args.get(1).and_then(|arg| arg.to_str()) == Some("config")
        && args.get(2).and_then(|arg| arg.to_str()) == Some("render")
}

pub fn run_render_command(args: Vec<OsString>) -> Result<String, String> {
    let mut target = None;
    let mut preset = None;
    let mut region = None;
    let mut cache_dir = None;
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
            "--preset" => {
                preset = Some(Preset::by_name(&value).ok_or_else(|| {
                    format!("unsupported preset `{value}`; expected minimal, basic, or brave")
                })?)
            }
            "--region" => region = Some(value),
            "--cache-dir" => cache_dir = Some(value),
            _ => {
                return Err(format!(
                    "unknown config render option `{flag}`\n\n{}",
                    usage()
                ));
            }
        }
    }

    let target = target.ok_or_else(|| "missing `--target <desktop|network>`".to_owned())?;
    let preset = preset.ok_or_else(|| "missing `--preset <minimal|basic|brave>`".to_owned())?;
    let region = region.ok_or_else(|| "missing `--region <other|cn>`".to_owned())?;
    let cache_dir = cache_dir.ok_or_else(|| "missing `--cache-dir <path>`".to_owned())?;
    preset_yaml(
        preset,
        &RenderOptions {
            region: &region,
            cache_dir: &cache_dir,
            target,
        },
    )
}

pub fn usage() -> &'static str {
    "Usage:\n  lumen-hub config render --target <desktop|network> --preset <minimal|basic|brave> --region <other|cn> --cache-dir <path>"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_render_is_loopback_only() {
        let yaml = run_render_command(
            [
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
            ]
            .into_iter()
            .map(OsString::from)
            .collect(),
        )
        .unwrap();
        let config: lumen_schema::LumenConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.metadata.cache_dir, "/tmp/models");
    }

    #[test]
    fn rejects_incomplete_or_unknown_intent() {
        assert!(
            run_render_command(
                ["lumen-hub", "config", "render"]
                    .into_iter()
                    .map(OsString::from)
                    .collect()
            )
            .is_err()
        );
    }
}
