//! Schema-versioned Hub release manifest.
//!
//! `HubManifest` is the single release catalog published with every Hub
//! release: capability Exact Terms, preset/custom options, model and dataset
//! options, resource guidance, dist platforms, artifact digests, and protocol
//! provenance. CLI/Launcher/Docker consume the underlying constants directly;
//! Site and Desktop consume this manifest (via the Photos `lumen.lock.json`
//! catalog sync) instead of maintaining parallel constants.
//!
//! The builder pulls everything from the canonical preset/capability tables in
//! this crate; xtask supplies the dist profiles and artifact digests.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::preset::{
    BIOCLIP_DATASETS, BIOCLIP_MODELS, CAPABILITIES, FACE_MODELS, OCR_MODELS, Preset, SERVICE_ORDER,
    SIGLIP_MODELS,
};

/// Current manifest layout. Bump only when a consumer-visible field changes in
/// a breaking way; consumers must reject unknown/newer `schemaVersion`.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// Data-plane protocol major, derived from the `home_native.vN` gRPC package
/// in `ml_service.proto`. Bumping this is a protocol-major release.
pub const DATA_PLANE_MAJOR: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestCapability {
    pub id: String,
    #[serde(rename = "zhCn")]
    pub zh_cn: String,
    #[serde(rename = "en")]
    pub en: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestResources {
    #[serde(rename = "ramGb")]
    pub ram_gb: u64,
    #[serde(rename = "vramGb")]
    pub vram_gb: u64,
    #[serde(rename = "diskGb")]
    pub disk_gb: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestPreset {
    pub id: String,
    pub capabilities: Vec<String>,
    #[serde(rename = "siglipModel")]
    pub siglip_model: String,
    #[serde(rename = "bioclipDataset")]
    pub bioclip_dataset: Option<String>,
    pub resources: ManifestResources,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestPlatform {
    /// User-facing platform id, for example `darwin-arm64` or `linux-x64`.
    pub platform: String,
    /// Dist profile id, for example `linux-x64-cuda`.
    pub profile: String,
    /// Compute backend, for example `metal`, `cpu`, `wgpu`, `cuda`, or `rocm`.
    pub backend: String,
    /// Rust target triple used to build this profile.
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestProtocolFile {
    /// Repository-relative path of the proto source at release time.
    pub path: String,
    /// SHA-256 of the proto file bytes at release time.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestProtocol {
    #[serde(rename = "dataPlaneMajor")]
    pub data_plane_major: u32,
    #[serde(rename = "mlService")]
    pub ml_service: ManifestProtocolFile,
    #[serde(rename = "control")]
    pub control: ManifestProtocolFile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestArtifact {
    pub profile: String,
    #[serde(rename = "file_name")]
    pub file_name: String,
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub version: String,
    /// The four public capability Exact Terms, in service order.
    pub capabilities: Vec<ManifestCapability>,
    /// Official presets with their resource guidance.
    pub presets: Vec<ManifestPreset>,
    /// Model options per capability service (service id -> model ids).
    pub models: BTreeMap<String, Vec<String>>,
    /// Dataset options for BioCLIP.
    pub datasets: Vec<String>,
    /// Dist platforms: every profile the release ships.
    pub platforms: Vec<ManifestPlatform>,
    /// Data-plane and control-plane protocol provenance.
    pub protocol: ManifestProtocol,
    /// Download artifacts, one entry per dist profile.
    pub hub: Vec<ManifestArtifact>,
}

/// Dist profile input provided by the release pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    pub name: String,
    pub target: String,
    pub backend: String,
}

/// Artifact input provided by the release pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactInfo {
    pub profile: String,
    pub file_name: String,
    pub sha256: String,
}

impl HubManifest {
    /// Build a complete manifest from the canonical tables plus release input.
    ///
    /// `version` is the release tag (for example `v0.1.1`); `base_url` is the
    /// download base that artifact URLs are derived from.
    pub fn build(
        version: &str,
        base_url: &str,
        platforms: &[PlatformInfo],
        artifacts: &[ArtifactInfo],
        protocol: ManifestProtocol,
    ) -> Self {
        let base_url = base_url.trim_end_matches('/');

        let mut hub: Vec<ManifestArtifact> = artifacts
            .iter()
            .map(|artifact| ManifestArtifact {
                profile: artifact.profile.clone(),
                file_name: artifact.file_name.clone(),
                url: format!("{base_url}/{}", artifact.file_name),
                sha256: artifact.sha256.clone(),
            })
            .collect();
        hub.sort_by(|left, right| left.profile.cmp(&right.profile));

        let mut platforms: Vec<ManifestPlatform> = platforms
            .iter()
            .map(|platform| ManifestPlatform {
                platform: platform_target_to_platform(&platform.target),
                profile: platform.name.clone(),
                backend: platform.backend.clone(),
                target: platform.target.clone(),
            })
            .collect();
        platforms.sort_by(|left, right| left.profile.cmp(&right.profile));

        let mut models = BTreeMap::new();
        for service in SERVICE_ORDER {
            let options = match service {
                "siglip" => &SIGLIP_MODELS[..],
                "face" => &FACE_MODELS[..],
                "ocr" => &OCR_MODELS[..],
                "bioclip" => &BIOCLIP_MODELS[..],
                _ => unreachable!("SERVICE_ORDER is the fixed four-service catalog"),
            };
            models.insert(
                service.to_owned(),
                options.iter().map(|model| (*model).to_owned()).collect(),
            );
        }

        let datasets = BIOCLIP_DATASETS
            .iter()
            .map(|dataset| (*dataset).to_owned())
            .collect::<Vec<_>>();

        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            version: version.to_owned(),
            capabilities: CAPABILITIES
                .iter()
                .map(|term| ManifestCapability {
                    id: term.service.to_owned(),
                    zh_cn: term.zh_cn.to_owned(),
                    en: term.en.to_owned(),
                })
                .collect(),
            presets: Preset::all()
                .iter()
                .map(|preset| ManifestPreset {
                    id: preset.name.to_owned(),
                    capabilities: preset.components.iter().map(|s| (*s).to_owned()).collect(),
                    siglip_model: preset.siglip_model.to_owned(),
                    bioclip_dataset: preset.bioclip_dataset.map(str::to_owned),
                    resources: ManifestResources {
                        ram_gb: preset.min_ram_gb,
                        vram_gb: preset.min_vram_gb,
                        disk_gb: preset.min_disk_gb,
                    },
                })
                .collect(),
            models,
            datasets,
            platforms,
            protocol,
            hub,
        }
    }
}

/// Map a Rust target triple to the user-facing platform id used across the
/// product (launcher platform profiles, Site, Desktop).
pub fn platform_target_to_platform(target: &str) -> String {
    let (os, arch) = match target {
        "aarch64-apple-darwin" => ("darwin", "arm64"),
        "x86_64-pc-windows-msvc" => ("windows", "x64"),
        "x86_64-unknown-linux-gnu" => ("linux", "x64"),
        "aarch64-unknown-linux-gnu" => ("linux", "arm64"),
        _ => {
            let mut parts = target.split('-');
            let arch = parts.next().unwrap_or(target);
            let os = parts.next_back().unwrap_or("unknown");
            (os, arch)
        }
    };
    format!("{os}-{arch}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_protocol() -> ManifestProtocol {
        ManifestProtocol {
            data_plane_major: DATA_PLANE_MAJOR,
            ml_service: ManifestProtocolFile {
                path: "crates/lumen-hub/proto/ml_service.proto".to_owned(),
                sha256: "a".repeat(64),
            },
            control: ManifestProtocolFile {
                path: "crates/lumen-hub/proto/control.proto".to_owned(),
                sha256: "b".repeat(64),
            },
        }
    }

    fn sample_manifest() -> HubManifest {
        HubManifest::build(
            "v0.0.0-test",
            "https://example.com/releases/download/v0.0.0-test/",
            &[
                PlatformInfo {
                    name: "linux-x64-cpu".to_owned(),
                    target: "x86_64-unknown-linux-gnu".to_owned(),
                    backend: "cpu".to_owned(),
                },
                PlatformInfo {
                    name: "darwin-arm64-metal".to_owned(),
                    target: "aarch64-apple-darwin".to_owned(),
                    backend: "metal".to_owned(),
                },
            ],
            &[
                ArtifactInfo {
                    profile: "darwin-arm64-metal".to_owned(),
                    file_name: "lumen-hub-darwin-arm64-metal.zip".to_owned(),
                    sha256: "c".repeat(64),
                },
                ArtifactInfo {
                    profile: "linux-x64-cpu".to_owned(),
                    file_name: "lumen-hub-linux-x64-cpu.zip".to_owned(),
                    sha256: "d".repeat(64),
                },
            ],
            sample_protocol(),
        )
    }

    #[test]
    fn manifest_contains_the_four_exact_capability_terms() {
        let manifest = sample_manifest();
        let terms = manifest
            .capabilities
            .iter()
            .map(|capability| {
                (
                    capability.id.as_str(),
                    capability.zh_cn.as_str(),
                    capability.en.as_str(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            terms,
            vec![
                ("siglip", "图像语义分析", "Image Semantic Analysis"),
                ("face", "人物识别", "Person Recognition"),
                ("ocr", "OCR文字识别", "OCR Text Recognition"),
                ("bioclip", "BioCLIP物种识别", "BioCLIP Species Recognition"),
            ]
        );
    }

    #[test]
    fn manifest_presets_carry_resource_guidance() {
        let manifest = sample_manifest();
        let minimal = manifest
            .presets
            .iter()
            .find(|preset| preset.id == "minimal")
            .expect("minimal preset");
        assert_eq!(minimal.capabilities, vec!["siglip", "face"]);
        assert_eq!(minimal.resources.ram_gb, 4);
        assert_eq!(minimal.resources.vram_gb, 2);
        assert_eq!(minimal.resources.disk_gb, 2);
        let brave = manifest
            .presets
            .iter()
            .find(|preset| preset.id == "brave")
            .expect("brave preset");
        assert_eq!(brave.siglip_model, crate::preset::SIGLIP_BRAVE_MODEL);
        assert_eq!(
            brave.bioclip_dataset.as_deref(),
            Some(crate::preset::BIOCLIP_FULL_DATASET)
        );
    }

    #[test]
    fn manifest_models_and_datasets_are_canonical() {
        let manifest = sample_manifest();
        assert_eq!(
            manifest.datasets,
            vec![
                crate::preset::BIOCLIP_CORE_DATASET,
                crate::preset::BIOCLIP_FULL_DATASET
            ]
        );
        assert_eq!(manifest.models["siglip"].len(), 2);
        assert_eq!(
            manifest.models["face"],
            vec![crate::preset::FACE_DEFAULT_MODEL]
        );
    }

    #[test]
    fn manifest_platforms_map_targets_to_platforms() {
        let manifest = sample_manifest();
        let darwin = manifest
            .platforms
            .iter()
            .find(|platform| platform.profile == "darwin-arm64-metal")
            .expect("darwin profile");
        assert_eq!(darwin.platform, "darwin-arm64");
        assert_eq!(darwin.backend, "metal");
        let linux = manifest
            .platforms
            .iter()
            .find(|platform| platform.profile == "linux-x64-cpu")
            .expect("linux profile");
        assert_eq!(linux.platform, "linux-x64");
    }

    #[test]
    fn manifest_serializes_with_expected_field_names() {
        let json = serde_json::to_value(sample_manifest()).expect("serialize manifest");
        assert_eq!(json["schemaVersion"], 2);
        assert_eq!(json["capabilities"][0]["zhCn"], "图像语义分析");
        assert_eq!(json["presets"][0]["resources"]["ramGb"], 4);
        assert_eq!(json["protocol"]["dataPlaneMajor"], DATA_PLANE_MAJOR);
        assert_eq!(
            json["hub"][0]["url"],
            "https://example.com/releases/download/v0.0.0-test/lumen-hub-darwin-arm64-metal.zip"
        );
    }
}
