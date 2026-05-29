use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use serde::Deserialize;

use super::dataset::BioClipDataset;
use super::factory::BioClipModelFactory;
use super::preprocess::ClipImagePreprocessConfig;
use super::task::BioClipClassifyTask;
use crate::backend::{BACKEND_NAME, Device};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

/// CLIP learned temperature default: `ln(100)`, so `logit_scale.exp() == 100`.
const DEFAULT_LOGIT_SCALE: f32 = 4.60517;

/// Parsed CLIP/BioCLIP `task_metadata` from `model_info.json`.
#[derive(Debug, Clone, Deserialize)]
struct ClipTaskMetadata {
    #[allow(dead_code)]
    #[serde(default)]
    embedding_dim: Option<usize>,
    tasks: BTreeMap<String, ClipTaskConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClipTaskConfig {
    /// Model component name: `"text"` or `"vision"`.
    component: String,
    /// Image preprocessing metadata (vision tasks).
    #[serde(default)]
    preprocess: Option<ClipImagePreprocessConfig>,
    /// CLIP learned temperature parameter.
    #[serde(default)]
    logit_scale: Option<f32>,
}

/// BioCLIP classification service backed by Burn.
///
/// A model alias with a configured `dataset` exposes `bioclip_classify`: the
/// vision encoder runs on Burn and results are classified against the dataset's
/// precomputed taxon embeddings (ANN search + rerank).
pub struct BioclipService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
}

impl BioclipService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        device: Arc<Device>,
    ) -> ServiceResult<Self> {
        let factory = BioClipModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();

        for model_config in service_config.models.values() {
            let model_name = &model_config.model;
            let runtime = model_config.runtime;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");
            let model_info = factory.load_model_info(model_name)?;
            let meta = parse_clip_metadata(model_name, &model_info)?;

            model_ids.push(model_name.to_owned());

            let dataset_name = model_config.dataset.as_deref().ok_or_else(|| {
                ServiceError::InvalidArgument(format!(
                    "BioCLIP model `{model_name}` requires a `dataset` in its service config to \
                     expose `bioclip_classify`"
                ))
            })?;

            let vision = meta
                .tasks
                .values()
                .find(|task| task.component == "vision")
                .ok_or_else(|| {
                    ServiceError::InvalidArgument(format!(
                        "BioCLIP model `{model_name}` requires a vision task in task_metadata"
                    ))
                })?;
            let preprocess = vision.preprocess.clone().ok_or_else(|| {
                ServiceError::InvalidArgument(format!(
                    "BioCLIP model `{model_name}` vision task requires `preprocess` metadata"
                ))
            })?;
            let logit_scale = meta
                .tasks
                .values()
                .find_map(|task| task.logit_scale)
                .unwrap_or(DEFAULT_LOGIT_SCALE);

            let model = factory.create_vision_model(model_name, runtime, precision, &device)?;
            let paths = factory.resolve_dataset_paths(model_name, dataset_name)?;
            let dataset = BioClipDataset::open(
                dataset_name.to_owned(),
                paths.embeddings_path,
                paths.labels_path,
                paths.index_path,
            )?;

            let task = BioClipClassifyTask::new(
                "bioclip_classify",
                Arc::new(model),
                model_name,
                preprocess,
                logit_scale,
                dataset_name,
                Arc::new(dataset),
            );
            tasks.register(task)?;
        }

        Ok(Self {
            name: service_name.to_owned(),
            tasks,
            model_ids,
        })
    }
}

fn parse_clip_metadata(
    model_name: &str,
    model_info: &lumen_schema::ModelInfo,
) -> ServiceResult<ClipTaskMetadata> {
    let raw_meta = serde_json::to_value(model_info.task_metadata.clone().unwrap_or_default())
        .map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` task_metadata serialization failed: {e}"
            ))
        })?;
    serde_json::from_value(raw_meta).map_err(|e| {
        ServiceError::InvalidArgument(format!(
            "model `{model_name}` task_metadata is not valid CLIP metadata: {e}"
        ))
    })
}

impl InferenceService for BioclipService {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks(&self) -> &TaskRegistry {
        &self.tasks
    }

    fn capability(&self) -> ServiceCapability {
        self.tasks
            .build_capability(&self.name, self.model_ids.clone(), BACKEND_NAME)
    }
}
