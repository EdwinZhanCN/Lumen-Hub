//! QA inference service: loads the tiny deterministic net through the exact
//! production weight path (`load_burnpack`, incl. fp16 adapter and runtime Q8
//! quantization) using the standard repository layout.

use std::{fs, path::PathBuf, sync::Arc};

use lumen_schema::ServiceConfig;

use super::model::QaNet;
use super::task::QaEmbedTask;
use crate::backend::{BACKEND_NAME, Device};
use crate::model_arch::load_burnpack;
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

pub const QA_COMPONENT: &str = "net";

pub struct QaService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
}

impl QaService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        device: Arc<Device>,
    ) -> ServiceResult<Self> {
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();

        for model_config in service_config.models.values() {
            let model_name = &model_config.model;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");
            let model_dir = PathBuf::from(cache_dir).join(model_name);

            // model_info.json must exist and parse — the lifecycle tests rely
            // on the loader honestly consuming the downloaded repository.
            let info_path = model_dir.join("model_info.json");
            let contents = fs::read_to_string(&info_path).map_err(|e| {
                ServiceError::InvalidArgument(format!(
                    "failed to read model_info.json at {}: {e}",
                    info_path.display()
                ))
            })?;
            lumen_schema::ModelInfo::from_json_str(&contents).map_err(|e| {
                ServiceError::InvalidArgument(format!(
                    "invalid model_info.json at {}: {e}",
                    info_path.display()
                ))
            })?;

            let weights = model_dir
                .join("burn")
                .join(format!("{QA_COMPONENT}.{precision}.bpk"));
            let net = load_burnpack(
                QaNet::deterministic(&device),
                &weights.display().to_string(),
                precision,
            )
            .map_err(ServiceError::InvalidArgument)?;

            model_ids.push(model_name.clone());
            tasks.register(QaEmbedTask::new(
                model_name.clone(),
                net,
                Arc::clone(&device),
            ))?;
        }

        Ok(Self {
            name: service_name.to_owned(),
            tasks,
            model_ids,
        })
    }
}

impl InferenceService for QaService {
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
