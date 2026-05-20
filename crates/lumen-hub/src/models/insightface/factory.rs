use std::{fs, path::PathBuf, sync::Arc};

use lumen_schema::{ModelInfo, Runtime};
#[cfg(feature = "mnn")]
use lumnn::mnn::MnnNode;
use lumnn::{
    core::{context::MLContext, node::MLNode},
    ort::node::OrtNode,
};

use crate::service::{ServiceError, ServiceResult};

pub struct InsightFaceModelFactory {
    cache_dir: String,
}

impl InsightFaceModelFactory {
    pub fn new(cache_dir: impl Into<String>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        PathBuf::from(&self.cache_dir).join(model_name)
    }

    pub fn load_model_info(&self, model_name: &str) -> ServiceResult<ModelInfo> {
        let path = self.model_dir(model_name).join("model_info.json");
        let contents = fs::read_to_string(&path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to read model_info.json at {}: {e}",
                path.display()
            ))
        })?;
        ModelInfo::from_json_str(&contents).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "invalid model_info.json at {}: {e}",
                path.display()
            ))
        })
    }

    pub fn resolve_component_path(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
    ) -> PathBuf {
        let runtime_dir = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
            Runtime::Mnn => "mnn",
            Runtime::MnnLlm => "mnn-llm",
        };
        let ext = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
            Runtime::Mnn => "mnn",
            Runtime::MnnLlm => "json",
        };
        self.model_dir(model_name)
            .join(runtime_dir)
            .join(format!("{component}.{precision}.{ext}"))
    }

    pub fn create_component(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
        context: &Arc<MLContext>,
    ) -> ServiceResult<Box<dyn MLNode>> {
        let model_path = self.resolve_component_path(model_name, runtime, component, precision);
        let path_str = model_path.to_str().ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model path is not valid UTF-8: {}",
                model_path.display()
            ))
        })?;
        let name = format!("{model_name}_{component}");
        match runtime {
            Runtime::Onnx => OrtNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            #[cfg(feature = "mnn")]
            Runtime::Mnn => MnnNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            #[cfg(not(feature = "mnn"))]
            Runtime::Mnn => Err(ServiceError::InvalidArgument(
                "InsightFace MNN runtime is not enabled in this lumen-hub build".to_owned(),
            )),
            Runtime::CandleOnnx => Err(ServiceError::InvalidArgument(
                "InsightFace Candle ONNX runtime is not implemented yet; use runtime=onnx"
                    .to_owned(),
            )),
            Runtime::Rknn => Err(ServiceError::InvalidArgument(
                "InsightFace RKNN runtime is not implemented yet".to_owned(),
            )),
            Runtime::MnnLlm => Err(ServiceError::InvalidArgument(
                "InsightFace MNN-LLM runtime is not supported".to_owned(),
            )),
        }
    }
}
