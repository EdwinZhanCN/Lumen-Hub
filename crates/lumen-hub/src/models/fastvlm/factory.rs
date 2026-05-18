use std::{fs, path::PathBuf, sync::Arc};

use lumen_schema::{ModelInfo, Runtime};
use lumnn::{
    core::{context::MLContext, node::MLNode},
    ort::node::OrtNode,
};

use crate::service::{ServiceError, ServiceResult};

pub struct FastVlmModelFactory {
    cache_dir: String,
}

impl FastVlmModelFactory {
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
        };
        let ext = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
            Runtime::Mnn => "mnn",
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
            Runtime::CandleOnnx => Err(ServiceError::InvalidArgument(
                "FastVLM Candle ONNX runtime is temporarily disabled because the exported vision graph currently fails at execution time in candle-onnx; use runtime=onnx for now"
                    .to_owned(),
            )),
            Runtime::Rknn => Err(ServiceError::InvalidArgument(
                "FastVLM RKNN runtime is not implemented yet".to_owned(),
            )),
            Runtime::Mnn => Err(ServiceError::InvalidArgument(
                "FastVLM MNN runtime is not implemented yet".to_owned(),
            )),
        }
    }

    pub fn load_tokenizer(&self, model_name: &str) -> ServiceResult<tokenizers::Tokenizer> {
        let tokenizer_path = self.model_dir(model_name).join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lumnn::core::context::{MLContext, MLContextOptions};

    use super::FastVlmModelFactory;
    use crate::service::ServiceError;

    #[test]
    fn rejects_candle_runtime_for_fastvlm_components() {
        let factory = FastVlmModelFactory::new("/tmp");
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let err = match factory.create_component(
            "fast-vlm-0.5b",
            lumen_schema::Runtime::CandleOnnx,
            "vision",
            "fp16",
            &Arc::clone(&context),
        ) {
            Ok(_) => panic!("FastVLM Candle runtime should be rejected"),
            Err(err) => err,
        };

        assert!(
            matches!(err, ServiceError::InvalidArgument(message) if message.contains("temporarily disabled"))
        );
    }
}
