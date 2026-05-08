use std::sync::Arc;

use lumnn::core::context::MLContext;

use crate::service::ServiceResult;

/// Builds a model from service config and the shared execution context.
///
/// The factory owns model selection and resource resolution — model paths are
/// derived from config fields (cache_dir, region, model, runtime, precision),
/// so no separate resources parameter is needed.
pub trait ModelFactory<C, M>: Send + Sync {
    fn create(&self, config: &C, context: Arc<MLContext>) -> ServiceResult<M>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lumnn::core::context::{MLContext, MLContextOptions};

    use crate::service::{ModelFactory, ServiceError, ServiceResult};

    struct TestConfig {
        model_id: String,
        model_path: String,
        scale: f32,
    }

    #[derive(Debug, PartialEq)]
    struct TestModel {
        model_id: String,
        model_path: String,
        scale: f32,
        accelerated: bool,
    }

    struct TestModelFactory;

    impl ModelFactory<TestConfig, TestModel> for TestModelFactory {
        fn create(&self, config: &TestConfig, context: Arc<MLContext>) -> ServiceResult<TestModel> {
            if config.model_id.is_empty() {
                return Err(ServiceError::InvalidArgument(
                    "model_id must not be empty".to_owned(),
                ));
            }

            Ok(TestModel {
                model_id: config.model_id.clone(),
                model_path: config.model_path.clone(),
                scale: config.scale,
                accelerated: context.accelerated(),
            })
        }
    }

    #[test]
    fn model_factory_creates_model_from_config_and_context() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let factory: Box<dyn ModelFactory<TestConfig, TestModel>> = Box::new(TestModelFactory);

        let model = factory
            .create(
                &TestConfig {
                    model_id: "test-model".to_owned(),
                    model_path: "/models/test.onnx".to_owned(),
                    scale: 0.5,
                },
                Arc::clone(&context),
            )
            .unwrap();

        assert_eq!(
            model,
            TestModel {
                model_id: "test-model".to_owned(),
                model_path: "/models/test.onnx".to_owned(),
                scale: 0.5,
                accelerated: false,
            }
        );
    }

    #[test]
    fn model_factory_can_validate_config() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let err = TestModelFactory
            .create(
                &TestConfig {
                    model_id: String::new(),
                    model_path: "/models/test.onnx".to_owned(),
                    scale: 1.0,
                },
                context,
            )
            .unwrap_err();

        assert!(
            matches!(err, ServiceError::InvalidArgument(message) if message.contains("model_id"))
        );
    }
}
