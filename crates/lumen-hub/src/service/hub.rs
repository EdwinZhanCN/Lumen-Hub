use std::{collections::HashMap, sync::Arc};

use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRequest, TaskResult,
};

/// Protocol-independent registry and router for inference services.
///
/// The hub owns service instances and routes task requests to the selected
/// service. Protocol adapters such as gRPC should translate transport messages
/// into `TaskRequest` values before calling into this type.
#[derive(Default)]
pub struct ServiceHub {
    services: HashMap<String, Arc<dyn InferenceService>>,
}

impl ServiceHub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<S>(&mut self, service: S) -> ServiceResult<()>
    where
        S: InferenceService + 'static,
    {
        self.register_arc(Arc::new(service))
    }

    pub fn register_arc(&mut self, service: Arc<dyn InferenceService>) -> ServiceResult<()> {
        let name = service.name().to_owned();
        if self.services.contains_key(&name) {
            return Err(ServiceError::DuplicateService(name));
        }

        self.services.insert(name, service);
        Ok(())
    }

    pub fn get(&self, service_name: &str) -> ServiceResult<Arc<dyn InferenceService>> {
        self.services
            .get(service_name)
            .cloned()
            .ok_or_else(|| ServiceError::service_not_found(service_name, self.service_names()))
    }

    pub async fn handle(
        &self,
        service_name: &str,
        task_name: &str,
        request: TaskRequest,
    ) -> ServiceResult<TaskResult> {
        let service = self.get(service_name)?;
        service.tasks().handle(task_name, request).await
    }

    pub fn capabilities(&self) -> Vec<ServiceCapability> {
        let mut capabilities = self
            .services
            .values()
            .map(|service| service.capability())
            .collect::<Vec<_>>();
        capabilities.sort_by(|left, right| left.service_name.cmp(&right.service_name));
        capabilities
    }

    pub fn service_names(&self) -> Vec<String> {
        let mut names = self.services.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn len(&self) -> usize {
        self.services.len()
    }

    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Bytes;

    use crate::service::{
        InferenceService, ServiceCapability, ServiceError, ServiceHub, ServiceResult, TaskHandler,
        TaskRegistry, TaskRequest, TaskResult, TaskSpec,
    };

    struct EchoService {
        name: String,
        model_id: String,
        tasks: TaskRegistry,
    }

    impl EchoService {
        fn new(name: &str, model_id: &str, task_name: &str) -> Self {
            let mut tasks = TaskRegistry::new();
            tasks.register(EchoTask::new(task_name)).unwrap();

            Self {
                name: name.to_owned(),
                model_id: model_id.to_owned(),
                tasks,
            }
        }
    }

    impl InferenceService for EchoService {
        fn name(&self) -> &str {
            &self.name
        }

        fn tasks(&self) -> &TaskRegistry {
            &self.tasks
        }

        fn capability(&self) -> ServiceCapability {
            self.tasks
                .build_capability(&self.name, vec![self.model_id.clone()], "cpu")
        }
    }

    struct EchoTask {
        spec: TaskSpec,
    }

    impl EchoTask {
        fn new(name: &str) -> Self {
            Self {
                spec: TaskSpec::new(name, "echo payload")
                    .with_input_mimes(["text/plain"])
                    .with_output_mime("text/plain"),
            }
        }
    }

    #[async_trait]
    impl TaskHandler for EchoTask {
        fn spec(&self) -> &TaskSpec {
            &self.spec
        }

        async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
            Ok(TaskResult::new(request.payload, "text/plain")
                .with_meta("task", self.spec.name.clone()))
        }
    }

    #[tokio::test]
    async fn routes_requests_to_selected_service_task() {
        let mut hub = ServiceHub::new();
        hub.register(EchoService::new("clip", "clip-model", "clip_text_embed"))
            .unwrap();

        let result = hub
            .handle(
                "clip",
                "clip_text_embed",
                TaskRequest::new(Bytes::from_static(b"hello"), "text/plain"),
            )
            .await
            .unwrap();

        assert_eq!(result.payload, Bytes::from_static(b"hello"));
        assert_eq!(result.meta.get("task"), Some(&"clip_text_embed".to_owned()));
    }

    #[test]
    fn rejects_duplicate_service_names() {
        let mut hub = ServiceHub::new();
        hub.register(EchoService::new("clip", "clip-model-a", "clip_text_embed"))
            .unwrap();

        let err = hub
            .register(EchoService::new("clip", "clip-model-b", "clip_image_embed"))
            .unwrap_err();

        assert!(matches!(err, ServiceError::DuplicateService(service) if service == "clip"));
    }

    #[test]
    fn reports_available_services_when_missing() {
        let mut hub = ServiceHub::new();
        hub.register(EchoService::new("clip", "clip-model", "clip_text_embed"))
            .unwrap();

        let err = match hub.get("ocr") {
            Ok(_) => panic!("missing service should fail"),
            Err(err) => err,
        };

        assert!(
            matches!(err, ServiceError::ServiceNotFound { service, available } if service == "ocr" && available == vec!["clip"])
        );
    }

    #[test]
    fn aggregates_capabilities_in_stable_order() {
        let mut hub = ServiceHub::new();
        hub.register(EchoService::new("ocr", "ocr-model", "ocr_image_text"))
            .unwrap();
        hub.register(EchoService::new("clip", "clip-model", "clip_text_embed"))
            .unwrap();

        let capabilities = hub.capabilities();

        assert_eq!(capabilities.len(), 2);
        assert_eq!(capabilities[0].service_name, "clip");
        assert_eq!(capabilities[0].model_ids, vec!["clip-model"]);
        assert_eq!(capabilities[0].tasks[0].name, "clip_text_embed");
        assert_eq!(capabilities[1].service_name, "ocr");
        assert_eq!(capabilities[1].model_ids, vec!["ocr-model"]);
        assert_eq!(capabilities[1].tasks[0].name, "ocr_image_text");
    }
}
