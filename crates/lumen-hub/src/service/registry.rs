use std::{collections::HashMap, sync::Arc};

use crate::service::{
    BatchKey, ServiceCapability, ServiceError, ServiceResult, TaskHandler, TaskRequest, TaskResult,
    TaskSpec,
};

#[derive(Default)]
pub struct TaskRegistry {
    tasks: HashMap<String, Arc<dyn TaskHandler>>,
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<H>(&mut self, handler: H) -> ServiceResult<()>
    where
        H: TaskHandler + 'static,
    {
        self.register_arc(Arc::new(handler))
    }

    pub fn register_arc(&mut self, handler: Arc<dyn TaskHandler>) -> ServiceResult<()> {
        let name = handler.spec().name.clone();
        if self.tasks.contains_key(&name) {
            return Err(ServiceError::DuplicateTask(name));
        }

        self.tasks.insert(name, handler);
        Ok(())
    }

    pub fn get(&self, task_name: &str) -> ServiceResult<Arc<dyn TaskHandler>> {
        self.tasks
            .get(task_name)
            .cloned()
            .ok_or_else(|| ServiceError::task_not_found(task_name, self.task_names()))
    }

    pub async fn handle(&self, task_name: &str, request: TaskRequest) -> ServiceResult<TaskResult> {
        self.get(task_name)?.handle(request).await
    }

    pub fn batch_key(
        &self,
        task_name: &str,
        request: &TaskRequest,
    ) -> ServiceResult<Option<BatchKey>> {
        self.get(task_name)?.batch_key(request)
    }

    pub async fn handle_batch(
        &self,
        task_name: &str,
        requests: Vec<TaskRequest>,
    ) -> ServiceResult<Vec<TaskResult>> {
        self.get(task_name)?.handle_batch(requests).await
    }

    pub fn task_names(&self) -> Vec<String> {
        let mut names = self.tasks.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn task_specs(&self) -> Vec<TaskSpec> {
        let mut specs = self
            .tasks
            .values()
            .map(|handler| handler.spec().clone())
            .collect::<Vec<_>>();
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        specs
    }

    pub fn build_capability(
        &self,
        service_name: impl Into<String>,
        model_ids: Vec<String>,
        runtime: impl Into<String>,
    ) -> ServiceCapability {
        ServiceCapability::new(service_name, model_ids, runtime, self.task_specs())
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Bytes;

    use crate::service::{
        ServiceError, ServiceResult, TaskHandler, TaskRegistry, TaskRequest, TaskResult, TaskSpec,
    };

    struct EchoHandler {
        spec: TaskSpec,
    }

    impl EchoHandler {
        fn new(name: &str) -> Self {
            Self {
                spec: TaskSpec::new(name, "echo payload")
                    .with_input_mimes(["text/plain"])
                    .with_output_mime("text/plain")
                    .with_limit("max_payload_size", "1024"),
            }
        }
    }

    #[async_trait]
    impl TaskHandler for EchoHandler {
        fn spec(&self) -> &TaskSpec {
            &self.spec
        }

        async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
            Ok(TaskResult::new(request.payload, "text/plain").with_meta("handled_by", "echo"))
        }
    }

    #[test]
    fn rejects_duplicate_task_names() {
        let mut registry = TaskRegistry::new();

        registry.register(EchoHandler::new("echo")).unwrap();
        let err = registry.register(EchoHandler::new("echo")).unwrap_err();

        assert!(matches!(err, ServiceError::DuplicateTask(task) if task == "echo"));
    }

    #[test]
    fn reports_available_tasks_when_task_is_missing() {
        let mut registry = TaskRegistry::new();
        registry.register(EchoHandler::new("echo")).unwrap();

        let err = match registry.get("missing") {
            Ok(_) => panic!("missing task should fail"),
            Err(err) => err,
        };

        assert!(
            matches!(err, ServiceError::TaskNotFound { task, available } if task == "missing" && available == vec!["echo"])
        );
    }

    #[tokio::test]
    async fn routes_requests_to_registered_handler() {
        let mut registry = TaskRegistry::new();
        registry.register(EchoHandler::new("echo")).unwrap();

        let result = registry
            .handle(
                "echo",
                TaskRequest::new(Bytes::from_static(b"hello"), "text/plain"),
            )
            .await
            .unwrap();

        assert_eq!(result.payload, Bytes::from_static(b"hello"));
        assert_eq!(result.payload_mime, "text/plain");
        assert_eq!(result.meta.get("handled_by"), Some(&"echo".to_owned()));
    }

    #[test]
    fn builds_protocol_independent_capability_from_task_specs() {
        let mut registry = TaskRegistry::new();
        registry.register(EchoHandler::new("echo")).unwrap();

        let capability =
            registry.build_capability("test-service", vec!["test-model".to_owned()], "cpu");

        assert_eq!(capability.service_name, "test-service");
        assert_eq!(capability.model_ids, vec!["test-model".to_owned()]);
        assert_eq!(capability.runtime, "cpu");
        assert_eq!(capability.tasks.len(), 1);
        assert_eq!(capability.tasks[0].name, "echo");
        assert_eq!(
            capability.tasks[0].limits.get("max_payload_size"),
            Some(&"1024".to_owned())
        );
    }
}
