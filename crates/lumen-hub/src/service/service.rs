use crate::service::{ServiceCapability, TaskRegistry};

/// Protocol-independent inference service boundary.
///
/// A service owns a task registry and exposes service-level capability metadata.
/// Transport layers such as gRPC should depend on this trait through `ServiceHub`
/// instead of making model services handle protocol messages directly.
pub trait InferenceService: Send + Sync {
    fn name(&self) -> &str;

    fn tasks(&self) -> &TaskRegistry;

    fn capability(&self) -> ServiceCapability;
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Bytes;

    use crate::service::{
        InferenceService, ServiceCapability, ServiceResult, TaskHandler, TaskRegistry, TaskRequest,
        TaskResult, TaskSpec,
    };

    struct EchoService {
        tasks: TaskRegistry,
    }

    impl EchoService {
        fn new() -> Self {
            let mut tasks = TaskRegistry::new();
            tasks.register(EchoTask::new()).unwrap();
            Self { tasks }
        }
    }

    impl InferenceService for EchoService {
        fn name(&self) -> &str {
            "echo"
        }

        fn tasks(&self) -> &TaskRegistry {
            &self.tasks
        }

        fn capability(&self) -> ServiceCapability {
            self.tasks
                .build_capability(self.name(), vec!["echo-model".to_owned()], "cpu")
        }
    }

    struct EchoTask {
        spec: TaskSpec,
    }

    impl EchoTask {
        fn new() -> Self {
            Self {
                spec: TaskSpec::new("echo_text", "echo text")
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
            Ok(TaskResult::new(request.payload, "text/plain"))
        }
    }

    #[tokio::test]
    async fn inference_service_exposes_tasks_and_capability() {
        let service: Box<dyn InferenceService> = Box::new(EchoService::new());

        assert_eq!(service.name(), "echo");
        assert_eq!(service.tasks().task_names(), vec!["echo_text"]);

        let result = service
            .tasks()
            .handle(
                "echo_text",
                TaskRequest::new(Bytes::from_static(b"hello"), "text/plain"),
            )
            .await
            .unwrap();
        assert_eq!(result.payload, Bytes::from_static(b"hello"));

        let capability = service.capability();
        assert_eq!(capability.service_name, "echo");
        assert_eq!(capability.model_ids, vec!["echo-model".to_owned()]);
        assert_eq!(capability.tasks.len(), 1);
        assert_eq!(capability.tasks[0].name, "echo_text");
    }
}
