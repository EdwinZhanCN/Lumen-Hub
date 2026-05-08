#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("service `{0}` is already registered")]
    DuplicateService(String),

    #[error("service `{service}` was not found; available services: {available:?}")]
    ServiceNotFound {
        service: String,
        available: Vec<String>,
    },

    #[error("task `{0}` is already registered")]
    DuplicateTask(String),

    #[error("task `{task}` was not found; available tasks: {available:?}")]
    TaskNotFound {
        task: String,
        available: Vec<String>,
    },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("service unavailable: {0}")]
    Unavailable(String),

    #[error("service internal error: {0}")]
    Internal(String),
}

impl ServiceError {
    pub fn service_not_found(service: impl Into<String>, available: Vec<String>) -> Self {
        Self::ServiceNotFound {
            service: service.into(),
            available,
        }
    }

    pub fn task_not_found(task: impl Into<String>, available: Vec<String>) -> Self {
        Self::TaskNotFound {
            task: task.into(),
            available,
        }
    }
}

pub type ServiceResult<T> = Result<T, ServiceError>;
