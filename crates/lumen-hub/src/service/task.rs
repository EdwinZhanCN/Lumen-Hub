use std::collections::HashMap;

use async_trait::async_trait;
use bytes::Bytes;

use crate::service::ServiceResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    pub name: String,
    pub description: String,
    pub input_mimes: Vec<String>,
    pub output_mimes: Vec<String>,
    pub limits: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

impl TaskSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_mimes: Vec::new(),
            output_mimes: Vec::new(),
            limits: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_input_mimes<I, S>(mut self, input_mimes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.input_mimes = input_mimes.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_output_mime(mut self, output_mime: impl Into<String>) -> Self {
        self.output_mimes = vec![output_mime.into()];
        self
    }

    pub fn with_output_mimes<I, S>(mut self, output_mimes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.output_mimes = output_mimes.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_limit(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.limits.insert(key.into(), value.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRequest {
    pub payload: Bytes,
    pub payload_mime: String,
    pub meta: HashMap<String, String>,
}

impl TaskRequest {
    pub fn new(payload: impl Into<Bytes>, payload_mime: impl Into<String>) -> Self {
        Self {
            payload: payload.into(),
            payload_mime: payload_mime.into(),
            meta: HashMap::new(),
        }
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.meta.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskResult {
    pub payload: Bytes,
    pub payload_mime: String,
    pub result_schema: Option<String>,
    pub meta: HashMap<String, String>,
}

impl TaskResult {
    pub fn new(payload: impl Into<Bytes>, payload_mime: impl Into<String>) -> Self {
        Self {
            payload: payload.into(),
            payload_mime: payload_mime.into(),
            result_schema: None,
            meta: HashMap::new(),
        }
    }

    pub fn with_result_schema(mut self, result_schema: impl Into<String>) -> Self {
        self.result_schema = Some(result_schema.into());
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.meta.insert(key.into(), value.into());
        self
    }
}

#[async_trait]
pub trait TaskHandler: Send + Sync {
    fn spec(&self) -> &TaskSpec;

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult>;
}
