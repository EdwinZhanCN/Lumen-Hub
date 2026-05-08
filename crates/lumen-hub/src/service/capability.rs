use std::collections::HashMap;

use crate::service::TaskSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceCapability {
    pub service_name: String,
    pub model_ids: Vec<String>,
    pub runtime: String,
    pub max_concurrency: u32,
    pub precisions: Vec<String>,
    pub extra: HashMap<String, String>,
    pub tasks: Vec<TaskSpec>,
    pub protocol_version: String,
}

impl ServiceCapability {
    pub fn new(
        service_name: impl Into<String>,
        model_ids: Vec<String>,
        runtime: impl Into<String>,
        tasks: Vec<TaskSpec>,
    ) -> Self {
        Self {
            service_name: service_name.into(),
            model_ids,
            runtime: runtime.into(),
            max_concurrency: 1,
            precisions: Vec::new(),
            extra: HashMap::new(),
            tasks,
            protocol_version: "1.0".to_owned(),
        }
    }

    pub fn with_max_concurrency(mut self, max_concurrency: u32) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    pub fn with_precisions<I, S>(mut self, precisions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.precisions = precisions.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    pub fn with_protocol_version(mut self, protocol_version: impl Into<String>) -> Self {
        self.protocol_version = protocol_version.into();
        self
    }
}
