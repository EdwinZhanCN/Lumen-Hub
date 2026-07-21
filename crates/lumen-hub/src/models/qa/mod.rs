//! QA test model package (feature `qa`): a tiny deterministic Burn model with
//! the same service/task/loading surface as the production packages. Compiled
//! only for the e2e harness — release dist profiles never enable the feature.

pub mod fixture;
pub mod model;
pub mod service;
pub mod task;

pub use fixture::{QA_MODEL_NAME, QA_PRECISIONS, write_model_fixture};
pub use model::{QA_EMBED_DIM, QA_INPUT_NUMEL, QA_INPUT_SIZE, QaNet};
pub use service::QaService;
pub use task::{QA_EMBED_TASK, QA_TENSOR_PREPROCESS_ID, QaEmbedTask, tensor_request_meta};
