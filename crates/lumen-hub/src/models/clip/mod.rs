pub mod factory;
pub mod nodes;
pub mod pipeline;
pub mod service;
pub mod task;

pub use factory::ClipModelFactory;
pub use nodes::L2NormalizeNode;
pub use pipeline::build_embedding_pipeline;
pub use service::ClipService;
pub use task::ClipTextEmbedTask;
