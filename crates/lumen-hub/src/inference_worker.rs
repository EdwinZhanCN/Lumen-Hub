use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{OnceLock, mpsc},
    thread,
};

use thiserror::Error;
use tokio::sync::oneshot;

use crate::backend::{cleanup_memory, default_device};

const INFERENCE_WORKER_STACK_SIZE: usize = 256 * 1024 * 1024;

type Job = Box<dyn FnOnce() + Send + 'static>;

static WORKER: OnceLock<InferenceWorker> = OnceLock::new();

#[derive(Debug, Error)]
pub enum InferenceWorkerError {
    #[error("inference worker has stopped")]
    Stopped,

    #[error("inference job panicked: {0}")]
    Panicked(String),
}

struct InferenceWorker {
    sender: mpsc::Sender<Job>,
}

impl InferenceWorker {
    fn start() -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        thread::Builder::new()
            .name("lumen-infer-worker".to_owned())
            .stack_size(INFERENCE_WORKER_STACK_SIZE)
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .expect("failed to spawn Lumen inference worker");

        Self { sender }
    }
}

/// Runs Burn/CubeCL work on a single OS thread.
///
/// CubeCL/Fusion uses a thread-local stream id, so dispatching inference through
/// Tokio's blocking pool can create one GPU stream per blocking worker thread.
/// This worker gives long-lived serving a stable stream and performs backend
/// cleanup on that same stream after each job.
pub async fn run<F, T>(f: F) -> Result<T, InferenceWorkerError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let worker = WORKER.get_or_init(InferenceWorker::start);
    let (reply, receive) = oneshot::channel();

    worker
        .sender
        .send(Box::new(move || {
            let result = catch_unwind(AssertUnwindSafe(f)).map_err(panic_message);
            // Reply before cleanup: the worker is a single OS thread, so the
            // next job still only starts after cleanup, but the caller does not
            // pay for memory reclamation + device sync on its critical path.
            let _ = reply.send(result);
            // Experiment escape hatch: LUMEN_SKIP_INFER_CLEANUP=1 disables
            // per-job memory cleanup for A/B benchmarks.
            if std::env::var_os("LUMEN_SKIP_INFER_CLEANUP").is_none() {
                cleanup_memory(&default_device());
            }
        }))
        .map_err(|_| InferenceWorkerError::Stopped)?;

    receive.await.map_err(|_| InferenceWorkerError::Stopped)?
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> InferenceWorkerError {
    let message = if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    };
    InferenceWorkerError::Panicked(message)
}
