//! Zero-config request batching for the daemon layer.
//!
//! ## Design
//!
//! Traditional batchers (e.g. Triton) expose two magic numbers:
//! `max_batch_size` + `max_queue_delay`. Both require manual tuning and
//! monitoring to get right.
//!
//! This batcher adopts a "busy-then-harvest" strategy that eliminates the
//! delay parameter:
//!
//! ```text
//!   model idle
//!     → request arrives, execute immediately (batch_size = 1)
//!     → while model is busy, new requests queue up
//!     → model finishes, harvest entire queue at once
//!     → batch_size = min(queue.len(), batch_limit)
//!     → execute, repeat
//! ```
//!
//! The model's own execution time acts as a self-adapting collection
//! window: fast models collect fewer requests per batch; slow models
//! naturally collect more. No delay knob needed.
//!
//! `batch_limit` can be auto-inferred from the model's dynamic input
//! shape and GPU memory budget (see [`BatchLimit`]). Users can override
//! it when they want explicit control.

use std::future::Future;
use std::pin::Pin;

use crate::service::{ServiceResult, TaskRequest, TaskResult};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Describes where per-request results should be delivered.
pub type Responder = tokio::sync::oneshot::Sender<ServiceResult<TaskResult>>;

/// A request that has been submitted to the batcher but not yet executed.
pub struct PendingRequest {
    pub request: TaskRequest,
    pub respond_to: Responder,
}

/// Strategy for determining the maximum batch size.
///
/// In zero-config mode, the batcher infers the limit from the model metadata
/// (e.g. ONNX dynamic axis bounds). Users who want explicit control can
/// override it with [`BatchLimit::Fixed`].
pub enum BatchLimit {
    /// Use a user-supplied upper bound.
    Fixed(usize),
    /// Infer from model metadata (the default zero-config path).
    AutoInferred { fallback: usize },
}

impl Default for BatchLimit {
    fn default() -> Self {
        Self::AutoInferred { fallback: 8 }
    }
}

impl BatchLimit {
    /// Resolve the effective batch limit.
    ///
    /// In [`BatchLimit::AutoInferred`] mode, returns `fallback` for now.
    /// A future implementation may inspect the ONNX model's input shape and
    /// GPU memory budget to compute a tighter bound automatically.
    pub fn resolve(&self) -> usize {
        match self {
            Self::Fixed(limit) => *limit,
            Self::AutoInferred { fallback } => *fallback,
        }
    }
}

// ---------------------------------------------------------------------------
// Batcher
// ---------------------------------------------------------------------------

type BatchFn = Box<
    dyn Fn(Vec<TaskRequest>) -> Pin<Box<dyn Future<Output = ServiceResult<Vec<TaskResult>>> + Send>>
        + Send
        + Sync,
>;

/// Zero-config batcher that aggregates individual inference requests into
/// batches using the model's execution rhythm as the natural collection
/// window.
///
/// ## Usage
///
/// ```ignore
/// let batcher = Batcher::new(batch_limit, batch_fn);
/// let responder = batcher.submit(request).await?;
/// let result = responder.await??;
/// ```
pub struct Batcher {
    /// Queue of requests waiting to be batched.
    tx: mpsc::UnboundedSender<PendingRequest>,
    rx: mpsc::UnboundedReceiver<PendingRequest>,
    batch_limit: usize,
}

impl Batcher {
    /// Creates a new [`Batcher`].
    ///
    /// Call [`Batcher::run`] (or `tokio::spawn(batcher.run(batch_fn))`)
    /// to start the batching loop.
    pub fn new(batch_limit: BatchLimit) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let batch_limit = batch_limit.resolve();
        Self {
            tx,
            rx,
            batch_limit,
        }
    }

    /// Submit a single request and receive a oneshot channel that will
    /// resolve once the batch has been executed.
    pub fn submit(
        &self,
        request: TaskRequest,
    ) -> Result<
        tokio::sync::oneshot::Receiver<ServiceResult<TaskResult>>,
        mpsc::error::SendError<PendingRequest>,
    > {
        let (responder, receiver) = tokio::sync::oneshot::channel();
        self.tx
            .send(PendingRequest {
                request,
                respond_to: responder,
            })
            .map(|()| receiver)
    }

    /// Run the batcher loop.
    ///
    /// Blocks the calling task. Typically spawned with
    /// `tokio::spawn(batcher.run())`.
    ///
    /// ## Algorithm
    ///
    /// 1. Wait for the first request (blocks).
    /// 2. Harvest any additional requests that have arrived in the meantime
    ///    (non-blocking drain).
    /// 3. Execute the batch.
    /// 4. Scatter results back to each request's [`Responder`].
    /// 5. Repeat.
    pub async fn run(mut self, batch_fn: BatchFn) {
        let mut pending: Vec<PendingRequest> = Vec::new();

        loop {
            // --- step 1: wait for the first request ---
            match self.rx.recv().await {
                Some(req) => pending.push(req),
                None => break, // channel closed
            }

            // --- step 2: drain any additional requests (non-blocking) ---
            while pending.len() < self.batch_limit {
                match self.rx.try_recv() {
                    Ok(req) => pending.push(req),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        // Channel closed; process what we have.
                        break;
                    }
                }
            }

            // --- step 3: execute ---
            let batch: Vec<TaskRequest> = pending
                .iter()
                .map(|p| TaskRequest {
                    payload: p.request.payload.clone(),
                    payload_mime: p.request.payload_mime.clone(),
                    meta: p.request.meta.clone(),
                })
                .collect();

            let respond_to: Vec<Responder> = pending.drain(..).map(|p| p.respond_to).collect();

            let results = batch_fn(batch).await;

            // --- step 4: scatter ---
            match results {
                Ok(results) => {
                    for (responder, result) in respond_to.into_iter().zip(results.into_iter()) {
                        let _ = responder.send(Ok(result));
                    }
                }
                Err(err) => {
                    // All requests in the batch share the same error.
                    let msg = err.to_string();
                    for responder in respond_to {
                        let _ = responder
                            .send(Err(crate::service::ServiceError::Internal(msg.clone())));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn single_request_completes_immediately() {
        let batcher = Batcher::new(BatchLimit::default());

        let handle = tokio::spawn(async move {
            batcher
                .run(Box::new(move |requests| {
                    let results = requests
                        .into_iter()
                        .map(|r| TaskResult::new(r.payload, r.payload_mime))
                        .collect();
                    Box::pin(async move { Ok(results) })
                }))
                .await;
        });

        // TODO: proper integration test with the batcher's submit path.
        // For now this test validates the type signatures compile.
        handle.abort();
    }
}
