//! Dynamic request batching for the daemon layer.
//!
//! The daemon batches only complete, already reassembled requests. Transport
//! chunking is handled before a request enters this layer.

use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::service::{BatchKey, ServiceError, ServiceResult, TaskRequest, TaskResult};
use tokio::{sync::mpsc, time};

pub type Responder = tokio::sync::oneshot::Sender<ServiceResult<TaskResult>>;
pub type ResponseReceiver = tokio::sync::oneshot::Receiver<ServiceResult<TaskResult>>;

pub type BatchFn = Arc<
    dyn Fn(Vec<TaskRequest>) -> Pin<Box<dyn Future<Output = ServiceResult<Vec<TaskResult>>> + Send>>
        + Send
        + Sync,
>;

pub struct PendingRequest {
    pub request: TaskRequest,
    pub respond_to: Responder,
}

#[derive(Debug, Clone)]
pub struct BatcherConfig {
    pub enabled: bool,
    pub max_batch_size: usize,
    pub queue_latency: Duration,
}

impl BatcherConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            max_batch_size: 1,
            queue_latency: Duration::from_millis(1),
        }
    }
}

#[derive(Clone)]
pub struct Batcher {
    config: BatcherConfig,
    queues: Arc<Mutex<HashMap<BatchKey, mpsc::UnboundedSender<PendingRequest>>>>,
}

impl Batcher {
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            queues: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn config(&self) -> &BatcherConfig {
        &self.config
    }

    pub fn submit(
        &self,
        key: BatchKey,
        request: TaskRequest,
        batch_fn: BatchFn,
    ) -> ServiceResult<ResponseReceiver> {
        let tx = self.queue_sender(key, batch_fn)?;
        let (respond_to, receiver) = tokio::sync::oneshot::channel();
        tx.send(PendingRequest {
            request,
            respond_to,
        })
        .map_err(|_| ServiceError::Unavailable("batch queue is closed".to_owned()))?;
        Ok(receiver)
    }

    fn queue_sender(
        &self,
        key: BatchKey,
        batch_fn: BatchFn,
    ) -> ServiceResult<mpsc::UnboundedSender<PendingRequest>> {
        let mut queues = self
            .queues
            .lock()
            .map_err(|_| ServiceError::Internal("batch queue registry lock poisoned".to_owned()))?;

        if let Some(tx) = queues.get(&key) {
            return Ok(tx.clone());
        }

        let (tx, rx) = mpsc::unbounded_channel();
        queues.insert(key, tx.clone());

        let max_batch_size = self.config.max_batch_size.max(1);
        let queue_latency = self.config.queue_latency;
        tokio::spawn(run_queue(rx, max_batch_size, queue_latency, batch_fn));

        Ok(tx)
    }
}

async fn run_queue(
    mut rx: mpsc::UnboundedReceiver<PendingRequest>,
    max_batch_size: usize,
    queue_latency: Duration,
    batch_fn: BatchFn,
) {
    loop {
        let Some(first) = rx.recv().await else {
            break;
        };

        let mut pending = Vec::with_capacity(max_batch_size);
        push_if_open(&mut pending, first);

        let deadline = time::sleep(queue_latency);
        tokio::pin!(deadline);

        while pending.len() < max_batch_size {
            tokio::select! {
                _ = &mut deadline => break,
                received = rx.recv() => {
                    match received {
                        Some(request) => push_if_open(&mut pending, request),
                        None => break,
                    }
                }
            }
        }

        pending.retain(|request| !request.respond_to.is_closed());
        if pending.is_empty() {
            continue;
        }

        let batch_size = pending.len();
        let started = std::time::Instant::now();
        flush(pending, &batch_fn).await;
        tracing::debug!(
            batch_size,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "batch flushed"
        );
    }
}

fn push_if_open(pending: &mut Vec<PendingRequest>, request: PendingRequest) {
    if !request.respond_to.is_closed() {
        pending.push(request);
    }
}

async fn flush(pending: Vec<PendingRequest>, batch_fn: &BatchFn) {
    let batch = pending
        .iter()
        .map(|pending| pending.request.clone())
        .collect::<Vec<_>>();
    let respond_to = pending
        .into_iter()
        .map(|pending| pending.respond_to)
        .collect::<Vec<_>>();

    match batch_fn(batch).await {
        Ok(results) if results.len() == respond_to.len() => {
            for (responder, result) in respond_to.into_iter().zip(results) {
                let _ = responder.send(Ok(result));
            }
        }
        Ok(results) => {
            let message = format!(
                "batch handler returned {} results for {} requests",
                results.len(),
                respond_to.len()
            );
            for responder in respond_to {
                let _ = responder.send(Err(ServiceError::Internal(message.clone())));
            }
        }
        Err(err) => {
            let message = err.to_string();
            for responder in respond_to {
                let _ = responder.send(Err(ServiceError::Internal(message.clone())));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::sync::Notify;

    fn echo_batch_fn() -> BatchFn {
        Arc::new(|requests| {
            Box::pin(async move {
                Ok(requests
                    .into_iter()
                    .map(|request| TaskResult::new(request.payload, request.payload_mime))
                    .collect())
            })
        })
    }

    #[tokio::test]
    async fn single_request_flushes_after_latency() {
        let batcher = Batcher::new(BatcherConfig {
            enabled: true,
            max_batch_size: 8,
            queue_latency: Duration::from_millis(1),
        });
        let receiver = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"one"), "text/plain"),
                echo_batch_fn(),
            )
            .unwrap();

        let result = receiver.await.unwrap().unwrap();
        assert_eq!(result.payload, Bytes::from_static(b"one"));
    }

    #[tokio::test]
    async fn max_batch_size_flushes_immediately() {
        let notify = Arc::new(Notify::new());
        let batches = Arc::new(AtomicUsize::new(0));
        let batch_fn: BatchFn = {
            let notify = Arc::clone(&notify);
            let batches = Arc::clone(&batches);
            Arc::new(move |requests| {
                let notify = Arc::clone(&notify);
                let batches = Arc::clone(&batches);
                Box::pin(async move {
                    batches.fetch_add(1, Ordering::SeqCst);
                    notify.notify_waiters();
                    Ok(requests
                        .into_iter()
                        .map(|request| TaskResult::new(request.payload, request.payload_mime))
                        .collect())
                })
            })
        };
        let batcher = Batcher::new(BatcherConfig {
            enabled: true,
            max_batch_size: 2,
            queue_latency: Duration::from_secs(60),
        });

        let r1 = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"one"), "text/plain"),
                Arc::clone(&batch_fn),
            )
            .unwrap();
        let r2 = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"two"), "text/plain"),
                batch_fn,
            )
            .unwrap();

        notify.notified().await;
        assert_eq!(batches.load(Ordering::SeqCst), 1);
        assert_eq!(
            r1.await.unwrap().unwrap().payload,
            Bytes::from_static(b"one")
        );
        assert_eq!(
            r2.await.unwrap().unwrap().payload,
            Bytes::from_static(b"two")
        );
    }

    #[tokio::test]
    async fn different_keys_use_different_queues() {
        let calls = Arc::new(AtomicUsize::new(0));
        let batch_fn: BatchFn = {
            let calls = Arc::clone(&calls);
            Arc::new(move |requests| {
                let calls = Arc::clone(&calls);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(requests
                        .into_iter()
                        .map(|request| TaskResult::new(request.payload, request.payload_mime))
                        .collect())
                })
            })
        };
        let batcher = Batcher::new(BatcherConfig {
            enabled: true,
            max_batch_size: 8,
            queue_latency: Duration::from_millis(1),
        });

        let r1 = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"one"), "text/plain"),
                Arc::clone(&batch_fn),
            )
            .unwrap();
        let r2 = batcher
            .submit(
                BatchKey::new("b"),
                TaskRequest::new(Bytes::from_static(b"two"), "text/plain"),
                batch_fn,
            )
            .unwrap();

        let _ = r1.await.unwrap().unwrap();
        let _ = r2.await.unwrap().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn dropped_receiver_is_not_included_in_batch() {
        let seen = Arc::new(AtomicUsize::new(0));
        let batch_fn: BatchFn = {
            let seen = Arc::clone(&seen);
            Arc::new(move |requests| {
                let seen = Arc::clone(&seen);
                Box::pin(async move {
                    seen.store(requests.len(), Ordering::SeqCst);
                    Ok(requests
                        .into_iter()
                        .map(|request| TaskResult::new(request.payload, request.payload_mime))
                        .collect())
                })
            })
        };
        let batcher = Batcher::new(BatcherConfig {
            enabled: true,
            max_batch_size: 2,
            queue_latency: Duration::from_millis(1),
        });

        let dropped = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"drop"), "text/plain"),
                Arc::clone(&batch_fn),
            )
            .unwrap();
        drop(dropped);
        let kept = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"keep"), "text/plain"),
                batch_fn,
            )
            .unwrap();

        let result = kept.await.unwrap().unwrap();
        assert_eq!(result.payload, Bytes::from_static(b"keep"));
        assert_eq!(seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mismatched_result_count_returns_internal_error() {
        let batch_fn: BatchFn = Arc::new(|_requests| Box::pin(async move { Ok(Vec::new()) }));
        let batcher = Batcher::new(BatcherConfig {
            enabled: true,
            max_batch_size: 1,
            queue_latency: Duration::from_secs(60),
        });

        let receiver = batcher
            .submit(
                BatchKey::new("a"),
                TaskRequest::new(Bytes::from_static(b"one"), "text/plain"),
                batch_fn,
            )
            .unwrap();

        let err = receiver.await.unwrap().unwrap_err();
        assert!(matches!(err, ServiceError::Internal(_)));
    }
}
