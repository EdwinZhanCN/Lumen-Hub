//! gRPC adapter for the control plane (`lumen.control.v1.Control`): converts
//! the protocol-independent `status` types into proto messages and bridges the
//! broadcast bus into server streams.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use tokio::sync::{broadcast, mpsc};
use tonic::{Request, Response, Status};

use crate::{
    daemon::proto::lumen::control::v1 as pb,
    status::{LogBuffer, LogEntry, LogLevel, Phase, StatusBus, StatusSnapshot},
};

/// Streams hand out a bounded channel; a forwarder task refills it from the
/// broadcast bus. Slow subscribers that lag are resynced with a fresh snapshot
/// rather than disconnected.
const STREAM_BUFFER: usize = 32;

pub struct ControlGrpcService {
    bus: Arc<StatusBus>,
    logs: Arc<LogBuffer>,
}

impl ControlGrpcService {
    pub fn new(bus: Arc<StatusBus>, logs: Arc<LogBuffer>) -> Self {
        Self { bus, logs }
    }
}

/// Minimal Stream over a tokio mpsc receiver (avoids a tokio-stream wrappers
/// dependency).
pub struct ChannelStream<T>(mpsc::Receiver<T>);

impl<T> tonic::codegen::tokio_stream::Stream for ChannelStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        self.0.poll_recv(cx)
    }
}

type ResponseStream<T> =
    Pin<Box<dyn tonic::codegen::tokio_stream::Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl pb::control_server::Control for ControlGrpcService {
    async fn get_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<pb::StatusSnapshot>, Status> {
        Ok(Response::new(snapshot_to_proto(&self.bus.snapshot())))
    }

    type WatchStatusStream = ResponseStream<pb::StatusSnapshot>;

    async fn watch_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchStatusStream>, Status> {
        let (snapshot, mut updates) = self.bus.subscribe();
        let bus = Arc::clone(&self.bus);
        let (tx, rx) = mpsc::channel(STREAM_BUFFER);

        tokio::spawn(async move {
            if tx.send(Ok(snapshot_to_proto(&snapshot))).await.is_err() {
                return;
            }
            loop {
                match updates.recv().await {
                    Ok(update) => {
                        if tx.send(Ok(snapshot_to_proto(&update))).await.is_err() {
                            return;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Dropped intermediate updates; resync with the latest.
                        if tx
                            .send(Ok(snapshot_to_proto(&bus.snapshot())))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        });

        Ok(Response::new(Box::pin(ChannelStream(rx))))
    }

    type TailLogsStream = ResponseStream<pb::LogEntry>;

    async fn tail_logs(
        &self,
        request: Request<pb::TailLogsRequest>,
    ) -> Result<Response<Self::TailLogsStream>, Status> {
        let request = request.into_inner();
        let min_level = if request.min_level.is_empty() {
            LogLevel::Info
        } else {
            LogLevel::parse(&request.min_level).ok_or_else(|| {
                Status::invalid_argument(format!("unknown log level `{}`", request.min_level))
            })?
        };

        let follow = request.follow;
        let (backlog, mut live) = self.logs.tail(request.backlog_lines as usize, min_level);
        let (tx, rx) = mpsc::channel(STREAM_BUFFER);

        tokio::spawn(async move {
            for entry in backlog {
                if tx.send(Ok(log_to_proto(&entry))).await.is_err() {
                    return;
                }
            }
            if !follow {
                return; // one-shot tail: dropping tx ends the stream
            }
            loop {
                match live.recv().await {
                    Ok(entry) => {
                        if entry.level < min_level {
                            continue;
                        }
                        if tx.send(Ok(log_to_proto(&entry))).await.is_err() {
                            return;
                        }
                    }
                    // Log lag just skips entries; the ring/file keep history.
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        });

        Ok(Response::new(Box::pin(ChannelStream(rx))))
    }
}

fn phase_to_proto(phase: Phase) -> pb::Phase {
    match phase {
        Phase::Starting => pb::Phase::Starting,
        Phase::Downloading => pb::Phase::Downloading,
        Phase::Loading => pb::Phase::Loading,
        Phase::Warmup => pb::Phase::Warmup,
        Phase::Ready => pb::Phase::Ready,
        Phase::Failed => pb::Phase::Failed,
        Phase::Stopping => pb::Phase::Stopping,
    }
}

fn snapshot_to_proto(snapshot: &StatusSnapshot) -> pb::StatusSnapshot {
    pb::StatusSnapshot {
        phase: phase_to_proto(snapshot.phase) as i32,
        version: snapshot.version.clone(),
        profile: snapshot.profile.clone(),
        started_at_unix_ms: snapshot.started_at_unix_ms,
        download: snapshot
            .download
            .as_ref()
            .map(|progress| pb::DownloadProgress {
                model: progress.model.clone(),
                file: progress.file.clone(),
                bytes_done: progress.bytes_done,
                bytes_total: progress.bytes_total,
                files_done: progress.files_done,
                files_total: progress.files_total,
            }),
        services: snapshot
            .services
            .iter()
            .map(|service| pb::ServiceState {
                service: service.service.clone(),
                phase: phase_to_proto(service.phase) as i32,
                error: service.error.clone(),
            })
            .collect(),
        error: snapshot.error.clone(),
        seq: snapshot.seq,
    }
}

fn log_to_proto(entry: &LogEntry) -> pb::LogEntry {
    pb::LogEntry {
        time_unix_ms: entry.time_unix_ms,
        level: entry.level.as_str().to_owned(),
        target: entry.target.clone(),
        message: entry.message.clone(),
        fields: entry.fields.iter().cloned().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status::DownloadProgress;
    use pb::control_server::Control;

    #[tokio::test]
    async fn watch_status_yields_current_snapshot_first() {
        let bus = Arc::new(StatusBus::new("1.2.3".into(), "metal".into()));
        let logs = Arc::new(LogBuffer::new());
        bus.set_download(DownloadProgress {
            model: "bioclip".into(),
            file: "burn/vision.fp32.bpk".into(),
            bytes_done: 5,
            bytes_total: 10,
            files_done: 0,
            files_total: 2,
        });

        let service = ControlGrpcService::new(Arc::clone(&bus), logs);
        let mut stream = service
            .watch_status(Request::new(()))
            .await
            .unwrap()
            .into_inner();

        use tonic::codegen::tokio_stream::StreamExt;
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.phase, pb::Phase::Downloading as i32);
        assert_eq!(first.version, "1.2.3");
        assert_eq!(first.download.unwrap().bytes_done, 5);

        bus.set_phase(Phase::Ready);
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.phase, pb::Phase::Ready as i32);
    }

    #[tokio::test]
    async fn tail_logs_replays_backlog_then_streams() {
        let bus = Arc::new(StatusBus::new("1.2.3".into(), "cpu".into()));
        let logs = Arc::new(LogBuffer::new());
        logs.push(LogEntry {
            time_unix_ms: 1,
            level: LogLevel::Info,
            target: "boot".into(),
            message: "hello".into(),
            fields: vec![("k".into(), "v".into())],
        });

        let service = ControlGrpcService::new(bus, Arc::clone(&logs));
        let mut stream = service
            .tail_logs(Request::new(pb::TailLogsRequest {
                backlog_lines: 10,
                min_level: String::new(),
                follow: true,
            }))
            .await
            .unwrap()
            .into_inner();

        use tonic::codegen::tokio_stream::StreamExt;
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.message, "hello");
        assert_eq!(first.fields.get("k").map(String::as_str), Some("v"));

        logs.push(LogEntry {
            time_unix_ms: 2,
            level: LogLevel::Error,
            target: "run".into(),
            message: "boom".into(),
            fields: Vec::new(),
        });
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.level, "ERROR");
    }

    #[tokio::test]
    async fn tail_logs_without_follow_closes_after_backlog() {
        let bus = Arc::new(StatusBus::new("1.2.3".into(), "cpu".into()));
        let logs = Arc::new(LogBuffer::new());
        logs.push(LogEntry {
            time_unix_ms: 1,
            level: LogLevel::Info,
            target: "boot".into(),
            message: "hello".into(),
            fields: Vec::new(),
        });

        let service = ControlGrpcService::new(bus, Arc::clone(&logs));
        let mut stream = service
            .tail_logs(Request::new(pb::TailLogsRequest {
                backlog_lines: 10,
                min_level: String::new(),
                follow: false,
            }))
            .await
            .unwrap()
            .into_inner();

        use tonic::codegen::tokio_stream::StreamExt;
        assert_eq!(stream.next().await.unwrap().unwrap().message, "hello");
        assert!(stream.next().await.is_none(), "one-shot tail must end");
    }
}
