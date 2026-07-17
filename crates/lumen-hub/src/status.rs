//! Control-plane status: a process-wide bus carrying the hub lifecycle
//! (phase, download progress, per-service state, fatal errors) plus a bounded
//! ring buffer of structured log entries.
//!
//! Protocol-independent by design — the gRPC adapter in `daemon::control`
//! converts these types to `lumen.control.v1` messages. Everything here is
//! cheap to clone and safe to publish from blocking download threads.

use std::{
    collections::VecDeque,
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tokio::sync::broadcast;

/// Hub/service lifecycle phase. Mirrors `lumen.control.v1.Phase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Starting,
    Downloading,
    Loading,
    Warmup,
    Ready,
    Failed,
    Stopping,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DownloadProgress {
    pub model: String,
    pub file: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub files_done: u32,
    pub files_total: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceState {
    pub service: String,
    pub phase: Phase,
    pub error: String,
}

#[derive(Debug, Clone, Default)]
pub struct StatusSnapshot {
    pub phase: Phase,
    pub version: String,
    pub profile: String,
    pub started_at_unix_ms: i64,
    pub download: Option<DownloadProgress>,
    pub services: Vec<ServiceState>,
    pub error: String,
    pub seq: u64,
}

/// How often byte-level download progress is re-published. State transitions
/// always publish immediately; only the high-frequency byte counter is
/// throttled so a multi-GB download does not flood subscribers.
const DOWNLOAD_PUBLISH_INTERVAL: Duration = Duration::from_millis(500);

struct BusState {
    snapshot: StatusSnapshot,
    last_download_publish: Option<Instant>,
}

/// Process-wide status bus: one authoritative snapshot plus a broadcast of
/// every published change. Subscribers first receive the current snapshot, so
/// they never start blind.
pub struct StatusBus {
    state: Mutex<BusState>,
    tx: broadcast::Sender<StatusSnapshot>,
}

impl StatusBus {
    pub fn new(version: String, profile: String) -> Self {
        let snapshot = StatusSnapshot {
            version,
            profile,
            started_at_unix_ms: unix_ms_now(),
            seq: 1,
            ..StatusSnapshot::default()
        };
        let (tx, _) = broadcast::channel(64);
        Self {
            state: Mutex::new(BusState {
                snapshot,
                last_download_publish: None,
            }),
            tx,
        }
    }

    pub fn snapshot(&self) -> StatusSnapshot {
        self.state
            .lock()
            .expect("status bus poisoned")
            .snapshot
            .clone()
    }

    /// Current snapshot plus a receiver for subsequent changes, taken under
    /// one lock so no update can fall between the two.
    pub fn subscribe(&self) -> (StatusSnapshot, broadcast::Receiver<StatusSnapshot>) {
        let state = self.state.lock().expect("status bus poisoned");
        (state.snapshot.clone(), self.tx.subscribe())
    }

    pub fn set_phase(&self, phase: Phase) {
        self.publish(|snapshot| {
            snapshot.phase = phase;
            if phase != Phase::Downloading {
                snapshot.download = None;
            }
        });
    }

    pub fn fail(&self, error: String) {
        self.publish(|snapshot| {
            snapshot.phase = Phase::Failed;
            snapshot.error = error;
        });
    }

    pub fn set_services(&self, services: Vec<ServiceState>) {
        self.publish(|snapshot| snapshot.services = services);
    }

    /// Byte-level progress; throttled. File/model transitions (a different
    /// file name than the last publish) always go out immediately.
    pub fn set_download(&self, progress: DownloadProgress) {
        let mut state = self.state.lock().expect("status bus poisoned");
        let file_changed =
            state.snapshot.download.as_ref().is_none_or(|current| {
                current.file != progress.file || current.model != progress.model
            });
        let due = state
            .last_download_publish
            .is_none_or(|last| last.elapsed() >= DOWNLOAD_PUBLISH_INTERVAL);
        if !file_changed && !due {
            return;
        }
        state.last_download_publish = Some(Instant::now());
        state.snapshot.phase = Phase::Downloading;
        state.snapshot.download = Some(progress);
        state.snapshot.seq += 1;
        let _ = self.tx.send(state.snapshot.clone());
    }

    fn publish(&self, mutate: impl FnOnce(&mut StatusSnapshot)) {
        let mut state = self.state.lock().expect("status bus poisoned");
        mutate(&mut state.snapshot);
        state.snapshot.seq += 1;
        let _ = self.tx.send(state.snapshot.clone());
    }
}

// ---- Structured log ring ----

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub time_unix_ms: i64,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
    pub fields: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "TRACE" => Some(Self::Trace),
            "DEBUG" => Some(Self::Debug),
            "INFO" => Some(Self::Info),
            "WARN" | "WARNING" => Some(Self::Warn),
            "ERROR" => Some(Self::Error),
            _ => None,
        }
    }
}

const LOG_RING_CAPACITY: usize = 512;

/// Bounded in-memory tail of structured log entries plus a live broadcast.
/// The on-disk log file (written by the supervisor) stays the post-mortem
/// source of truth; this ring only serves `Control.TailLogs`.
pub struct LogBuffer {
    ring: Mutex<VecDeque<LogEntry>>,
    tx: broadcast::Sender<LogEntry>,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl LogBuffer {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            ring: Mutex::new(VecDeque::with_capacity(LOG_RING_CAPACITY)),
            tx,
        }
    }

    pub fn push(&self, entry: LogEntry) {
        let mut ring = self.ring.lock().expect("log ring poisoned");
        if ring.len() == LOG_RING_CAPACITY {
            ring.pop_front();
        }
        ring.push_back(entry.clone());
        drop(ring);
        let _ = self.tx.send(entry);
    }

    /// Backlog (most recent `limit` entries at or above `min_level`) plus a
    /// receiver for live entries, taken under one lock.
    pub fn tail(
        &self,
        limit: usize,
        min_level: LogLevel,
    ) -> (Vec<LogEntry>, broadcast::Receiver<LogEntry>) {
        let ring = self.ring.lock().expect("log ring poisoned");
        let backlog = ring
            .iter()
            .filter(|entry| entry.level >= min_level)
            .rev()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        (backlog, self.tx.subscribe())
    }
}

pub fn unix_ms_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_returns_current_snapshot_and_later_changes() {
        let bus = StatusBus::new("1.0.0".into(), "test".into());
        let (snapshot, mut rx) = bus.subscribe();
        assert_eq!(snapshot.phase, Phase::Starting);
        assert_eq!(snapshot.seq, 1);

        bus.set_phase(Phase::Loading);
        let update = rx.try_recv().expect("phase change broadcast");
        assert_eq!(update.phase, Phase::Loading);
        assert_eq!(update.seq, 2);
    }

    #[test]
    fn download_progress_is_throttled_per_file() {
        let bus = StatusBus::new("1.0.0".into(), "test".into());
        let (_, mut rx) = bus.subscribe();

        let progress = |bytes: u64| DownloadProgress {
            model: "bioclip".into(),
            file: "burn/vision.fp32.bpk".into(),
            bytes_done: bytes,
            bytes_total: 100,
            files_done: 0,
            files_total: 3,
        };

        bus.set_download(progress(1));
        bus.set_download(progress(2)); // throttled: same file, too soon
        assert_eq!(
            rx.try_recv()
                .expect("first publish")
                .download
                .unwrap()
                .bytes_done,
            1
        );
        assert!(rx.try_recv().is_err(), "second update should be throttled");

        // A new file publishes immediately regardless of the interval.
        bus.set_download(DownloadProgress {
            file: "burn/text.fp32.bpk".into(),
            ..progress(0)
        });
        let update = rx.try_recv().expect("file change publish");
        assert_eq!(update.download.unwrap().file, "burn/text.fp32.bpk");
        assert_eq!(update.phase, Phase::Downloading);
    }

    #[test]
    fn failure_carries_error_and_phase() {
        let bus = StatusBus::new("1.0.0".into(), "test".into());
        bus.fail("model repo unreachable".into());
        let snapshot = bus.snapshot();
        assert_eq!(snapshot.phase, Phase::Failed);
        assert_eq!(snapshot.error, "model repo unreachable");
    }

    #[test]
    fn log_ring_keeps_a_bounded_filtered_tail() {
        let buffer = LogBuffer::new();
        for i in 0..600 {
            buffer.push(LogEntry {
                time_unix_ms: i,
                level: if i % 2 == 0 {
                    LogLevel::Info
                } else {
                    LogLevel::Debug
                },
                target: "test".into(),
                message: format!("entry {i}"),
                fields: Vec::new(),
            });
        }
        let (backlog, _) = buffer.tail(10, LogLevel::Info);
        assert_eq!(backlog.len(), 10);
        assert!(backlog.iter().all(|entry| entry.level >= LogLevel::Info));
        assert_eq!(backlog.last().unwrap().message, "entry 598");
        // Oldest entries fell out of the ring.
        let (full, _) = buffer.tail(usize::MAX, LogLevel::Trace);
        assert_eq!(full.len(), 512);
        assert_eq!(full.first().unwrap().message, "entry 88");
    }
}
