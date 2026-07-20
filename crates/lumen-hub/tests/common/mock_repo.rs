//! Minimal local model repository: a std-only HTTP server implementing the
//! two Hugging Face endpoints `HfHubClient` consumes, serving files from a
//! fixture directory. Point the hub at it with `LUMEN_MODEL_ENDPOINT`.

#![allow(dead_code)]

use std::{
    collections::HashSet,
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
};

const HF_ORG: &str = "Lumilio-Photos";

pub struct MockRepo {
    port: u16,
    root: PathBuf,
    resolve_hits: Arc<AtomicUsize>,
    missing: Arc<Mutex<HashSet<String>>>,
    corrupt: Arc<Mutex<HashSet<String>>>,
    held: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl MockRepo {
    /// Serves `root` (layout: `root/<model>/<files…>`) on an ephemeral port.
    pub fn serve(root: PathBuf) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock repo");
        let port = listener.local_addr().expect("mock repo addr").port();
        let resolve_hits = Arc::new(AtomicUsize::new(0));
        let missing = Arc::new(Mutex::new(HashSet::new()));
        let corrupt = Arc::new(Mutex::new(HashSet::new()));
        let held = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        {
            let root = root.clone();
            let resolve_hits = Arc::clone(&resolve_hits);
            let missing = Arc::clone(&missing);
            let corrupt = Arc::clone(&corrupt);
            let held = Arc::clone(&held);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let Ok(stream) = stream else { return };
                    // Handle each request on its own thread so a held download
                    // does not block unrelated requests.
                    let root = root.clone();
                    let resolve_hits = Arc::clone(&resolve_hits);
                    let missing = Arc::clone(&missing);
                    let corrupt = Arc::clone(&corrupt);
                    let held = Arc::clone(&held);
                    let stop = Arc::clone(&stop);
                    thread::spawn(move || {
                        let _ = handle(
                            stream,
                            &root,
                            &resolve_hits,
                            &missing,
                            &corrupt,
                            &held,
                            &stop,
                        );
                    });
                }
            });
        }

        Self {
            port,
            root,
            resolve_hits,
            missing,
            corrupt,
            held,
            stop,
        }
    }

    /// Holds every artifact download until [`Self::release_downloads`] — lets
    /// a test subscribe to WatchStatus while the hub is provably mid-download.
    pub fn hold_downloads(&self) {
        self.held.store(true, Ordering::SeqCst);
    }

    pub fn release_downloads(&self) {
        self.held.store(false, Ordering::SeqCst);
    }

    pub fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Number of `resolve/main` file downloads served so far.
    pub fn download_count(&self) -> usize {
        self.resolve_hits.load(Ordering::SeqCst)
    }

    /// Makes any artifact whose remote path ends with `suffix` answer 404.
    pub fn fail_resolve(&self, suffix: &str) {
        self.missing.lock().unwrap().insert(suffix.to_owned());
    }

    /// Makes any artifact whose remote path ends with `suffix` return garbage
    /// bytes instead of the fixture content.
    pub fn corrupt_resolve(&self, suffix: &str) {
        self.corrupt.lock().unwrap().insert(suffix.to_owned());
    }
}

impl Drop for MockRepo {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Unblock the accept loop.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn handle(
    mut stream: TcpStream,
    root: &Path,
    resolve_hits: &AtomicUsize,
    missing: &Mutex<HashSet<String>>,
    corrupt: &Mutex<HashSet<String>>,
    held: &AtomicBool,
    stop: &AtomicBool,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    // Drain headers.
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }

    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_owned();
    let path_only = path.split('?').next().unwrap_or("");

    // GET /api/models/{org}/{model}/tree/main?recursive=1
    if let Some(rest) = path_only.strip_prefix(&format!("/api/models/{HF_ORG}/")) {
        if let Some(model) = rest.strip_suffix("/tree/main") {
            let model_dir = root.join(model);
            if !model_dir.is_dir() {
                return respond(&mut stream, 404, b"model not found");
            }
            let mut entries = Vec::new();
            walk(&model_dir, &model_dir, &mut entries)?;
            let body = serde_json::to_vec(
                &entries
                    .iter()
                    .map(|p| serde_json::json!({ "path": p, "type": "file" }))
                    .collect::<Vec<_>>(),
            )
            .expect("tree json");
            return respond(&mut stream, 200, &body);
        }
    }

    // GET /{org}/{model}/resolve/main/{remote_path}
    if let Some(rest) = path_only.strip_prefix(&format!("/{HF_ORG}/")) {
        if let Some((model, remote_path)) = rest.split_once("/resolve/main/") {
            while held.load(Ordering::SeqCst) && !stop.load(Ordering::SeqCst) {
                thread::sleep(std::time::Duration::from_millis(20));
            }
            let is_missing = missing
                .lock()
                .unwrap()
                .iter()
                .any(|s| remote_path.ends_with(s.as_str()));
            if is_missing {
                return respond(&mut stream, 404, b"injected failure");
            }
            let is_corrupt = corrupt
                .lock()
                .unwrap()
                .iter()
                .any(|s| remote_path.ends_with(s.as_str()));
            if is_corrupt {
                resolve_hits.fetch_add(1, Ordering::SeqCst);
                return respond(&mut stream, 200, b"this is not a valid artifact");
            }
            // remote_path is repo-internal (validated hub-side); simple join.
            let file = root.join(model).join(remote_path);
            return match fs::read(&file) {
                Ok(body) => {
                    resolve_hits.fetch_add(1, Ordering::SeqCst);
                    respond(&mut stream, 200, &body)
                }
                Err(_) => respond(&mut stream, 404, b"file not found"),
            };
        }
    }

    respond(&mut stream, 404, b"unknown route")
}

fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk(base, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(base) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

fn respond(stream: &mut TcpStream, status: u16, body: &[u8]) -> std::io::Result<()> {
    let reason = if status == 200 { "OK" } else { "Not Found" };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}
