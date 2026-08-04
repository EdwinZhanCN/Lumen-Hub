use std::{
    net::{SocketAddr, TcpListener},
    sync::{Arc, Mutex},
    time::Duration,
};

use lumen_hub::{
    daemon::control_plane,
    status::{LogBuffer, Phase, StatusBus},
};
use lumen_launcher::daemon::{DaemonError, HubPhase, ReadyWaitConfig, wait_for_ready};

fn available_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind probe socket");
    listener.local_addr().expect("read probe address")
}

#[tokio::test(flavor = "multi_thread")]
async fn launcher_waits_for_ready_and_surfaces_terminal_failure() {
    let addr = available_addr();
    let bus = Arc::new(StatusBus::new("test".to_owned(), "cpu".to_owned()));
    let (server, ready) = control_plane(Arc::clone(&bus), Arc::new(LogBuffer::new()));
    ready.init().await;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server_task = tokio::spawn(async move {
        server
            .serve_with_shutdown(addr, async {
                let _ = shutdown_rx.await;
            })
            .await
    });

    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_for_wait = Arc::clone(&observed);
    let ready_wait = tokio::task::spawn_blocking(move || {
        wait_for_ready(
            &ReadyWaitConfig {
                addr,
                timeout: Duration::from_secs(3),
                interval: Duration::from_millis(20),
            },
            |status| observed_for_wait.lock().unwrap().push(status.phase),
        )
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    bus.set_phase(Phase::Ready);
    ready_wait.await.unwrap().unwrap();
    assert!(observed.lock().unwrap().contains(&HubPhase::Ready));

    bus.fail("model download unavailable".to_owned());
    let failed_wait = tokio::task::spawn_blocking(move || {
        wait_for_ready(
            &ReadyWaitConfig {
                addr,
                timeout: Duration::from_secs(3),
                interval: Duration::from_millis(20),
            },
            |_| {},
        )
    });
    let error = failed_wait.await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        DaemonError::HubStartupFailed(message) if message == "model download unavailable"
    ));

    let _ = shutdown_tx.send(());
    server_task.await.unwrap().unwrap();
}
