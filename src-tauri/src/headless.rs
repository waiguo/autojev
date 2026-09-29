//! Headless gateway entry: runs the routing gateway without a desktop window.
//!
//! Upstream AutoJev is a Tauri desktop app — `run()` builds a webview and a tray icon, and
//! the gateway binds loopback only. This module lets the same gateway run as a plain server
//! (Docker/systemd) so it can be reached over the network.
//!
//! Configuration lives in the same SQLite database the GUI uses, so a profile exported from
//! a desktop install can be mounted in as `~/.autojev/autojev.db`.
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::{
    config::{ConfigStore, DEFAULT_PORT},
    lifecycle, performance,
};

/// Returns true when the process was started as the headless gateway, in which case `run()`
/// must not build the Tauri app. Mirrors `lifecycle::watchdog_entry`'s argument sniffing.
pub fn entry() -> bool {
    if !std::env::args().any(|arg| arg == "--headless") {
        return false;
    }
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("AutoJev headless runtime: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = runtime.block_on(serve()) {
        eprintln!("AutoJev headless gateway stopped: {error:#}");
        std::process::exit(1);
    }
    true
}

async fn serve() -> Result<()> {
    let root = lifecycle::root()?;
    let store = Arc::new(ConfigStore::load(root.join("autojev.db"))?);

    let port: u16 = std::env::var("AUTOJEV_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let host = std::env::var("AUTOJEV_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    if store.read().port != port {
        store.update(|config| config.port = port)?;
    }
    // Same instance lock the desktop app uses, so a GUI and this cannot share a port.
    let _lock = lifecycle::lock(port)?;

    // Held for the process lifetime — dropping the handle shuts the listener down.
    // No crash watchdog: a container restart policy already covers process death, and the
    // watchdog's agent-config restore is a desktop concern this mode never touches.
    let _gateway = crate::proxy::start_on(store.clone(), &host).await?;

    // Upstream's probe loop bills real provider requests and is normally only reachable once
    // enabled from the desktop UI. Off by default on a server; opt in with AUTOJEV_PERFORMANCE=1.
    if std::env::var("AUTOJEV_PERFORMANCE").as_deref() == Ok("1") {
        tokio::spawn(performance::schedule(
            store.clone(),
            Arc::new(performance::Runner::default()),
        ));
    }

    let address = format!("{host}:{port}");
    address
        .parse::<std::net::SocketAddr>()
        .context("invalid AUTOJEV_HOST/AUTOJEV_PORT")?;
    eprintln!("AutoJev headless gateway listening on http://{address}");
    // Park forever. tokio's `signal` feature is not enabled upstream, so SIGTERM takes the
    // default disposition and the container stops immediately — fine for a stateless proxy.
    std::future::pending::<()>().await;
    Ok(())
}
