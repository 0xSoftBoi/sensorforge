// Qualia IPC: Unix domain socket control plane for supervisor-runner communication.

pub use qualia_types::*;

use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Control message types
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMsg {
    /// Graceful shutdown of the entire engine.
    Shutdown = 0,
    /// Emergency stop — all layers halt immediately.
    Estop = 1,
    /// Pause a specific layer (payload = layer_id).
    Pause = 2,
    /// Resume a specific layer (payload = layer_id).
    Resume = 3,
    /// Request status (payload = optional layer_id, or 0xFF for all).
    Status = 4,
}

impl ControlMsg {
    fn from_u8(v: u8) -> io::Result<Self> {
        match v {
            0 => Ok(ControlMsg::Shutdown),
            1 => Ok(ControlMsg::Estop),
            2 => Ok(ControlMsg::Pause),
            3 => Ok(ControlMsg::Resume),
            4 => Ok(ControlMsg::Status),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown control message type: {}", v),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// ControlListener — server side (supervisor)
// ---------------------------------------------------------------------------

pub struct ControlListener {
    listener: UnixListener,
    path: PathBuf,
}

impl ControlListener {
    /// Bind a Unix domain socket at `path`. Removes any stale socket first.
    pub fn bind<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        // Remove stale socket from a previous run.
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path)?;
        listener.set_nonblocking(true)?;
        Ok(Self { listener, path })
    }

    /// Try to accept a connection without blocking. Returns `None` if no
    /// connection is pending.
    pub fn try_accept(&self) -> Option<ControlStream> {
        self.listener
            .accept()
            .ok()
            .map(|(stream, _)| ControlStream { stream })
    }

    /// Accept a connection, blocking until one arrives.
    pub fn accept_blocking(&self) -> io::Result<ControlStream> {
        self.listener.set_nonblocking(false)?;
        let (stream, _) = self.listener.accept()?;
        self.listener.set_nonblocking(true)?;
        Ok(ControlStream { stream })
    }

    /// Returns the path this listener is bound to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ControlListener {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// ControlStream — bidirectional message stream
// ---------------------------------------------------------------------------

pub struct ControlStream {
    stream: UnixStream,
}

impl ControlStream {
    /// Connect to a supervisor's control socket.
    pub fn connect<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        Ok(Self { stream })
    }

    /// Send a control message with an optional 1-byte payload.
    pub fn send(&mut self, msg: ControlMsg, payload: Option<u8>) -> io::Result<()> {
        self.stream.write_all(&[msg as u8])?;
        if let Some(p) = payload {
            self.stream.write_all(&[p])?;
        }
        self.stream.flush()
    }

    /// Receive a control message. Returns the message type and optional
    /// payload byte.
    pub fn recv(&mut self) -> io::Result<(ControlMsg, Option<u8>)> {
        let mut buf = [0u8; 2];
        let n = self.stream.read(&mut buf)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "control stream closed",
            ));
        }
        let msg = ControlMsg::from_u8(buf[0])?;
        let payload = if n > 1 { Some(buf[1]) } else { None };
        Ok((msg, payload))
    }

    /// Set the stream to non-blocking mode.
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.stream.set_nonblocking(nonblocking)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn roundtrip_control_msg() {
        let sock_path = format!("/tmp/qualia_ipc_test_{}.sock", std::process::id());

        let listener = ControlListener::bind(&sock_path).expect("bind failed");

        let handle = {
            let p = sock_path.clone();
            thread::spawn(move || {
                let mut client = ControlStream::connect(&p).expect("connect failed");
                client
                    .send(ControlMsg::Pause, Some(3))
                    .expect("send failed");
            })
        };

        // Wait a moment for the client to connect.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let mut server_stream = listener.try_accept().expect("no connection");
        let (msg, payload) = server_stream.recv().expect("recv failed");
        assert_eq!(msg, ControlMsg::Pause);
        assert_eq!(payload, Some(3));

        handle.join().unwrap();
    }

    #[test]
    fn shutdown_message() {
        let sock_path = format!("/tmp/qualia_ipc_shutdown_{}.sock", std::process::id());

        let listener = ControlListener::bind(&sock_path).expect("bind failed");

        let handle = {
            let p = sock_path.clone();
            thread::spawn(move || {
                let mut client = ControlStream::connect(&p).expect("connect failed");
                client
                    .send(ControlMsg::Shutdown, None)
                    .expect("send failed");
            })
        };

        std::thread::sleep(std::time::Duration::from_millis(50));

        let mut server_stream = listener.try_accept().expect("no connection");
        let (msg, payload) = server_stream.recv().expect("recv failed");
        assert_eq!(msg, ControlMsg::Shutdown);
        assert!(payload.is_none() || payload == Some(0));

        handle.join().unwrap();
    }
}
