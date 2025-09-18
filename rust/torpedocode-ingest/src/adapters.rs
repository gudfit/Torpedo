use crate::{IngestError, RawEvent};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream;
use serde_json::Value;
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::SourceAdapter;

pub struct NdjsonFileSourceAdapter {
    path: PathBuf,
}

impl NdjsonFileSourceAdapter {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl SourceAdapter for NdjsonFileSourceAdapter {
    async fn connect(&mut self) -> Result<(), IngestError> {
        if !self.path.exists() {
            return Err(IngestError::Io(format!(
                "NDJSON source not found at {}",
                self.path.display()
            )));
        }
        Ok(())
    }

    async fn stream<'a>(&'a mut self) -> Result<crate::BoxStream<'a, RawEvent>, IngestError> {
        let file = File::open(&self.path)
            .await
            .map_err(|e| IngestError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut events: Vec<RawEvent> = Vec::new();
        while let Some(res) = lines.next_line().await.transpose() {
            let line = res.map_err(|e| IngestError::Io(e.to_string()))?;
            if line.is_empty() { continue; }
            let parsed: Result<Value, _> = serde_json::from_str(&line);
            if let Ok(payload) = parsed {
                if let Some(ts_s) = payload.get("timestamp").and_then(|v| v.as_str()) {
                    if let Ok(ts) = ts_s.parse::<DateTime<Utc>>() {
                        events.push(RawEvent { timestamp: ts, payload });
                    }
                }
            }
        }
        Ok(Box::new(stream::iter(events.into_iter())))
    }
}

pub struct TcpSocketSourceAdapter {
    addr: String,
}

impl TcpSocketSourceAdapter {
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }
}

#[async_trait]
impl SourceAdapter for TcpSocketSourceAdapter {
    async fn connect(&mut self) -> Result<(), IngestError> {
        if self.addr.is_empty() {
            return Err(IngestError::Io("empty socket address".into()));
        }
        Ok(())
    }

    async fn stream<'a>(&'a mut self) -> Result<crate::BoxStream<'a, RawEvent>, IngestError> {
        Ok(Box::new(stream::empty()))
    }
}

pub use NdjsonFileSourceAdapter as ITCHFileSourceAdapter;
pub use NdjsonFileSourceAdapter as OUCHFileSourceAdapter;
