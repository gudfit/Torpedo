//! Streaming ingestion utilities for TorpedoCode.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod adapters;
pub mod normalisers;

type BoxStream<'a, T> = Box<dyn Stream<Item = T> + Send + Unpin + 'a>;

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvent {
    pub timestamp: DateTime<Utc>,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub size: f64,
    pub price: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side: Option<String>,
}

#[async_trait]
pub trait SourceAdapter {
    async fn connect(&mut self) -> Result<(), IngestError>;
    async fn stream<'a>(&'a mut self) -> Result<BoxStream<'a, RawEvent>, IngestError>;
}

#[async_trait]
pub trait Normaliser {
    async fn normalise(&self, event: RawEvent) -> Result<CanonicalEvent, IngestError>;
}

pub struct Pipeline<S, N>
where
    S: SourceAdapter + Send,
    N: Normaliser + Send + Sync,
{
    pub source: S,
    pub normaliser: N,
}

impl<S, N> Pipeline<S, N>
where
    S: SourceAdapter + Send,
    N: Normaliser + Send + Sync,
{
    pub fn new(source: S, normaliser: N) -> Self {
        Self { source, normaliser }
    }

    pub async fn run(&mut self) -> Result<Vec<CanonicalEvent>, IngestError> {
        self.source.connect().await?;
        let mut stream = self.source.stream().await?;
        let mut results = Vec::new();
        while let Some(event) = stream.next().await {
            let canonical = self.normaliser.normalise(event).await?;
            results.push(canonical);
        }
        Ok(results)
    }

    pub async fn run_to_ndjson<W>(&mut self, mut writer: W) -> Result<u64, IngestError>
    where
        W: tokio::io::AsyncWrite + Unpin + Send,
    {
        use tokio::io::AsyncWriteExt;

        self.source.connect().await?;
        let mut stream = self.source.stream().await?;
        let mut count = 0u64;
        while let Some(event) = stream.next().await {
            let canonical = self.normaliser.normalise(event).await?;
            let line = serde_json::to_string(&canonical)
                .map_err(|e| IngestError::Parse(e.to_string()))?;
            writer
                .write_all(line.as_bytes())
                .await
                .map_err(|e| IngestError::Io(e.to_string()))?;
            writer
                .write_all(b"\n")
                .await
                .map_err(|e| IngestError::Io(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }
}
