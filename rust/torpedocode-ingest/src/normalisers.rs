use crate::{CanonicalEvent, IngestError, RawEvent};
use async_trait::async_trait;
use serde::Deserialize;

use crate::Normaliser;

#[derive(Debug, Deserialize)]
#[serde(tag = "msg_type")]
enum ItchMessage {
    #[serde(rename = "A")] // Add Order
    AddOrder {
        side: String,
        price: f64,
        size: f64,
        level: Option<u32>,
    },
    #[serde(rename = "F")] // Add Order with MPID Attribution
    AddAttributed {
        side: String,
        price: f64,
        size: f64,
        level: Option<u32>,
    },
    #[serde(rename = "U")] // Order Replace
    Replace { side: String, price: f64, size: f64, level: Option<u32> },
    #[serde(rename = "E")] // Order Executed
    Execute { side: String, price: f64, size: f64 },
    #[serde(rename = "X")] // Order Cancel
    Cancel { side: String, price: f64, size: f64 },
}

pub struct NasdaqItchNormaliser;

#[async_trait]
impl Normaliser for NasdaqItchNormaliser {
    async fn normalise(&self, event: RawEvent) -> Result<CanonicalEvent, IngestError> {
        let msg: ItchMessage = serde_json::from_value(event.payload)
            .map_err(|e| IngestError::Parse(e.to_string()))?;
        let (event_type, size, price, level, side) = match msg {
            ItchMessage::AddOrder { side, price, size, level }
            | ItchMessage::AddAttributed { side, price, size, level } => {
                let et = match side.as_str() {
                    "B" | "b" => "LO+",
                    "S" | "s" => "LO-",
                    _ => "LO?",
                };
                (et.to_string(), size, price, level, Some(side))
            }
            ItchMessage::Replace { side, price, size, level } => {
                let et = match side.as_str() {
                    "B" | "b" => "LO+",
                    "S" | "s" => "LO-",
                    _ => "LO?",
                };
                (et.to_string(), size, price, level, Some(side))
            }
            ItchMessage::Execute { side, price, size } => {
                let et = match side.as_str() {
                    "B" | "b" => "MO+",
                    "S" | "s" => "MO-",
                    _ => "MO?",
                };
                (et.to_string(), size, price, None, Some(side))
            }
            ItchMessage::Cancel { side, price, size } => {
                let et = match side.as_str() {
                    "B" | "b" => "CX+",
                    "S" | "s" => "CX-",
                    _ => "CX?",
                };
                (et.to_string(), size, price, None, Some(side))
            }
        };

        Ok(CanonicalEvent {
            timestamp: event.timestamp,
            event_type,
            size,
            price,
            level,
            side,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "msg_type")]
enum OuchMessage {
    #[serde(rename = "O")] // Enter Order
    Enter { side: String, price: f64, size: f64, level: Option<u32> },
    #[serde(rename = "U")] // Replace Order
    Replace { side: String, price: f64, size: f64, level: Option<u32> },
    #[serde(rename = "C")] // Cancel Order
    Cancel { side: String, price: f64, size: f64 },
    #[serde(rename = "E")] // Execution
    Execute { side: String, price: f64, size: f64 },
}

pub struct NasdaqOuchNormaliser;

#[async_trait]
impl Normaliser for NasdaqOuchNormaliser {
    async fn normalise(&self, event: RawEvent) -> Result<CanonicalEvent, IngestError> {
        let msg: OuchMessage = serde_json::from_value(event.payload)
            .map_err(|e| IngestError::Parse(e.to_string()))?;
        let (event_type, size, price, level, side) = match msg {
            OuchMessage::Enter { side, price, size, level }
            | OuchMessage::Replace { side, price, size, level } => {
                let et = match side.as_str() {
                    "B" | "b" => "LO+",
                    "S" | "s" => "LO-",
                    _ => "LO?",
                };
                (et.to_string(), size, price, level, Some(side))
            }
            OuchMessage::Cancel { side, price, size } => {
                let et = match side.as_str() {
                    "B" | "b" => "CX+",
                    "S" | "s" => "CX-",
                    _ => "CX?",
                };
                (et.to_string(), size, price, None, Some(side))
            }
            OuchMessage::Execute { side, price, size } => {
                let et = match side.as_str() {
                    "B" | "b" => "MO+",
                    "S" | "s" => "MO-",
                    _ => "MO?",
                };
                (et.to_string(), size, price, None, Some(side))
            }
        };

        Ok(CanonicalEvent {
            timestamp: event.timestamp,
            event_type,
            size,
            price,
            level,
            side,
        })
    }
}

pub use NasdaqItchNormaliser as ITCHNormaliser;
pub use NasdaqOuchNormaliser as OUCHNormaliser;
