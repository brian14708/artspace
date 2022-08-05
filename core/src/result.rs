use thiserror::Error;

use crate::ort;

#[derive(Error, Debug)]
pub enum Error {
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("tsar error")]
    Tsar(#[from] tsar::Error),
    #[error("json error")]
    Json(#[from] serde_json::Error),
    #[error("ort error")]
    Ort(#[from] ort::Error),
    #[error("tsar error")]
    Tokenizer(Box<dyn std::error::Error + Send + Sync>),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported {0} model: {1}")]
    UnsupportedModel(String, String),
    #[error("unknown error")]
    Unknown,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
