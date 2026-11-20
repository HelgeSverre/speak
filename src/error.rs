use std::io;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Usage(String),
    #[error("{backend} voice '{name}' is not built. Run: {hint}")]
    VoiceMissing {
        backend: String,
        name: String,
        hint: String,
    },
    #[error("{file}: {why}")]
    DataInvalid { file: String, why: String },
    #[error("{what}")]
    Unsupported { what: String },
    #[error("{cause}")]
    Audio { cause: String },
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("synthesis failed: {0}")]
    Synthesis(String),
    #[error("{0}")]
    Config(String),
    /// SIGINT: audio stopped, output file finalized.
    #[error("interrupted")]
    Interrupted,
}

impl Error {
    pub fn usage(msg: impl Into<String>) -> Self {
        Error::Usage(msg.into())
    }

    pub fn audio(msg: impl Into<String>) -> Self {
        Error::Audio { cause: msg.into() }
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported { what: msg.into() }
    }

    pub fn invalid(file: impl Into<String>, why: impl Into<String>) -> Self {
        Error::DataInvalid {
            file: file.into(),
            why: why.into(),
        }
    }

    /// Process exit code (spec §3.5, §15).
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::Usage(_) | Error::Unsupported { .. } | Error::Config(_) => 2,
            Error::VoiceMissing { .. } | Error::DataInvalid { .. } => 3,
            Error::Audio { .. } => 4,
            Error::Io(_) | Error::Synthesis(_) => 1,
            Error::Interrupted => 130,
        }
    }
}
