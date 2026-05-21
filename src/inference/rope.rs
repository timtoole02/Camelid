use std::env;

use crate::{BackendError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    AdjacentEvenOdd,
    SplitHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePositionMode {
    ZeroBased,
    OneBased,
}

impl RopePairing {
    pub fn label(self) -> &'static str {
        match self {
            Self::AdjacentEvenOdd => "adjacent_even_odd",
            Self::SplitHalf => "split_half",
        }
    }
}

impl RopeDirection {
    pub fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Inverse => "inverse",
        }
    }
}

impl RopePositionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ZeroBased => "zero_based",
            Self::OneBased => "one_based",
        }
    }

    pub(super) fn effective_position(self, position: usize) -> usize {
        match self {
            Self::ZeroBased => position,
            Self::OneBased => position + 1,
        }
    }
}

pub fn diagnostic_rope_pairing() -> Result<RopePairing> {
    match env::var("CAMELID_ROPE_PAIRING") {
        Ok(value) if value == "split_half" => Ok(RopePairing::SplitHalf),
        Ok(value) if value == "adjacent_even_odd" || value.is_empty() => {
            Ok(RopePairing::AdjacentEvenOdd)
        }
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ROPE_PAIRING {value:?}; expected adjacent_even_odd or split_half"
        ))),
        Err(env::VarError::NotPresent) => Ok(RopePairing::AdjacentEvenOdd),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ROPE_PAIRING: {err}"
        ))),
    }
}

pub fn diagnostic_rope_direction() -> Result<RopeDirection> {
    match env::var("CAMELID_ROPE_DIRECTION") {
        Ok(value) if value == "inverse" => Ok(RopeDirection::Inverse),
        Ok(value) if value == "forward" || value.is_empty() => Ok(RopeDirection::Forward),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ROPE_DIRECTION {value:?}; expected forward or inverse"
        ))),
        Err(env::VarError::NotPresent) => Ok(RopeDirection::Forward),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ROPE_DIRECTION: {err}"
        ))),
    }
}

pub fn diagnostic_rope_position_mode() -> Result<RopePositionMode> {
    match env::var("CAMELID_ROPE_POSITION_MODE") {
        Ok(value) if value == "one_based" => Ok(RopePositionMode::OneBased),
        Ok(value) if value == "zero_based" || value.is_empty() => Ok(RopePositionMode::ZeroBased),
        Ok(value) => Err(BackendError::InvalidModelMetadata(format!(
            "unsupported CAMELID_ROPE_POSITION_MODE {value:?}; expected zero_based or one_based"
        ))),
        Err(env::VarError::NotPresent) => Ok(RopePositionMode::ZeroBased),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid CAMELID_ROPE_POSITION_MODE: {err}"
        ))),
    }
}
