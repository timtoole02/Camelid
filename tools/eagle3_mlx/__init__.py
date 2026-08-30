"""Apple-Silicon training utilities for Camelid's pinned EAGLE-3 head."""

from .contract import (
    DRAFT_VOCAB_SIZE,
    TARGET_VOCAB_SIZE,
    TENSOR_SPECS,
    validate_checkpoint,
)

__all__ = [
    "DRAFT_VOCAB_SIZE",
    "TARGET_VOCAB_SIZE",
    "TENSOR_SPECS",
    "validate_checkpoint",
]
