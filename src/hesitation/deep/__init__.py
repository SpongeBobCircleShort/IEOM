"""Deep temporal modeling utilities for Phase 3.5."""

from hesitation.deep.pipeline import (
    compare_models,
    compare_models_multiseed,
    evaluate_deep,
    evaluate_deep_calibrated,
    infer_sequence_deep,
    train_deep,
    train_deep_multiseed,
    tune_thresholds,
)

__all__ = [
    "train_deep",
    "evaluate_deep",
    "evaluate_deep_calibrated",
    "infer_sequence_deep",
    "tune_thresholds",
    "train_deep_multiseed",
    "compare_models",
    "compare_models_multiseed",
]
