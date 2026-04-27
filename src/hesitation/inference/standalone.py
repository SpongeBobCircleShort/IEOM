"""Standalone inference module (no external dependencies on main package).

This can be used independently for MATLAB integration or testing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sys

# Try to import torch model, fall back to logistic if unavailable
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class Prediction:
    """Output of hesitation predictor."""

    state: str
    state_probabilities: dict
    future_hesitation_prob: float
    future_correction_prob: float
    confidence: float
    window_size_frames: int = 20
    frame_rate_hz: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class HesitationPredictor:
    """Deterministic interface to hesitation model.

    Load once, call many times. Thread-safe after initialization.
    """

    def __init__(
        self,
        torch_model_path: str | None = None,
        fallback_model_path: str | None = None,
        window_size: int = 20,
        frame_rate_hz: int = 10,
    ) -> None:
        """Initialize predictor with optional pre-trained model.

        Args:
            torch_model_path: Path to .pt checkpoint (if using PyTorch)
            fallback_model_path: Path to .json checkpoint (if using fallback)
            window_size: Expected temporal window size (frames)
            frame_rate_hz: Frame rate (Hz)
        """
        self.window_size = window_size
        self.frame_rate_hz = frame_rate_hz
        self.torch_model = None
        self.fallback_model = None
        self.use_torch = False

        if torch_model_path and TORCH_AVAILABLE:
            try:
                self.torch_model = self._load_torch_model(torch_model_path)
                self.use_torch = True
            except Exception as e:
                print(f"Warning: Could not load torch model: {e}", file=sys.stderr)

        if fallback_model_path and not self.use_torch:
            try:
                self.fallback_model = self._load_fallback_model(fallback_model_path)
            except Exception as e:
                print(f"Warning: Could not load fallback model: {e}", file=sys.stderr)

        if not self.use_torch and not self.fallback_model:
            if torch_model_path or fallback_model_path:
                print(
                    "Warning: No model loaded. Predictor will return dummy predictions.",
                    file=sys.stderr,
                )

    @staticmethod
    def _load_torch_model(path: str) -> Any:
        """Load PyTorch model checkpoint."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint

    @staticmethod
    def _load_fallback_model(path: str) -> Any:
        """Load fallback (logistic) model checkpoint."""
        data = json.loads(Path(path).read_text())
        return data

    def predict_single(self, features: dict) -> Prediction:
        """Predict from single feature vector.

        Args:
            features: Dict with keys: mean_hand_speed, pause_ratio, progress_delta,
                     reversal_count, retry_count, task_step_id, human_robot_distance

        Returns:
            Prediction with state, probabilities, and confidence.
        """
        # Convert dict to feature vector
        feature_vector = self._dict_to_vector(features)

        if self.use_torch and self.torch_model:
            return self._predict_torch(feature_vector)
        elif self.fallback_model:
            return self._predict_fallback(feature_vector)
        else:
            return self._predict_dummy(features)

    @staticmethod
    def _dict_to_vector(features: dict) -> list:
        """Convert feature dict to ordered vector."""
        # Standard feature order (must match training)
        key_order = [
            "mean_hand_speed",
            "pause_ratio",
            "progress_delta",
            "reversal_count",
            "retry_count",
            "task_step_id",
            "human_robot_distance",
        ]
        return [features.get(k, 0.0) for k in key_order]

    def _predict_torch(self, feature_vector: list) -> Prediction:
        """Predict using PyTorch model."""
        import torch

        # Convert to tensor (batch of 1, single window)
        x = torch.tensor([feature_vector], dtype=torch.float32)

        with torch.no_grad():
            output = self.torch_model(x)

        # Unpack multi-head output
        if isinstance(output, tuple) and len(output) == 3:
            state_logits, future_hes_logit, future_corr_logit = output
        else:
            state_logits = output

        # Convert to probabilities
        state_probs_raw = torch.softmax(state_logits, dim=1)[0].numpy()
        future_hes = torch.sigmoid(future_hes_logit)[0].item()
        future_corr = torch.sigmoid(future_corr_logit)[0].item()

        # State names (must match your model)
        state_names = [
            "normal_progress",
            "mild_hesitation",
            "strong_hesitation",
            "correction_rework",
            "ready_for_robot_action",
            "overlap_risk",
        ]

        state_probs = {name: float(prob) for name, prob in zip(state_names, state_probs_raw)}
        predicted_state = state_names[state_probs_raw.argmax()]
        confidence = float(state_probs_raw.max())

        return Prediction(
            state=predicted_state,
            state_probabilities=state_probs,
            future_hesitation_prob=future_hes,
            future_correction_prob=future_corr,
            confidence=confidence,
        )

    def _predict_fallback(self, feature_vector: list) -> Prediction:
        """Predict using fallback logistic model."""
        # Placeholder: implement if needed
        state_names = [
            "normal_progress",
            "mild_hesitation",
            "strong_hesitation",
            "correction_rework",
            "ready_for_robot_action",
            "overlap_risk",
        ]
        state_probs = {name: 1.0 / len(state_names) for name in state_names}

        return Prediction(
            state=state_names[0],
            state_probabilities=state_probs,
            future_hesitation_prob=0.0,
            future_correction_prob=0.0,
            confidence=1.0 / len(state_names),
        )

    def _predict_dummy(self, features: dict) -> Prediction:
        """Return dummy prediction when no model loaded."""
        state_names = [
            "normal_progress",
            "mild_hesitation",
            "strong_hesitation",
            "correction_rework",
            "ready_for_robot_action",
            "overlap_risk",
        ]
        state_probs = {name: 1.0 / len(state_names) for name in state_names}

        return Prediction(
            state=state_names[0],
            state_probabilities=state_probs,
            future_hesitation_prob=0.0,
            future_correction_prob=0.0,
            confidence=1.0 / len(state_names),
        )

    @classmethod
    def load_default(cls) -> HesitationPredictor:
        """Load predictor with default paths (if models exist)."""
        # Try to find default model paths
        model_dir = Path(__file__).parent.parent / "artifacts"

        torch_path = None
        fallback_path = None

        if (model_dir / "deep_model.pt").exists():
            torch_path = str(model_dir / "deep_model.pt")
        if (model_dir / "deep_model.json").exists():
            fallback_path = str(model_dir / "deep_model.json")

        return cls(torch_model_path=torch_path, fallback_model_path=fallback_path)
