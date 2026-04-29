"""Clean inference wrapper for hesitation model.

Provides deterministic, MATLAB-callable interface to trained model.
Input: feature dict → Output: HesitationPrediction dataclass
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Try to import torch model, fall back to logistic if unavailable
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


CANONICAL_FEATURE_ORDER = [
    "mean_speed",
    "speed_variance",
    "pause_ratio",
    "direction_changes",
    "progress_delta",
    "backtrack_ratio",
    "mean_workspace_distance",
]

STATE_NAMES = [
    "normal_progress",
    "mild_hesitation",
    "strong_hesitation",
    "correction_rework",
    "ready_for_robot_action",
    "overlap_risk",
]


@dataclass
class Prediction:
    """Output of hesitation predictor."""

    state: str
    state_probabilities: dict[str, float]
    future_hesitation_prob: float
    future_correction_prob: float
    confidence: float
    window_size_frames: int = 20
    frame_rate_hz: int = 10

    def to_dict(self) -> dict[str, Any]:
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

    def predict_single(self, features: dict[str, float]) -> Prediction:
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
    def _dict_to_vector(features: dict[str, float]) -> list[float]:
        """Convert feature dict to ordered vector."""
        canonical = HesitationPredictor._canonicalize_features(features)
        return [canonical[name] for name in CANONICAL_FEATURE_ORDER]

    @staticmethod
    def _canonicalize_features(features: dict[str, float]) -> dict[str, float]:
        """Map MATLAB legacy names onto the training-time feature contract."""
        return {
            "mean_speed": float(
                features.get("mean_speed", features.get("mean_hand_speed", 0.0))
            ),
            "speed_variance": float(
                features.get("speed_variance", features.get("hand_speed_variance", 0.0))
            ),
            "pause_ratio": float(features.get("pause_ratio", 0.0)),
            "direction_changes": float(
                features.get("direction_changes", features.get("reversal_count", 0.0))
            ),
            "progress_delta": float(features.get("progress_delta", 0.0)),
            "backtrack_ratio": float(
                features.get(
                    "backtrack_ratio",
                    min(float(features.get("retry_count", 0.0)), 1.0),
                )
            ),
            "mean_workspace_distance": float(
                features.get("mean_workspace_distance", features.get("human_robot_distance", 0.0))
            ),
        }

    def _predict_torch(self, feature_vector: list[float]) -> Prediction:
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
            future_hes_logit = torch.tensor([0.0])
            future_corr_logit = torch.tensor([0.0])

        # Convert to probabilities
        state_probs_raw = torch.softmax(state_logits, dim=1)[0].numpy()
        future_hes = torch.sigmoid(future_hes_logit)[0].item()
        future_corr = torch.sigmoid(future_corr_logit)[0].item()

        state_probs = {
            name: float(prob)
            for name, prob in zip(STATE_NAMES, state_probs_raw, strict=False)
        }
        predicted_state = STATE_NAMES[state_probs_raw.argmax()]
        confidence = float(state_probs_raw.max())

        return Prediction(
            state=predicted_state,
            state_probabilities=state_probs,
            future_hesitation_prob=future_hes,
            future_correction_prob=future_corr,
            confidence=confidence,
        )

    def _predict_fallback(self, feature_vector: list[float]) -> Prediction:
        """Predict using fallback logistic model."""
        if "scaler" not in self.fallback_model or "state" not in self.fallback_model:
            return self._predict_dummy({})

        payload = self.fallback_model
        model_features = self._select_model_features(feature_vector, payload)
        scaled_features = [
            (value - mean) / std
            for value, mean, std in zip(
                model_features,
                payload["scaler"]["means"],
                payload["scaler"]["stds"],
                strict=False,
            )
        ]

        state_names = list(payload["state"]["classes"])
        raw_state_probs = {
            name: self._sigmoid(
                self._dot(payload["state"]["weights"][name], scaled_features)
                + float(payload["state"]["biases"][name])
            )
            for name in state_names
        }
        denom = sum(raw_state_probs.values()) + 1e-9
        state_probs = {name: 0.0 for name in STATE_NAMES}
        state_probs.update({name: value / denom for name, value in raw_state_probs.items()})
        predicted_state = max(raw_state_probs, key=lambda name: state_probs[name])
        confidence = float(state_probs[predicted_state])
        if set(state_names) != set(STATE_NAMES):
            confidence = min(confidence, 0.50)
        future_hes = self._sigmoid(
            self._dot(payload["future_hesitation"]["weights"], scaled_features)
            + float(payload["future_hesitation"]["bias"])
        )
        future_corr = self._sigmoid(
            self._dot(payload["future_correction"]["weights"], scaled_features)
            + float(payload["future_correction"]["bias"])
        )

        return Prediction(
            state=predicted_state,
            state_probabilities=state_probs,
            future_hesitation_prob=float(future_hes),
            future_correction_prob=float(future_corr),
            confidence=confidence,
        )

    @staticmethod
    def _select_model_features(
        feature_vector: list[float],
        payload: dict[str, Any],
    ) -> list[float]:
        feature_order = payload.get("feature_order", CANONICAL_FEATURE_ORDER)
        by_name = dict(zip(CANONICAL_FEATURE_ORDER, feature_vector, strict=False))
        full_vector = [float(by_name.get(name, 0.0)) for name in feature_order]
        feature_indices = payload.get("feature_indices")
        if feature_indices is None:
            return full_vector
        return [full_vector[int(index)] for index in feature_indices]

    @staticmethod
    def _dot(weights: list[float], values: list[float]) -> float:
        return sum(
            float(weight) * float(value)
            for weight, value in zip(weights, values, strict=False)
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def _predict_dummy(self, features: dict[str, float]) -> Prediction:
        """Return dummy prediction when no model loaded."""
        state_names = STATE_NAMES
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
        repo_root = Path(__file__).resolve().parents[3]

        torch_path = None
        fallback_path = None

        if (model_dir / "deep_model.pt").exists():
            torch_path = str(model_dir / "deep_model.pt")
        if (model_dir / "deep_model.json").exists():
            fallback_path = str(model_dir / "deep_model.json")
        for candidate in [
            model_dir / "classical_model.json",
            repo_root / "reports" / "phase1_validation" / "classical_model.json",
            repo_root / "simulations" / "classical_model.json",
        ]:
            if fallback_path is None and candidate.exists():
                fallback_path = str(candidate)

        return cls(torch_model_path=torch_path, fallback_model_path=fallback_path)
