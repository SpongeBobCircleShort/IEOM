from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hesitation.evaluation.metrics import binary_metrics, multiclass_metrics
from hesitation.ml.dataset import DatasetRow, load_rows, split_train_val
from hesitation.ml.logistic import StandardScaler
from hesitation.schemas.events import FrameObservation
from hesitation.schemas.labels import HesitationState

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in no-torch environments
    torch = None
    nn = None
    F = None

STATE_CLASSES = [state.value for state in HesitationState]
SEQUENCE_FEATURE_ORDER = [
    "hand_x",
    "hand_y",
    "hand_speed",
    "hand_accel",
    "distance_to_robot_workspace",
    "progress",
    "confidence",
    "task_step_id",
    "is_dropout",
]
DEFAULT_THRESHOLDS = {
    "future_hesitation": 0.5,
    "future_correction": 0.5,
}


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for deep workflows. Install with `pip install -e \".[dev,deep]\"`.")


if nn is not None:
    class GRURiskModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, state_dim: int) -> None:
            super().__init__()
            self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.state_head = nn.Linear(hidden_size, state_dim)
            self.future_hesitation_head = nn.Linear(hidden_size, 1)
            self.future_correction_head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            _, hidden = self.encoder(x)
            encoded = hidden[-1]
            return {
                "state_logits": self.state_head(encoded),
                "future_hesitation_logits": self.future_hesitation_head(encoded).squeeze(-1),
                "future_correction_logits": self.future_correction_head(encoded).squeeze(-1),
            }
else:  # pragma: no cover - exercised in no-torch environments
    class GRURiskModel:  # type: ignore[override]
        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_torch()


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sequence_features(row: DatasetRow) -> list[float]:
    return [
        float(row["hand_x"]),
        float(row["hand_y"]),
        float(row["hand_speed"]),
        float(row["hand_accel"]),
        float(row["distance_to_robot_workspace"]),
        float(row["progress"]),
        float(row["confidence"]),
        float(row["task_step_id"]),
        float(int(bool(row.get("is_dropout", False)))),
    ]


def build_sequence_windows(
    rows: list[DatasetRow],
    window_size: int,
    horizon_frames: int,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    sessions: dict[str, list[DatasetRow]] = {}
    for row in rows:
        sessions.setdefault(str(row["session_id"]), []).append(row)

    for session_id, session_rows in sessions.items():
        session_rows.sort(key=lambda r: int(r["frame_idx"]))
        observations = [FrameObservation.model_validate(r) for r in session_rows]
        states = [str(r.get("latent_state", HesitationState.NORMAL_PROGRESS.value)) for r in session_rows]
        for end in range(window_size, len(observations) - horizon_frames):
            frame_slice = session_rows[end - window_size : end]
            future_slice = states[end : end + horizon_frames]
            windows.append(
                {
                    "session_id": session_id,
                    "end_frame_idx": observations[end - 1].frame_idx,
                    "sequence_features": [_sequence_features(row) for row in frame_slice],
                    "current_state": states[end - 1],
                    "future_hesitation": int(
                        any(
                            state in {HesitationState.MILD_HESITATION.value, HesitationState.STRONG_HESITATION.value}
                            for state in future_slice
                        )
                    ),
                    "future_correction": int(any(state == HesitationState.CORRECTION_REWORK.value for state in future_slice)),
                }
            )
    return windows


def _scale_sequences(sequences: list[list[list[float]]], scaler: StandardScaler) -> list[list[list[float]]]:
    transformed: list[list[list[float]]] = []
    for sequence in sequences:
        transformed.append(scaler.transform(sequence))
    return transformed


def _prepare_datasets(
    rows: list[DatasetRow],
    window_size: int,
    horizon_frames: int,
) -> dict[str, Any]:
    windows = build_sequence_windows(rows, window_size=window_size, horizon_frames=horizon_frames)
    train, val = split_train_val(windows)
    if not train or not val:
        raise ValueError("Deep training requires both train and validation windows. Increase the dataset size or lower the window settings.")

    scaler = StandardScaler.fit([frame for window in train for frame in window["sequence_features"]])  # type: ignore[index]
    train_sequences = _scale_sequences([window["sequence_features"] for window in train], scaler)  # type: ignore[index]
    val_sequences = _scale_sequences([window["sequence_features"] for window in val], scaler)  # type: ignore[index]

    state_to_idx = {name: idx for idx, name in enumerate(STATE_CLASSES)}
    return {
        "scaler": scaler,
        "state_to_idx": state_to_idx,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "y_state_train": [state_to_idx[str(window["current_state"])] for window in train],
        "y_state_val": [str(window["current_state"]) for window in val],
        "y_fh_train": [int(window["future_hesitation"]) for window in train],
        "y_fh_val": [int(window["future_hesitation"]) for window in val],
        "y_fc_train": [int(window["future_correction"]) for window in train],
        "y_fc_val": [int(window["future_correction"]) for window in val],
        "counts": {
            "train_windows": len(train),
            "val_windows": len(val),
        },
    }


def _tensorize(sequences: list[list[list[float]]], labels: list[int]) -> tuple[Any, Any]:
    _require_torch()
    x = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def _tensorize_binary(labels: list[int]) -> Any:
    _require_torch()
    return torch.tensor(labels, dtype=torch.float32)


def _fit_threshold(y_true: list[int], y_prob: list[float]) -> float:
    candidates = sorted({0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, *y_prob})
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_prob, threshold=float(threshold))
        if float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_threshold = float(threshold)
    return best_threshold


def _build_model(hidden_size: int) -> GRURiskModel:
    _require_torch()
    return GRURiskModel(
        input_size=len(SEQUENCE_FEATURE_ORDER),
        hidden_size=hidden_size,
        state_dim=len(STATE_CLASSES),
    )


def _run_model(model: GRURiskModel, sequences: list[list[list[float]]]) -> dict[str, list[Any]]:
    _require_torch()
    x = torch.tensor(sequences, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        state_probs = torch.softmax(outputs["state_logits"], dim=1).tolist()
        state_pred = torch.argmax(outputs["state_logits"], dim=1).tolist()
        future_hesitation = torch.sigmoid(outputs["future_hesitation_logits"]).tolist()
        future_correction = torch.sigmoid(outputs["future_correction_logits"]).tolist()
    return {
        "state_probs": state_probs,
        "state_pred": [STATE_CLASSES[int(idx)] for idx in state_pred],
        "future_hesitation": [float(value) for value in future_hesitation],
        "future_correction": [float(value) for value in future_correction],
    }


def _load_checkpoint(model_path: str | Path) -> dict[str, Any]:
    _require_torch()
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    model = _build_model(hidden_size=int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["model_state"])
    scaler = StandardScaler(
        means=[float(value) for value in checkpoint["scaler"]["means"]],
        stds=[float(value) for value in checkpoint["scaler"]["stds"]],
    )
    return {
        "checkpoint": checkpoint,
        "model": model,
        "scaler": scaler,
    }


def _prepare_eval_sequences(
    rows: list[DatasetRow],
    checkpoint: dict[str,
    Any],
    scaler: StandardScaler
) -> dict[str, Any]:
    windows = build_sequence_windows(
        rows,
        window_size=int(checkpoint["window_size"]),
        horizon_frames=int(checkpoint["horizon_frames"]),
    )
    _, val = split_train_val(windows)
    if not val:
        raise ValueError("Deep evaluation requires validation windows. Increase the dataset size or lower the window settings.")
    sequences = _scale_sequences([window["sequence_features"] for window in val], scaler)  # type: ignore[index]
    return {
        "sequences": sequences,
        "y_state": [str(window["current_state"]) for window in val],
        "y_fh": [int(window["future_hesitation"]) for window in val],
        "y_fc": [int(window["future_correction"]) for window in val],
        "count": len(val),
    }


def _evaluate_predictions(
    predictions: dict[str, list[Any]],
    y_state: list[str],
    y_fh: list[int],
    y_fc: list[int],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_thresholds = thresholds or DEFAULT_THRESHOLDS
    return {
        "current_state_deep": multiclass_metrics(y_state, predictions["state_pred"], STATE_CLASSES),
        "future_hesitation": binary_metrics(
            y_fh,
            predictions["future_hesitation"],
            active_thresholds["future_hesitation"]
        ),
        "future_correction": binary_metrics(
            y_fc,
            predictions["future_correction"],
            active_thresholds["future_correction"]
        ),
        "thresholds": active_thresholds,
        "windows": len(y_state),
    }


def train_deep(
    input_path: str,
    output_dir: str,
    window_size: int,
    horizon_frames: int,
    seed: int = 7,
    hidden_size: int = 24,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1e-2,
) -> dict[str, Any]:
    _require_torch()
    torch.manual_seed(seed)

    prepared = _prepare_datasets(
        load_rows(input_path),
        window_size=window_size,
        horizon_frames=horizon_frames
    )
    x_train, y_state_train = _tensorize(prepared["train_sequences"], prepared["y_state_train"])
    y_fh_train = _tensorize_binary(prepared["y_fh_train"])
    y_fc_train = _tensorize_binary(prepared["y_fc_train"])

    model = _build_model(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        permutation = torch.randperm(x_train.size(0))
        model.train()
        for start in range(0, x_train.size(0), batch_size):
            batch_indices = permutation[start : start + batch_size]
            outputs = model(x_train[batch_indices])
            loss = (
                F.cross_entropy(outputs["state_logits"], y_state_train[batch_indices])
                + F.binary_cross_entropy_with_logits(outputs["future_hesitation_logits"], y_fh_train[batch_indices])
                + F.binary_cross_entropy_with_logits(outputs["future_correction_logits"], y_fc_train[batch_indices])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    validation_predictions = _run_model(model, prepared["val_sequences"])
    metrics = _evaluate_predictions(
        validation_predictions,
        prepared["y_state_val"],
        prepared["y_fh_val"],
        prepared["y_fc_val"],
        thresholds=DEFAULT_THRESHOLDS,
    )
    metrics["counts"] = prepared["counts"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden_size": hidden_size,
            "feature_order": SEQUENCE_FEATURE_ORDER,
            "state_classes": STATE_CLASSES,
            "window_size": window_size,
            "horizon_frames": horizon_frames,
            "seed": seed,
            "scaler": {
                "means": prepared["scaler"].means,
                "stds": prepared["scaler"].stds,
            },
        },
        output / "deep_model.pt",
    )
    _save_json(output / "deep_metrics.json", metrics)
    return metrics


def evaluate_deep(
    input_path: str,
    model_path: str,
    output_path: str | None = None
) -> dict[str, Any]:
    runtime = _load_checkpoint(model_path)
    prepared = _prepare_eval_sequences(
        load_rows(input_path),
        runtime["checkpoint"],
        runtime["scaler"]
    )
    predictions = _run_model(runtime["model"], prepared["sequences"])
    metrics = _evaluate_predictions(
        predictions,
        prepared["y_state"],
        prepared["y_fh"],
        prepared["y_fc"]
    )
    if output_path:
        _save_json(output_path, metrics)
    return metrics


def tune_thresholds(
    input_path: str,
    model_path: str,
    output_path: str | None = None
) -> dict[str, float]:
    runtime = _load_checkpoint(model_path)
    prepared = _prepare_eval_sequences(
        load_rows(input_path),
        runtime["checkpoint"],
        runtime["scaler"]
    )
    predictions = _run_model(runtime["model"], prepared["sequences"])
    thresholds = {
        "future_hesitation": _fit_threshold(prepared["y_fh"], predictions["future_hesitation"]),
        "future_correction": _fit_threshold(prepared["y_fc"], predictions["future_correction"]),
    }
    if output_path:
        _save_json(output_path, thresholds)
    return thresholds


def evaluate_deep_calibrated(
    input_path: str,
    model_path: str,
    threshold_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    runtime = _load_checkpoint(model_path)
    thresholds = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    prepared = _prepare_eval_sequences(
        load_rows(input_path),
        runtime["checkpoint"],
        runtime["scaler"]
    )
    predictions = _run_model(runtime["model"], prepared["sequences"])
    metrics = _evaluate_predictions(
        predictions,
        prepared["y_state"],
        prepared["y_fh"],
        prepared["y_fc"],
        thresholds={
            "future_hesitation": float(thresholds["future_hesitation"]),
            "future_correction": float(thresholds["future_correction"]),
        },
    )
    if output_path:
        _save_json(output_path, metrics)
    return metrics
