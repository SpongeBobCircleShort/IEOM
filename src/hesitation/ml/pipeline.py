from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hesitation.baselines.rules_engine import classify_window
from hesitation.evaluation.metrics import binary_metrics, multiclass_metrics
from hesitation.io.config import load_config
from hesitation.ml.dataset import FEATURE_ORDER, build_windows, load_rows, split_train_val
from hesitation.ml.logistic import BinaryLogisticRegression, OVRLogisticModel, StandardScaler
from hesitation.schemas.features import FeatureWindow


def _fit_ovr_model(x: list[list[float]], y: list[str], classes: list[str]) -> OVRLogisticModel:
    models: dict[str, BinaryLogisticRegression] = {}
    for cls_name in classes:
        y_bin = [1 if label == cls_name else 0 for label in y]
        clf = BinaryLogisticRegression(n_features=len(x[0]))
        clf.fit(x, y_bin)
        models[cls_name] = clf
    return OVRLogisticModel(classes=classes, models=models)


def _save_model(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_classical(
    input_path: str,
    output_dir: str,
    window_size: int,
    pause_speed_threshold: float,
    horizon_frames: int,
) -> dict[str, Any]:
    rows = load_rows(input_path)
    windows = build_windows(rows, window_size, pause_speed_threshold, horizon_frames)
    train, val = split_train_val(windows)

    x_train = [w["features"] for w in train]
    x_val = [w["features"] for w in val]

    scaler = StandardScaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_val_std = scaler.transform(x_val)

    classes = sorted({str(w["current_state"]) for w in train})
    y_state_train = [str(w["current_state"]) for w in train]
    y_state_val = [str(w["current_state"]) for w in val]
    state_model = _fit_ovr_model(x_train_std, y_state_train, classes=classes)

    y_fh_train = [int(w["future_hesitation"]) for w in train]
    y_fh_val = [int(w["future_hesitation"]) for w in val]
    fh_model = BinaryLogisticRegression(n_features=len(FEATURE_ORDER))
    fh_model.fit(x_train_std, y_fh_train)

    y_fc_train = [int(w["future_correction"]) for w in train]
    y_fc_val = [int(w["future_correction"]) for w in val]
    fc_model = BinaryLogisticRegression(n_features=len(FEATURE_ORDER))
    fc_model.fit(x_train_std, y_fc_train)

    state_pred = state_model.predict(x_val_std)
    state_model.predict_proba(x_val_std)
    fh_probs = fh_model.predict_proba(x_val_std)
    [1 if p >= 0.5 else 0 for p in fh_probs]
    fc_probs = fc_model.predict_proba(x_val_std)
    [1 if p >= 0.5 else 0 for p in fc_probs]

    rules_cfg = load_config("configs/baseline/rules_v1.yaml")
    rules_pred: list[str] = []
    for w in val:
        f = FeatureWindow(
            session_id=str(w["session_id"]),
            end_frame_idx=int(w["end_frame_idx"]),
            mean_speed=float(w["features"][0]),
            speed_variance=float(w["features"][1]),
            pause_ratio=float(w["features"][2]),
            direction_changes=int(w["features"][3]),
            progress_delta=float(w["features"][4]),
            backtrack_ratio=float(w["features"][5]),
            mean_workspace_distance=float(w["features"][6]),
        )
        rules_out = classify_window(
            f,
            thresholds=rules_cfg["thresholds"],
            risk_cfg=rules_cfg["risk"]
        )
        rules_pred.append(rules_out.current_state.value)

    metrics = {
        "current_state_classical": multiclass_metrics(y_state_val, state_pred, classes),
        "current_state_rules": multiclass_metrics(y_state_val, rules_pred, classes),
        "future_hesitation": binary_metrics(y_fh_val, fh_probs, threshold=0.5),
        "future_correction": binary_metrics(y_fc_val, fc_probs, threshold=0.5),
        "counts": {
            "train_windows": len(train),
            "val_windows": len(val),
        },
    }

    output = Path(output_dir)
    model_payload = {
        "feature_order": FEATURE_ORDER,
        "scaler": {"means": scaler.means, "stds": scaler.stds},
        "state": {
            "classes": classes,
            "weights": {name: clf.weights for name, clf in state_model.models.items()},
            "biases": {name: clf.bias for name, clf in state_model.models.items()},
        },
        "future_hesitation": {"weights": fh_model.weights, "bias": fh_model.bias},
        "future_correction": {"weights": fc_model.weights, "bias": fc_model.bias},
        "window_size": window_size,
        "pause_speed_threshold": pause_speed_threshold,
        "horizon_frames": horizon_frames,
    }
    _save_model(output / "classical_model.json", model_payload)
    _save_model(output / "metrics.json", metrics)
    return metrics


def _load_runtime_model(model_path: str | Path) -> dict[str, Any]:
    payload = _load_model(model_path)
    scaler = StandardScaler(means=payload["scaler"]["means"], stds=payload["scaler"]["stds"])

    state_models: dict[str, BinaryLogisticRegression] = {}
    for cls_name in payload["state"]["classes"]:
        clf = BinaryLogisticRegression(n_features=len(payload["feature_order"]))
        clf.weights = [float(v) for v in payload["state"]["weights"][cls_name]]
        clf.bias = float(payload["state"]["biases"][cls_name])
        state_models[cls_name] = clf
    state_model = OVRLogisticModel(classes=payload["state"]["classes"], models=state_models)

    fh = BinaryLogisticRegression(n_features=len(payload["feature_order"]))
    fh.weights = [float(v) for v in payload["future_hesitation"]["weights"]]
    fh.bias = float(payload["future_hesitation"]["bias"])

    fc = BinaryLogisticRegression(n_features=len(payload["feature_order"]))
    fc.weights = [float(v) for v in payload["future_correction"]["weights"]]
    fc.bias = float(payload["future_correction"]["bias"])

    return {
        "payload": payload,
        "scaler": scaler,
        "state": state_model,
        "future_hesitation": fh,
        "future_correction": fc,
    }


def load_classical_runtime(model_path: str | Path) -> dict[str, Any]:
    """Load a saved classical artifact for inference-time reuse."""
    return _load_runtime_model(model_path)


def predict_classical_window(
    runtime: dict[str, Any],
    features: list[float],
) -> dict[str, Any]:
    """Run the saved classical model on one feature window."""
    x_std = runtime["scaler"].transform([features])
    state_probabilities = runtime["state"].predict_proba(x_std)[0]
    predicted_state = runtime["state"].predict(x_std)[0]
    future_hesitation_probability = float(runtime["future_hesitation"].predict_proba(x_std)[0])
    future_correction_probability = float(runtime["future_correction"].predict_proba(x_std)[0])
    return {
        "predicted_state": predicted_state,
        "state_probabilities": state_probabilities,
        "future_hesitation_probability": future_hesitation_probability,
        "future_correction_probability": future_correction_probability,
        "thresholds": {
            "future_hesitation": 0.5,
            "future_correction": 0.5,
        },
    }


def evaluate_classical(input_path: str, model_path: str) -> dict[str, Any]:
    runtime = _load_runtime_model(model_path)
    payload = runtime["payload"]
    rows = load_rows(input_path)
    windows = build_windows(
        rows,
        window_size=int(payload["window_size"]),
        pause_speed_threshold=float(payload["pause_speed_threshold"]),
        horizon_frames=int(payload["horizon_frames"]),
    )
    _, val = split_train_val(windows)
    x_val = [w["features"] for w in val]
    x_std = runtime["scaler"].transform(x_val)

    y_state = [str(w["current_state"]) for w in val]
    y_state_pred = runtime["state"].predict(x_std)
    y_fh = [int(w["future_hesitation"]) for w in val]
    y_fh_probs = runtime["future_hesitation"].predict_proba(x_std)
    y_fc = [int(w["future_correction"]) for w in val]
    y_fc_probs = runtime["future_correction"].predict_proba(x_std)

    return {
        "current_state_classical": multiclass_metrics(
            y_state,
            y_state_pred,
            runtime["state"].classes
        ),
        "future_hesitation": binary_metrics(y_fh, y_fh_probs, threshold=0.5),
        "future_correction": binary_metrics(y_fc, y_fc_probs, threshold=0.5),
        "windows": len(val),
    }


def infer_sequence(input_path: str, model_path: str) -> list[dict[str, Any]]:
    runtime = _load_runtime_model(model_path)
    payload = runtime["payload"]
    rows = load_rows(input_path)
    windows = build_windows(
        rows,
        window_size=int(payload["window_size"]),
        pause_speed_threshold=float(payload["pause_speed_threshold"]),
        horizon_frames=int(payload["horizon_frames"]),
    )
    x = runtime["scaler"].transform([w["features"] for w in windows])
    state_probs = runtime["state"].predict_proba(x)
    state_pred = runtime["state"].predict(x)
    out: list[dict[str, Any]] = []
    for w, p, s in zip(windows, state_probs, state_pred, strict=False):
        out.append(
            {
                "session_id": w["session_id"],
                "end_frame_idx": w["end_frame_idx"],
                "predicted_state": s,
                "state_probabilities": p,
            }
        )
    return out


def predict_future_risk(input_path: str, model_path: str) -> list[dict[str, Any]]:
    runtime = _load_runtime_model(model_path)
    payload = runtime["payload"]
    rows = load_rows(input_path)
    windows = build_windows(
        rows,
        window_size=int(payload["window_size"]),
        pause_speed_threshold=float(payload["pause_speed_threshold"]),
        horizon_frames=int(payload["horizon_frames"]),
    )
    x = runtime["scaler"].transform([w["features"] for w in windows])
    fh = runtime["future_hesitation"].predict_proba(x)
    fc = runtime["future_correction"].predict_proba(x)
    out: list[dict[str, Any]] = []
    for w, p_h, p_c in zip(windows, fh, fc, strict=False):
        out.append(
            {
                "session_id": w["session_id"],
                "end_frame_idx": w["end_frame_idx"],
                "future_hesitation_within_horizon": p_h,
                "future_correction_within_horizon": p_c,
            }
        )
    return out
