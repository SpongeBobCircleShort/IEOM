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


def _project_feature_rows(
    feature_rows: list[list[float]],
    feature_indices: list[int] | None,
) -> list[list[float]]:
    if feature_indices is None:
        return feature_rows
    return [[row[index] for index in feature_indices] for row in feature_rows]


def _label_space(*label_sets: list[str]) -> list[str]:
    labels: set[str] = set()
    for values in label_sets:
        labels.update(values)
    return sorted(labels)


def feature_window_from_window(window: dict[str, object]) -> FeatureWindow:
    """Rebuild a rule-engine feature window from a vectorized classical example."""
    values = [float(value) for value in window["features"]]
    return FeatureWindow(
        session_id=str(window["session_id"]),
        end_frame_idx=int(window["end_frame_idx"]),
        mean_speed=values[0],
        speed_variance=values[1],
        pause_ratio=values[2],
        direction_changes=int(values[3]),
        progress_delta=values[4],
        backtrack_ratio=values[5],
        mean_workspace_distance=values[6],
    )


def predict_rules_windows(
    windows: list[dict[str, object]],
) -> list[dict[str, Any]]:
    """Run the rules baseline over precomputed feature windows."""
    rules_cfg = load_config("configs/baseline/rules_v1.yaml")
    outputs: list[dict[str, Any]] = []
    for window in windows:
        rules_out = classify_window(
            feature_window_from_window(window),
            thresholds=rules_cfg["thresholds"],
            risk_cfg=rules_cfg["risk"],
        )
        outputs.append(
            {
                "session_id": str(window["session_id"]),
                "end_frame_idx": int(window["end_frame_idx"]),
                "predicted_state": rules_out.current_state.value,
                "triggered_rules": list(rules_out.triggered_rules),
                "hesitation_risk": float(rules_out.hesitation_risk),
                "correction_rework_risk": float(rules_out.correction_rework_risk),
                "overlap_risk": float(rules_out.overlap_risk),
            }
        )
    return outputs


def train_classical_on_windows(
    train_windows: list[dict[str, object]],
    eval_windows: list[dict[str, object]],
    output_dir: str,
    window_size: int,
    pause_speed_threshold: float,
    horizon_frames: int,
    feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Train and evaluate the classical baseline on explicit window splits."""
    x_train_full = [[float(value) for value in window["features"]] for window in train_windows]
    x_eval_full = [[float(value) for value in window["features"]] for window in eval_windows]
    x_train = _project_feature_rows(x_train_full, feature_indices)
    x_eval = _project_feature_rows(x_eval_full, feature_indices)

    scaler = StandardScaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_eval_std = scaler.transform(x_eval)

    classes = _label_space(
        [str(window["current_state"]) for window in train_windows],
        [str(window["current_state"]) for window in eval_windows],
    )
    y_state_train = [str(window["current_state"]) for window in train_windows]
    y_state_eval = [str(window["current_state"]) for window in eval_windows]
    state_model = _fit_ovr_model(x_train_std, y_state_train, classes=classes)

    y_fh_train = [int(window["future_hesitation"]) for window in train_windows]
    y_fh_eval = [int(window["future_hesitation"]) for window in eval_windows]
    fh_model = BinaryLogisticRegression(n_features=len(x_train_std[0]))
    fh_model.fit(x_train_std, y_fh_train)

    y_fc_train = [int(window["future_correction"]) for window in train_windows]
    y_fc_eval = [int(window["future_correction"]) for window in eval_windows]
    fc_model = BinaryLogisticRegression(n_features=len(x_train_std[0]))
    fc_model.fit(x_train_std, y_fc_train)

    state_pred = state_model.predict(x_eval_std)
    fh_probs = fh_model.predict_proba(x_eval_std)
    fc_probs = fc_model.predict_proba(x_eval_std)
    rules_outputs = predict_rules_windows(eval_windows)
    rules_pred = [str(record["predicted_state"]) for record in rules_outputs]

    metrics = {
        "current_state_classical": multiclass_metrics(y_state_eval, state_pred, classes),
        "current_state_rules": multiclass_metrics(y_state_eval, rules_pred, classes),
        "future_hesitation": binary_metrics(y_fh_eval, fh_probs, threshold=0.5),
        "future_correction": binary_metrics(y_fc_eval, fc_probs, threshold=0.5),
        "counts": {
            "train_windows": len(train_windows),
            "eval_windows": len(eval_windows),
        },
    }

    selected_feature_order = [
        FEATURE_ORDER[index]
        for index in (feature_indices if feature_indices is not None else list(range(len(FEATURE_ORDER))))
    ]
    output = Path(output_dir)
    model_payload = {
        "feature_order": FEATURE_ORDER,
        "selected_feature_order": selected_feature_order,
        "feature_indices": feature_indices,
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


def train_classical(
    input_path: str,
    output_dir: str,
    window_size: int,
    pause_speed_threshold: float,
    horizon_frames: int,
) -> dict[str, Any]:
    """Train the classical baseline using the default session holdout split."""
    rows = load_rows(input_path)
    windows = build_windows(rows, window_size, pause_speed_threshold, horizon_frames)
    train, val = split_train_val(windows)
    return train_classical_on_windows(
        train_windows=train,
        eval_windows=val,
        output_dir=output_dir,
        window_size=window_size,
        pause_speed_threshold=pause_speed_threshold,
        horizon_frames=horizon_frames,
    )


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
    payload = runtime["payload"]
    feature_indices = payload.get("feature_indices")
    model_features = features
    if feature_indices is not None and len(features) != len(payload.get("selected_feature_order", [])):
        model_features = [features[index] for index in feature_indices]
    x_std = runtime["scaler"].transform([model_features])
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


def evaluate_classical_windows(
    runtime: dict[str, Any],
    windows: list[dict[str, object]],
) -> dict[str, Any]:
    """Evaluate a saved classical model against an explicit evaluation split."""
    y_state = [str(window["current_state"]) for window in windows]
    state_predictions: list[str] = []
    y_fh = [int(window["future_hesitation"]) for window in windows]
    y_fh_probs: list[float] = []
    y_fc = [int(window["future_correction"]) for window in windows]
    y_fc_probs: list[float] = []
    for window in windows:
        prediction = predict_classical_window(runtime, [float(value) for value in window["features"]])
        state_predictions.append(str(prediction["predicted_state"]))
        y_fh_probs.append(float(prediction["future_hesitation_probability"]))
        y_fc_probs.append(float(prediction["future_correction_probability"]))

    classes = _label_space(y_state, list(runtime["state"].classes), state_predictions)
    return {
        "current_state_classical": multiclass_metrics(y_state, state_predictions, classes),
        "future_hesitation": binary_metrics(y_fh, y_fh_probs, threshold=0.5),
        "future_correction": binary_metrics(y_fc, y_fc_probs, threshold=0.5),
        "windows": len(windows),
    }


def evaluate_classical(input_path: str, model_path: str) -> dict[str, Any]:
    """Evaluate the classical model using the default session holdout split."""
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
    return evaluate_classical_windows(runtime, val)


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
