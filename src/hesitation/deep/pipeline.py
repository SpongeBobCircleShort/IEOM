"""Phase 3 deep temporal training, inference, and comparison pipelines."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from hesitation.baselines.rules_engine import classify_window
from hesitation.deep.dataset import SequenceWindow, build_sequence_windows, load_rows, split_train_val
from hesitation.deep.model import FallbackDeepModel, TorchGRUMultiHead, torch
from hesitation.deep.serialize import load_json, save_json
from hesitation.evaluation.metrics import binary_metrics, multiclass_metrics
from hesitation.evaluation.reporting import write_comparison_report
from hesitation.features.pipeline import window_to_features
from hesitation.io.config import load_config
from hesitation.ml.logistic import BinaryLogisticRegression, OVRLogisticModel, StandardScaler
from hesitation.ml.pipeline import evaluate_classical
from hesitation.schemas.events import FrameObservation
from hesitation.schemas.features import FeatureWindow


def _flatten_sequence(seq: list[list[float]]) -> list[float]:
    return [value for row in seq for value in row]


def _fit_ovr(features: list[list[float]], labels: list[str], classes: list[str]) -> OVRLogisticModel:
    models: dict[str, BinaryLogisticRegression] = {}
    for cls_name in classes:
        y = [1 if label == cls_name else 0 for label in labels]
        clf = BinaryLogisticRegression(n_features=len(features[0]))
        clf.fit(features, y, epochs=100)
        models[cls_name] = clf
    return OVRLogisticModel(classes=classes, models=models)


def _build_feature_window_from_sequence(window: SequenceWindow) -> FeatureWindow:
    frames = [
        FrameObservation(
            session_id=window["session_id"],
            frame_idx=i,
            timestamp_ms=i * 100,
            task_step_id=0,
            hand_x=row[0],
            hand_y=row[1],
            hand_speed=row[2],
            hand_accel=row[3],
            distance_to_robot_workspace=row[4],
            progress=row[5],
            confidence=row[6],
            is_dropout=bool(row[7]),
        )
        for i, row in enumerate(window["sequence"])
    ]
    return window_to_features(frames, pause_speed_threshold=0.03)


def _evaluate_rules(windows: list[SequenceWindow]) -> list[str]:
    rules_cfg = load_config("configs/baseline/rules_v1.yaml")
    preds: list[str] = []
    for w in windows:
        f = _build_feature_window_from_sequence(w)
        out = classify_window(f, thresholds=rules_cfg["thresholds"], risk_cfg=rules_cfg["risk"])
        preds.append(out.current_state.value)
    return preds


def _train_fallback(
    train: list[SequenceWindow],
    val: list[SequenceWindow],
    output_dir: str,
    window_size: int,
    horizon_frames: int,
) -> dict[str, Any]:
    x_train = [_flatten_sequence(w["sequence"]) for w in train]
    x_val = [_flatten_sequence(w["sequence"]) for w in val]

    scaler = StandardScaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_val_std = scaler.transform(x_val)

    classes = sorted({w["current_state"] for w in train})
    y_state_train = [w["current_state"] for w in train]
    y_state_val = [w["current_state"] for w in val]
    state_model = _fit_ovr(x_train_std, y_state_train, classes)

    y_fh_train = [w["future_hesitation"] for w in train]
    y_fh_val = [w["future_hesitation"] for w in val]
    y_fc_train = [w["future_correction"] for w in train]
    y_fc_val = [w["future_correction"] for w in val]

    fh_model = BinaryLogisticRegression(n_features=len(x_train_std[0]))
    fh_model.fit(x_train_std, y_fh_train, epochs=100)
    fc_model = BinaryLogisticRegression(n_features=len(x_train_std[0]))
    fc_model.fit(x_train_std, y_fc_train, epochs=100)

    state_pred = state_model.predict(x_val_std)
    fh_probs = fh_model.predict_proba(x_val_std)
    fc_probs = fc_model.predict_proba(x_val_std)

    rules_pred = _evaluate_rules(val)

    metrics = {
        "backend": "fallback",
        "current_state_deep": multiclass_metrics(y_state_val, state_pred, classes),
        "current_state_rules": multiclass_metrics(y_state_val, rules_pred, classes),
        "future_hesitation": binary_metrics(y_fh_val, fh_probs, threshold=0.5),
        "future_correction": binary_metrics(y_fc_val, fc_probs, threshold=0.5),
        "counts": {"train_windows": len(train), "val_windows": len(val)},
    }

    payload = {
        "backend": "fallback",
        "window_size": window_size,
        "horizon_frames": horizon_frames,
        "scaler": {"means": scaler.means, "stds": scaler.stds},
        "classes": classes,
        "state": {
            "weights": {name: model.weights for name, model in state_model.models.items()},
            "biases": {name: model.bias for name, model in state_model.models.items()},
        },
        "future_hesitation": {"weights": fh_model.weights, "bias": fh_model.bias},
        "future_correction": {"weights": fc_model.weights, "bias": fc_model.bias},
    }

    target = Path(output_dir)
    save_json(target / "deep_model.json", payload)
    save_json(target / "deep_metrics.json", metrics)
    return metrics


def _torch_train_stub(
    train: list[SequenceWindow],
    val: list[SequenceWindow],
    output_dir: str,
    window_size: int,
    horizon_frames: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    assert torch is not None
    random.seed(seed)
    torch.manual_seed(seed)

    classes = sorted({w["current_state"] for w in train})
    class_to_idx = {name: i for i, name in enumerate(classes)}

    def to_tensor(split: list[SequenceWindow]) -> tuple[Any, Any, Any, Any]:
        x = torch.tensor([w["sequence"] for w in split], dtype=torch.float32)
        y_state = torch.tensor([class_to_idx[w["current_state"]] for w in split], dtype=torch.long)
        y_fh = torch.tensor([w["future_hesitation"] for w in split], dtype=torch.float32).unsqueeze(1)
        y_fc = torch.tensor([w["future_correction"] for w in split], dtype=torch.float32).unsqueeze(1)
        return x, y_state, y_fh, y_fc

    x_train, ys_train, yh_train, yc_train = to_tensor(train)
    x_val, ys_val, yh_val, yc_val = to_tensor(val)

    model = TorchGRUMultiHead(input_dim=x_train.shape[2], hidden_dim=hidden_dim, n_state_classes=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        state_logits, fh_logits, fc_logits = model(x_train)
        loss = ce(state_logits, ys_train) + bce(fh_logits, yh_train) + bce(fc_logits, yc_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        state_logits, fh_logits, fc_logits = model(x_val)
        state_probs = torch.softmax(state_logits, dim=1)
        state_pred_idx = torch.argmax(state_probs, dim=1).tolist()
        fh_probs = torch.sigmoid(fh_logits).squeeze(1).tolist()
        fc_probs = torch.sigmoid(fc_logits).squeeze(1).tolist()

    y_state_val = [w["current_state"] for w in val]
    state_pred = [classes[i] for i in state_pred_idx]
    y_fh_val = [w["future_hesitation"] for w in val]
    y_fc_val = [w["future_correction"] for w in val]

    rules_pred = _evaluate_rules(val)

    metrics = {
        "backend": "torch",
        "current_state_deep": multiclass_metrics(y_state_val, state_pred, classes),
        "current_state_rules": multiclass_metrics(y_state_val, rules_pred, classes),
        "future_hesitation": binary_metrics(y_fh_val, [float(p) for p in fh_probs], threshold=0.5),
        "future_correction": binary_metrics(y_fc_val, [float(p) for p in fc_probs], threshold=0.5),
        "counts": {"train_windows": len(train), "val_windows": len(val)},
    }

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "window_size": window_size,
            "horizon_frames": horizon_frames,
            "input_dim": int(x_train.shape[2]),
            "hidden_dim": hidden_dim,
            "backend": "torch",
        },
        target / "deep_model.pt",
    )
    save_json(target / "deep_metrics.json", metrics)
    return metrics


def train_deep(
    input_path: str,
    output_dir: str,
    window_size: int,
    horizon_frames: int,
    epochs: int = 20,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train deep temporal baseline (PyTorch GRU default, fallback backend when torch missing)."""
    rows = load_rows(input_path)
    windows = build_sequence_windows(rows, window_size=window_size, horizon_frames=horizon_frames)
    train, val = split_train_val(windows)

    if not train or not val:
        raise ValueError("Insufficient windows for train/validation split")

    if torch is None:
        return _train_fallback(train, val, output_dir, window_size, horizon_frames)

    return _torch_train_stub(
        train=train,
        val=val,
        output_dir=output_dir,
        window_size=window_size,
        horizon_frames=horizon_frames,
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )


def _load_fallback_runtime(model_path: str | Path) -> tuple[dict[str, Any], StandardScaler, FallbackDeepModel]:
    payload = load_json(model_path)
    scaler = StandardScaler(means=payload["scaler"]["means"], stds=payload["scaler"]["stds"])
    classes = payload["classes"]
    state_models: dict[str, BinaryLogisticRegression] = {}
    n_features = len(payload["state"]["weights"][classes[0]])
    for cls_name in classes:
        clf = BinaryLogisticRegression(n_features=n_features)
        clf.weights = [float(v) for v in payload["state"]["weights"][cls_name]]
        clf.bias = float(payload["state"]["biases"][cls_name])
        state_models[cls_name] = clf

    state = OVRLogisticModel(classes=classes, models=state_models)
    fh = BinaryLogisticRegression(n_features=n_features)
    fh.weights = [float(v) for v in payload["future_hesitation"]["weights"]]
    fh.bias = float(payload["future_hesitation"]["bias"])

    fc = BinaryLogisticRegression(n_features=n_features)
    fc.weights = [float(v) for v in payload["future_correction"]["weights"]]
    fc.bias = float(payload["future_correction"]["bias"])

    return payload, scaler, FallbackDeepModel(classes=classes, state_model=state, future_hes_model=fh, future_corr_model=fc)


def infer_sequence_deep(input_path: str, model_path: str) -> list[dict[str, Any]]:
    """Run deep current-state and near-future risk inference."""
    if str(model_path).endswith(".pt") and torch is not None:
        ckpt = torch.load(model_path, map_location="cpu")
        classes = ckpt["classes"]
        model = TorchGRUMultiHead(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"], n_state_classes=len(classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        rows = load_rows(input_path)
        windows = build_sequence_windows(rows, window_size=int(ckpt["window_size"]), horizon_frames=int(ckpt["horizon_frames"]))
        x = torch.tensor([w["sequence"] for w in windows], dtype=torch.float32)
        with torch.no_grad():
            s_logit, h_logit, c_logit = model(x)
            s_prob = torch.softmax(s_logit, dim=1).tolist()
            h_prob = torch.sigmoid(h_logit).squeeze(1).tolist()
            c_prob = torch.sigmoid(c_logit).squeeze(1).tolist()
        records: list[dict[str, Any]] = []
        for w, sp, hp, cp in zip(windows, s_prob, h_prob, c_prob):
            probs = {classes[i]: float(sp[i]) for i in range(len(classes))}
            records.append(
                {
                    "session_id": w["session_id"],
                    "end_frame_idx": w["end_frame_idx"],
                    "predicted_state": max(probs.items(), key=lambda kv: kv[1])[0],
                    "state_probabilities": probs,
                    "future_hesitation_within_horizon": float(hp),
                    "future_correction_within_horizon": float(cp),
                }
            )
        return records

    payload, scaler, model = _load_fallback_runtime(model_path)
    rows = load_rows(input_path)
    windows = build_sequence_windows(rows, window_size=int(payload["window_size"]), horizon_frames=int(payload["horizon_frames"]))
    x = scaler.transform([_flatten_sequence(w["sequence"]) for w in windows])
    s_probs = model.predict_state_proba(x)
    s_pred = model.predict_state(x)
    h_probs, c_probs = model.predict_future(x)

    records: list[dict[str, Any]] = []
    for w, sp, ss, hp, cp in zip(windows, s_probs, s_pred, h_probs, c_probs):
        records.append(
            {
                "session_id": w["session_id"],
                "end_frame_idx": w["end_frame_idx"],
                "predicted_state": ss,
                "state_probabilities": sp,
                "future_hesitation_within_horizon": hp,
                "future_correction_within_horizon": cp,
            }
        )
    return records


def evaluate_deep(input_path: str, model_path: str) -> dict[str, Any]:
    """Evaluate deep model on validation split protocol."""
    rows = load_rows(input_path)
    if str(model_path).endswith(".pt") and torch is not None:
        ckpt = torch.load(model_path, map_location="cpu")
        windows = build_sequence_windows(rows, window_size=int(ckpt["window_size"]), horizon_frames=int(ckpt["horizon_frames"]))
        _, val = split_train_val(windows)
        all_preds = infer_sequence_deep(input_path, model_path)
        pred_lookup = {(r["session_id"], r["end_frame_idx"]): r for r in all_preds}
    else:
        payload = load_json(model_path)
        windows = build_sequence_windows(rows, window_size=int(payload["window_size"]), horizon_frames=int(payload["horizon_frames"]))
        _, val = split_train_val(windows)
        all_preds = infer_sequence_deep(input_path, model_path)
        pred_lookup = {(r["session_id"], r["end_frame_idx"]): r for r in all_preds}

    y_state: list[str] = []
    y_state_pred: list[str] = []
    y_h: list[int] = []
    y_c: list[int] = []
    p_h: list[float] = []
    p_c: list[float] = []

    for w in val:
        pred = pred_lookup[(w["session_id"], w["end_frame_idx"])]
        y_state.append(w["current_state"])
        y_state_pred.append(str(pred["predicted_state"]))
        y_h.append(int(w["future_hesitation"]))
        y_c.append(int(w["future_correction"]))
        p_h.append(float(pred["future_hesitation_within_horizon"]))
        p_c.append(float(pred["future_correction_within_horizon"]))

    classes = sorted({w["current_state"] for w in windows})
    return {
        "current_state_deep": multiclass_metrics(y_state, y_state_pred, classes),
        "future_hesitation": binary_metrics(y_h, p_h, threshold=0.5),
        "future_correction": binary_metrics(y_c, p_c, threshold=0.5),
        "windows": len(val),
    }


def compare_models(input_path: str, classical_model_path: str, deep_model_path: str, output_dir: str) -> dict[str, Any]:
    """Compare rules, classical, and deep models on aligned split protocol and emit report files."""
    classical = evaluate_classical(input_path, classical_model_path)
    deep = evaluate_deep(input_path, deep_model_path)

    # Compute rules baseline on the deep split configuration for explicit 3-way comparison.
    if str(deep_model_path).endswith(".pt") and torch is not None:
        ckpt = torch.load(deep_model_path, map_location="cpu")
        window_size = int(ckpt["window_size"])
        horizon_frames = int(ckpt["horizon_frames"])
    else:
        payload = load_json(deep_model_path)
        window_size = int(payload["window_size"])
        horizon_frames = int(payload["horizon_frames"])

    rows = load_rows(input_path)
    windows = build_sequence_windows(rows, window_size=window_size, horizon_frames=horizon_frames)
    _, val = split_train_val(windows)
    y_state = [w["current_state"] for w in val]
    rules_pred = _evaluate_rules(val)
    classes = sorted({w["current_state"] for w in windows})
    rules_metrics = multiclass_metrics(y_state, rules_pred, classes)

    summary = {
        "current_state_accuracy": {
            "rules": rules_metrics["accuracy"],
            "classical": classical["current_state_classical"]["accuracy"],
            "deep": deep["current_state_deep"]["accuracy"],
        },
        "current_state_macro_f1": {
            "rules": rules_metrics["macro_f1"],
            "classical": classical["current_state_classical"]["macro_f1"],
            "deep": deep["current_state_deep"]["macro_f1"],
        },
        "future_hesitation_auprc": {
            "classical": classical["future_hesitation"]["auprc"],
            "deep": deep["future_hesitation"]["auprc"],
        },
        "future_correction_auprc": {
            "classical": classical["future_correction"]["auprc"],
            "deep": deep["future_correction"]["auprc"],
        },
    }
    report = {
        "rules": {"current_state_rules": rules_metrics},
        "classical": classical,
        "deep": deep,
        "summary": summary,
    }
    write_comparison_report(output_dir, report)
    return report
