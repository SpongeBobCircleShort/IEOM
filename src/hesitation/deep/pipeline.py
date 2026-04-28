"""Phase 3.5 deep temporal training, calibration, and comparison utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from hesitation.baselines.rules_engine import classify_window
from hesitation.deep.dataset import (
    FRAME_FEATURE_ORDER,
    SequenceWindow,
    build_sequence_windows,
    load_rows,
    split_train_val,
)
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


@dataclass(slots=True)
class ThresholdConfig:
    """Decision thresholds for future-risk heads."""

    future_hesitation: float = 0.5
    future_correction: float = 0.5


@dataclass(slots=True)
class DeepTrainConfig:
    """Deep training configuration."""

    window_size: int = 20
    horizon_frames: int = 20
    epochs: int = 20
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    batch_size: int = 64


def _flatten_sequence(seq: list[list[float]]) -> list[float]:
    return [value for row in seq for value in row]


def _project_sequence(
    sequence: list[list[float]],
    frame_feature_indices: list[int] | None,
) -> list[list[float]]:
    if frame_feature_indices is None:
        return sequence
    return [[row[index] for index in frame_feature_indices] for row in sequence]


def _project_windows(
    windows: list[SequenceWindow],
    frame_feature_indices: list[int] | None,
) -> list[SequenceWindow]:
    if frame_feature_indices is None:
        return windows
    return [
        SequenceWindow(
            session_id=window["session_id"],
            dataset_name=window["dataset_name"],
            end_frame_idx=window["end_frame_idx"],
            sequence=_project_sequence(window["sequence"], frame_feature_indices),
            current_state=window["current_state"],
            future_hesitation=window["future_hesitation"],
            future_correction=window["future_correction"],
        )
        for window in windows
    ]


def _label_space(*label_sets: list[str]) -> list[str]:
    labels: set[str] = set()
    for values in label_sets:
        labels.update(values)
    return sorted(labels)


def _fit_ovr(
    features: list[list[float]],
    labels: list[str],
    classes: list[str]
) -> OVRLogisticModel:
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


def _compute_pos_weight(targets: list[int]) -> float:
    pos = sum(targets)
    neg = max(0, len(targets) - pos)
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def _train_fallback(
    train: list[SequenceWindow],
    val: list[SequenceWindow],
    output_dir: str,
    cfg: DeepTrainConfig,
    frame_feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    train_projected = _project_windows(train, frame_feature_indices)
    val_projected = _project_windows(val, frame_feature_indices)
    x_train = [_flatten_sequence(w["sequence"]) for w in train_projected]
    x_val = [_flatten_sequence(w["sequence"]) for w in val_projected]

    scaler = StandardScaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_val_std = scaler.transform(x_val)

    classes = _label_space(
        [w["current_state"] for w in train],
        [w["current_state"] for w in val],
    )
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
        "imbalance": {
            "future_hes_pos_weight": _compute_pos_weight(y_fh_train),
            "future_corr_pos_weight": _compute_pos_weight(y_fc_train),
        },
    }

    payload = {
        "backend": "fallback",
        "window_size": cfg.window_size,
        "horizon_frames": cfg.horizon_frames,
        "frame_feature_order": [
            name
            for index, name in enumerate(FRAME_FEATURE_ORDER)
            if frame_feature_indices is None or index in frame_feature_indices
        ],
        "frame_feature_indices": frame_feature_indices,
        "scaler": {"means": scaler.means, "stds": scaler.stds},
        "classes": classes,
        "state": {
            "weights": {name: model.weights for name, model in state_model.models.items()},
            "biases": {name: model.bias for name, model in state_model.models.items()},
        },
        "future_hesitation": {"weights": fh_model.weights, "bias": fh_model.bias},
        "future_correction": {"weights": fc_model.weights, "bias": fc_model.bias},
        "train_config": {
                "window_size": cfg.window_size,
                "horizon_frames": cfg.horizon_frames,
                "epochs": cfg.epochs,
                "hidden_dim": cfg.hidden_dim,
                "learning_rate": cfg.learning_rate,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
            },
    }

    target = Path(output_dir)
    save_json(target / "deep_model.json", payload)
    save_json(target / "deep_metrics.json", metrics)
    return metrics


def _iter_minibatches(n_samples: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    indices = list(range(n_samples))
    rng.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, n_samples, batch_size)]


def _torch_train(
    train: list[SequenceWindow],
    val: list[SequenceWindow],
    output_dir: str,
    cfg: DeepTrainConfig,
    frame_feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    assert torch is not None
    rng = random.Random(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_projected = _project_windows(train, frame_feature_indices)
    val_projected = _project_windows(val, frame_feature_indices)
    classes = _label_space(
        [w["current_state"] for w in train],
        [w["current_state"] for w in val],
    )
    class_to_idx = {name: i for i, name in enumerate(classes)}

    def to_tensor(split: list[SequenceWindow]) -> tuple[Any, Any, Any, Any]:
        x = torch.tensor([w["sequence"] for w in split], dtype=torch.float32)
        y_state = torch.tensor([class_to_idx[w["current_state"]] for w in split], dtype=torch.long)
        y_fh = torch.tensor([w["future_hesitation"] for w in split], dtype=torch.float32).unsqueeze(1)
        y_fc = torch.tensor([w["future_correction"] for w in split], dtype=torch.float32).unsqueeze(1)
        return x, y_state, y_fh, y_fc

    x_train, ys_train, yh_train, yc_train = to_tensor(train_projected)
    x_val, ys_val, yh_val, yc_val = to_tensor(val_projected)

    state_counts = [sum(1 for w in train if w["current_state"] == cls_name) for cls_name in classes]
    state_weights = [1.0 / math.sqrt(max(1, c)) for c in state_counts]
    state_weights_tensor = torch.tensor(state_weights, dtype=torch.float32)

    pos_w_h = torch.tensor([_compute_pos_weight([w["future_hesitation"] for w in train])], dtype=torch.float32)
    pos_w_c = torch.tensor([_compute_pos_weight([w["future_correction"] for w in train])], dtype=torch.float32)

    model = TorchGRUMultiHead(
        input_dim=x_train.shape[2],
        hidden_dim=cfg.hidden_dim,
        n_state_classes=len(classes)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ce = torch.nn.CrossEntropyLoss(weight=state_weights_tensor)
    bce_h = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_h)
    bce_c = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_c)

    model.train()
    history: list[float] = []
    for _ in range(cfg.epochs):
        batch_losses: list[float] = []
        for batch_idx in _iter_minibatches(len(train), cfg.batch_size, rng):
            xb = x_train[batch_idx]
            ysb = ys_train[batch_idx]
            yhb = yh_train[batch_idx]
            ycb = yc_train[batch_idx]

            optimizer.zero_grad()
            state_logits, fh_logits, fc_logits = model(xb)
            loss_state = ce(state_logits, ysb)
            loss_h = bce_h(fh_logits, yhb)
            loss_c = bce_c(fc_logits, ycb)
            loss = loss_state + loss_h + loss_c
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        history.append(fmean(batch_losses) if batch_losses else 0.0)

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
        "imbalance": {
            "state_class_weights": state_weights,
            "future_hes_pos_weight": float(pos_w_h.item()),
            "future_corr_pos_weight": float(pos_w_c.item()),
        },
        "loss_history": history,
    }

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "window_size": cfg.window_size,
            "horizon_frames": cfg.horizon_frames,
            "input_dim": int(x_train.shape[2]),
            "hidden_dim": cfg.hidden_dim,
            "frame_feature_indices": frame_feature_indices,
            "backend": "torch",
            "train_config": {
                "window_size": cfg.window_size,
                "horizon_frames": cfg.horizon_frames,
                "epochs": cfg.epochs,
                "hidden_dim": cfg.hidden_dim,
                "learning_rate": cfg.learning_rate,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
            },
        },
        target / "deep_model.pt",
    )
    save_json(target / "deep_metrics.json", metrics)
    return metrics


def train_deep_on_windows(
    train_windows: list[SequenceWindow],
    eval_windows: list[SequenceWindow],
    output_dir: str,
    cfg: DeepTrainConfig,
    frame_feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Train the deep baseline on explicit train/eval splits."""
    if not train_windows or not eval_windows:
        raise ValueError("Insufficient windows for train/eval split")
    if torch is None:
        return _train_fallback(train_windows, eval_windows, output_dir, cfg, frame_feature_indices)
    return _torch_train(train_windows, eval_windows, output_dir, cfg, frame_feature_indices)


def train_deep(
    input_path: str,
    output_dir: str,
    window_size: int,
    horizon_frames: int,
    epochs: int = 20,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Train deep temporal baseline. PyTorch GRU is primary backend when available."""
    cfg = DeepTrainConfig(
        window_size=window_size,
        horizon_frames=horizon_frames,
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
        batch_size=batch_size,
    )
    rows = load_rows(input_path)
    windows = build_sequence_windows(
        rows,
        window_size=cfg.window_size,
        horizon_frames=cfg.horizon_frames
    )
    train, val = split_train_val(windows)
    return train_deep_on_windows(train, val, output_dir, cfg)


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

    return payload, scaler, FallbackDeepModel(
        classes=classes,
        state_model=state,
        future_hes_model=fh,
        future_corr_model=fc
    )


def _load_deep_windows_for_model(input_path: str, model_path: str | Path) -> list[SequenceWindow]:
    rows = load_rows(input_path)
    if str(model_path).endswith(".pt") and torch is not None:
        ckpt = torch.load(model_path, map_location="cpu")
        return build_sequence_windows(
            rows,
            window_size=int(ckpt["window_size"]),
            horizon_frames=int(ckpt["horizon_frames"])
        )
    payload = load_json(model_path)
    return build_sequence_windows(
        rows,
        window_size=int(payload["window_size"]),
        horizon_frames=int(payload["horizon_frames"])
    )


def infer_sequence_deep(input_path: str, model_path: str) -> list[dict[str, Any]]:
    """Run deep current-state and near-future risk inference with probabilities."""
    windows = _load_deep_windows_for_model(input_path, model_path)
    return infer_sequence_deep_windows(windows, model_path)


def infer_sequence_deep_windows(
    windows: list[SequenceWindow],
    model_path: str | Path,
) -> list[dict[str, Any]]:
    """Run deep inference over an explicit evaluation split."""
    if str(model_path).endswith(".pt") and torch is not None:
        ckpt = torch.load(model_path, map_location="cpu")
        classes = ckpt["classes"]
        frame_feature_indices = ckpt.get("frame_feature_indices")
        model = TorchGRUMultiHead(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            n_state_classes=len(classes)
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        projected = _project_windows(windows, frame_feature_indices)
        x = torch.tensor([w["sequence"] for w in projected], dtype=torch.float32)
        with torch.no_grad():
            s_logit, h_logit, c_logit = model(x)
            s_prob = torch.softmax(s_logit, dim=1).tolist()
            h_prob = torch.sigmoid(h_logit).squeeze(1).tolist()
            c_prob = torch.sigmoid(c_logit).squeeze(1).tolist()
        records: list[dict[str, Any]] = []
        for w, sp, hp, cp in zip(windows, s_prob, h_prob, c_prob, strict=False):
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
    projected = _project_windows(windows, payload.get("frame_feature_indices"))
    x = scaler.transform([_flatten_sequence(w["sequence"]) for w in projected])
    s_probs = model.predict_state_proba(x)
    s_pred = model.predict_state(x)
    h_probs, c_probs = model.predict_future(x)

    records: list[dict[str, Any]] = []
    for w, sp, ss, hp, cp in zip(windows, s_probs, s_pred, h_probs, c_probs, strict=False):
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


def tune_thresholds(input_path: str, model_path: str, output_path: str) -> dict[str, float]:
    """Tune binary decision thresholds on validation split by maximizing F1."""
    windows = _load_deep_windows_for_model(input_path, model_path)
    _, val = split_train_val(windows)
    preds = infer_sequence_deep(input_path, model_path)
    lookup = {(r["session_id"], r["end_frame_idx"]): r for r in preds}

    y_h: list[int] = []
    y_c: list[int] = []
    p_h: list[float] = []
    p_c: list[float] = []
    for w in val:
        rec = lookup[(w["session_id"], w["end_frame_idx"])]
        y_h.append(int(w["future_hesitation"]))
        y_c.append(int(w["future_correction"]))
        p_h.append(float(rec["future_hesitation_within_horizon"]))
        p_c.append(float(rec["future_correction_within_horizon"]))

    def best_threshold(y_true: list[int], y_prob: list[float]) -> float:
        best_t = 0.5
        best_f1 = -1.0
        for i in range(1, 20):
            t = i / 20.0
            f1 = float(binary_metrics(y_true, y_prob, threshold=t)["f1"])
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    tuned = {
        "future_hesitation": best_threshold(y_h, p_h),
        "future_correction": best_threshold(y_c, p_c),
    }
    save_json(output_path, tuned)
    return tuned


def evaluate_deep_calibrated(
    input_path: str,
    model_path: str,
    threshold_path: str
) -> dict[str, Any]:
    """Evaluate deep model using externally tuned thresholds."""
    thresholds = load_json(threshold_path)
    t_h = float(thresholds.get("future_hesitation", 0.5))
    t_c = float(thresholds.get("future_correction", 0.5))

    windows = _load_deep_windows_for_model(input_path, model_path)
    _, val = split_train_val(windows)
    all_preds = infer_sequence_deep_windows(val, model_path)
    lookup = {(r["session_id"], r["end_frame_idx"]): r for r in all_preds}

    y_state: list[str] = []
    y_state_pred: list[str] = []
    y_h: list[int] = []
    y_c: list[int] = []
    p_h: list[float] = []
    p_c: list[float] = []
    for w in val:
        rec = lookup[(w["session_id"], w["end_frame_idx"])]
        y_state.append(w["current_state"])
        y_state_pred.append(str(rec["predicted_state"]))
        y_h.append(int(w["future_hesitation"]))
        y_c.append(int(w["future_correction"]))
        p_h.append(float(rec["future_hesitation_within_horizon"]))
        p_c.append(float(rec["future_correction_within_horizon"]))

    classes = _label_space(
        [w["current_state"] for w in windows],
        y_state_pred,
    )
    return {
        "current_state_deep": multiclass_metrics(y_state, y_state_pred, classes),
        "future_hesitation": binary_metrics(y_h, p_h, threshold=t_h),
        "future_correction": binary_metrics(y_c, p_c, threshold=t_c),
        "thresholds": {"future_hesitation": t_h, "future_correction": t_c},
        "windows": len(val),
    }


def evaluate_deep_windows(
    windows: list[SequenceWindow],
    model_path: str,
    thresholds: ThresholdConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a deep model against an explicit evaluation split."""
    threshold_cfg = thresholds or ThresholdConfig()
    predictions = infer_sequence_deep_windows(windows, model_path)
    lookup = {(r["session_id"], r["end_frame_idx"]): r for r in predictions}

    y_state: list[str] = []
    y_state_pred: list[str] = []
    y_h: list[int] = []
    y_c: list[int] = []
    p_h: list[float] = []
    p_c: list[float] = []
    for window in windows:
        rec = lookup[(window["session_id"], window["end_frame_idx"])]
        y_state.append(window["current_state"])
        y_state_pred.append(str(rec["predicted_state"]))
        y_h.append(int(window["future_hesitation"]))
        y_c.append(int(window["future_correction"]))
        p_h.append(float(rec["future_hesitation_within_horizon"]))
        p_c.append(float(rec["future_correction_within_horizon"]))

    classes = _label_space(y_state, y_state_pred)
    return {
        "current_state_deep": multiclass_metrics(y_state, y_state_pred, classes),
        "future_hesitation": binary_metrics(y_h, p_h, threshold=threshold_cfg.future_hesitation),
        "future_correction": binary_metrics(y_c, p_c, threshold=threshold_cfg.future_correction),
        "thresholds": {
            "future_hesitation": threshold_cfg.future_hesitation,
            "future_correction": threshold_cfg.future_correction,
        },
        "windows": len(windows),
    }


def evaluate_deep(input_path: str, model_path: str) -> dict[str, Any]:
    """Evaluate deep model on validation split using default thresholds."""
    tmp_threshold = {"future_hesitation": 0.5, "future_correction": 0.5}
    temp_path = Path("artifacts") / "_tmp_default_thresholds.json"
    save_json(temp_path, tmp_threshold)
    out = evaluate_deep_calibrated(input_path, model_path, str(temp_path))
    return out


def train_deep_multiseed(
    input_path: str,
    output_dir: str,
    seeds: list[int],
    window_size: int,
    horizon_frames: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
) -> dict[str, Any]:
    """Train deep model for multiple seeds and save aggregate metrics."""
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = Path(output_dir) / f"seed_{seed}"
        metrics = train_deep(
            input_path=input_path,
            output_dir=str(seed_dir),
            window_size=window_size,
            horizon_frames=horizon_frames,
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
            batch_size=batch_size,
        )
        per_seed.append({"seed": seed, "metrics": metrics})

    accs = [float(item["metrics"]["current_state_deep"]["accuracy"]) for item in per_seed]
    macro_f1 = [float(item["metrics"]["current_state_deep"]["macro_f1"]) for item in per_seed]

    aggregate = {
        "n_seeds": len(seeds),
        "current_state_accuracy_mean": fmean(accs),
        "current_state_macro_f1_mean": fmean(macro_f1),
        "per_seed": per_seed,
    }
    save_json(Path(output_dir) / "multiseed_metrics.json", aggregate)
    return aggregate


def compare_models(
    input_path: str,
    classical_model_path: str,
    deep_model_path: str,
    output_dir: str
) -> dict[str, Any]:
    """Compare rules, classical, and deep models and emit report files."""
    classical = evaluate_classical(input_path, classical_model_path)
    deep = evaluate_deep(input_path, deep_model_path)

    windows = _load_deep_windows_for_model(input_path, deep_model_path)
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


def compare_models_multiseed(
    input_path: str,
    classical_model_path: str,
    deep_root_dir: str,
    seeds: list[int],
    output_dir: str,
) -> dict[str, Any]:
    """Compare classical/rules/deep for each seed and aggregate tables."""
    records: list[dict[str, Any]] = []
    for seed in seeds:
        deep_model = Path(deep_root_dir) / f"seed_{seed}" / "deep_model.pt"
        if not deep_model.exists():
            deep_model = Path(deep_root_dir) / f"seed_{seed}" / "deep_model.json"
        report = compare_models(
            input_path=input_path,
            classical_model_path=classical_model_path,
            deep_model_path=str(deep_model),
            output_dir=str(Path(output_dir) / f"seed_{seed}"),
        )
        records.append({"seed": seed, "summary": report["summary"]})

    deep_acc = [float(r["summary"]["current_state_accuracy"]["deep"]) for r in records]
    deep_f1 = [float(r["summary"]["current_state_macro_f1"]["deep"]) for r in records]
    aggregate = {
        "n_seeds": len(seeds),
        "deep_current_state_accuracy_mean": fmean(deep_acc),
        "deep_current_state_macro_f1_mean": fmean(deep_f1),
        "per_seed": records,
    }
    save_json(Path(output_dir) / "comparison_multiseed.json", aggregate)
    return aggregate
