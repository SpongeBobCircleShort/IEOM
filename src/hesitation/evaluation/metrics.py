from __future__ import annotations


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def multiclass_metrics(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, object]:
    cm: dict[str, dict[str, int]] = {c: {k: 0 for k in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[r][cls] for r in classes if r != cls)
        fn = sum(cm[cls][c] for c in classes if c != cls)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)

    return {
        "accuracy": _safe_div(correct, len(y_true)),
        "macro_f1": _safe_div(sum(f1s), len(f1s)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _binary_curve(y_true: list[int], y_prob: list[float]) -> list[tuple[float, int]]:
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)
    return pairs


def auroc_score(y_true: list[int], y_prob: list[float]) -> float:
    pairs = _binary_curve(y_true, y_prob)
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / pos
        fpr = fp / neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return auc


def auprc_score(y_true: list[int], y_prob: list[float]) -> float:
    pairs = _binary_curve(y_true, y_prob)
    pos = sum(y_true)
    if pos == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    prev_precision = 1.0
    area = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, pos)
        area += (recall - prev_recall) * (precision + prev_precision) / 2
        prev_recall, prev_precision = recall, precision
    return area


def binary_metrics(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, object]:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc_score(y_true, y_prob),
        "auprc": auprc_score(y_true, y_prob),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "threshold": threshold,
    }
