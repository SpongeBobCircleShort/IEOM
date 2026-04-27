from hesitation.schemas.features import FeatureWindow


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def predict_risks(feature: FeatureWindow, high_pause_ratio: float) -> tuple[float, float, float]:
    hesitation_risk = _clip01((feature.pause_ratio / max(high_pause_ratio, 1e-6)) * 0.7 + feature.backtrack_ratio * 0.3)
    correction_risk = _clip01(feature.backtrack_ratio * 1.2)
    overlap_risk = _clip01((1.0 - min(1.0, feature.mean_workspace_distance)) * 0.7 + feature.pause_ratio * 0.3)
    return hesitation_risk, correction_risk, overlap_risk
