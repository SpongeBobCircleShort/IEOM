from hesitation.baselines.predictor import predict_risks
from hesitation.labeling.heuristics import infer_state_from_features
from hesitation.schemas.features import FeatureWindow
from hesitation.schemas.labels import LabelOutput


def classify_window(
    feature: FeatureWindow,
    thresholds: dict[str, float],
    risk_cfg: dict[str, float],
) -> LabelOutput:
    state, triggered = infer_state_from_features(
        feature,
        pause_ratio_mild=thresholds["pause_ratio_mild"],
        pause_ratio_strong=thresholds["pause_ratio_strong"],
        correction_backtrack=thresholds["correction_backtrack"],
        ready_progress=thresholds["ready_progress"],
        overlap_distance=thresholds["overlap_distance"],
    )
    h_risk, c_risk, o_risk = predict_risks(feature, high_pause_ratio=risk_cfg["high_pause_ratio"])
    return LabelOutput(
        current_state=state,
        hesitation_risk=h_risk,
        correction_rework_risk=c_risk,
        overlap_risk=o_risk,
        triggered_rules=triggered,
    )
