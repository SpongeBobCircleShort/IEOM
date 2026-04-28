"""Standalone test for inference interface (no dependencies on main package)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only our inference module (not the whole package)
from hesitation.inference.predictor import HesitationPredictor, Prediction


def test_prediction_creation():
    """Test Prediction dataclass creation."""
    pred = Prediction(
        state="normal_progress",
        state_probabilities={
            "normal_progress": 0.8,
            "mild_hesitation": 0.1,
            "strong_hesitation": 0.05,
            "correction_rework": 0.02,
            "ready_for_robot_action": 0.02,
            "overlap_risk": 0.01,
        },
        future_hesitation_prob=0.1,
        future_correction_prob=0.05,
        confidence=0.8,
    )
    assert pred.state == "normal_progress"
    assert pred.confidence == 0.8
    print("✓ Prediction dataclass works")


def test_predictor_init():
    """Test predictor initialization."""
    predictor = HesitationPredictor()
    assert predictor.window_size == 20
    assert predictor.frame_rate_hz == 10
    print("✓ Predictor initialization works")


def test_feature_vector_conversion():
    """Test feature dict to vector conversion."""
    features = {
        "mean_hand_speed": 0.5,
        "pause_ratio": 0.1,
        "progress_delta": 0.8,
        "reversal_count": 1,
        "retry_count": 0,
        "task_step_id": 2,
        "human_robot_distance": 0.3,
    }
    vector = HesitationPredictor._dict_to_vector(features)
    assert len(vector) == 7
    assert vector[0] == 0.5
    print("✓ Feature vector conversion works")


def test_dummy_prediction():
    """Test dummy prediction (no model loaded)."""
    predictor = HesitationPredictor()
    features = {
        "mean_hand_speed": 0.5,
        "pause_ratio": 0.1,
        "progress_delta": 0.8,
        "reversal_count": 0,
        "retry_count": 0,
        "task_step_id": 2,
        "human_robot_distance": 0.3,
    }
    pred = predictor.predict_single(features)

    assert isinstance(pred, Prediction)
    assert pred.state in [
        "normal_progress",
        "mild_hesitation",
        "strong_hesitation",
        "correction_rework",
        "ready_for_robot_action",
        "overlap_risk",
    ]
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.state_probabilities) == 6
    print(f"✓ Dummy prediction works: {pred.state}")


def test_determinism():
    """Test predictions are deterministic."""
    predictor = HesitationPredictor()
    features = {
        "mean_hand_speed": 0.5,
        "pause_ratio": 0.1,
        "progress_delta": 0.8,
        "reversal_count": 0,
        "retry_count": 0,
        "task_step_id": 2,
        "human_robot_distance": 0.3,
    }
    pred1 = predictor.predict_single(features)
    pred2 = predictor.predict_single(features)

    assert pred1.state == pred2.state
    assert pred1.confidence == pred2.confidence
    print("✓ Predictions are deterministic")


def test_json_export():
    """Test JSON export."""
    pred = Prediction(
        state="mild_hesitation",
        state_probabilities={
            "normal_progress": 0.2,
            "mild_hesitation": 0.6,
            "strong_hesitation": 0.1,
            "correction_rework": 0.05,
            "ready_for_robot_action": 0.03,
            "overlap_risk": 0.02,
        },
        future_hesitation_prob=0.3,
        future_correction_prob=0.1,
        confidence=0.6,
    )
    json_str = pred.to_json()
    assert "mild_hesitation" in json_str
    assert "0.6" in json_str or "0.60" in json_str
    print("✓ JSON export works")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Hesitation Inference Interface")
    print("=" * 60 + "\n")

    test_prediction_creation()
    test_predictor_init()
    test_feature_vector_conversion()
    test_dummy_prediction()
    test_determinism()
    test_json_export()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")
