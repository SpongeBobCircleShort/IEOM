"""Test suite for inference interface."""

import pytest
from hesitation.inference import HesitationPredictor, Prediction


def test_prediction_dataclass():
    """Test Prediction output format."""
    pred = Prediction(
        state="normal_progress",
        state_probabilities={"normal_progress": 1.0},
        future_hesitation_prob=0.0,
        future_correction_prob=0.0,
        confidence=1.0,
    )
    assert pred.state == "normal_progress"
    assert pred.to_json() is not None
    payload = pred.to_dict()
    assert payload["state"] == "normal_progress"
    assert payload["state_probabilities"]["normal_progress"] == 1.0


def test_predictor_initialization():
    """Test predictor can initialize without model."""
    predictor = HesitationPredictor()
    assert predictor.window_size == 20
    assert predictor.frame_rate_hz == 10


def test_predictor_dummy_prediction():
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
    assert 0.0 <= pred.future_hesitation_prob <= 1.0
    assert 0.0 <= pred.future_correction_prob <= 1.0
    assert len(pred.state_probabilities) == 6


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
    assert vector[0] == 0.5  # mean_hand_speed
    assert vector[1] == 0.1  # pause_ratio
    assert vector[6] == 0.3  # human_robot_distance


def test_load_default():
    """Test load_default factory method."""
    predictor = HesitationPredictor.load_default()
    assert isinstance(predictor, HesitationPredictor)


def test_multiple_predictions_consistent():
    """Test that multiple predictions are deterministic."""
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

    # Same input should give same output
    assert pred1.state == pred2.state
    assert pred1.confidence == pred2.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
