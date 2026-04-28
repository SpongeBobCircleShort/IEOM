from hesitation.policy.recommender import PolicyInput, recommend_policy


def test_policy_hold_on_strong_hesitation() -> None:
    rec = recommend_policy(
        PolicyInput(
            inferred_current_state="strong_hesitation",
            current_hesitation_probability=0.8,
            future_hesitation_probability=0.85,
            future_correction_probability=0.5,
            workspace_distance=0.4,
        )
    )
    assert rec.recommended_robot_mode == "hold"
    assert rec.recommend_confirmation_gate


def test_policy_proceed_on_ready_and_low_risk() -> None:
    rec = recommend_policy(
        PolicyInput(
            inferred_current_state="ready_for_robot_action",
            current_hesitation_probability=0.1,
            future_hesitation_probability=0.2,
            future_correction_probability=0.1,
            workspace_distance=0.6,
        )
    )
    assert rec.recommended_robot_mode == "proceed"
    assert rec.recommended_speed_scale > 0.8
