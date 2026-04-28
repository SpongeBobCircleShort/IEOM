from hesitation.deep.pipeline import _compute_pos_weight


def test_pos_weight_increases_when_imbalanced() -> None:
    balanced = _compute_pos_weight([0, 1, 0, 1])
    imbalanced = _compute_pos_weight([0, 0, 0, 0, 1])
    assert balanced >= 1.0
    assert imbalanced > balanced
