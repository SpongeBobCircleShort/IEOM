import pytest


pytestmark = pytest.mark.deep


def test_phase3_model_forward_shapes() -> None:
    torch = pytest.importorskip("torch")

    from hesitation.ml.deep import GRURiskModel, STATE_CLASSES, SEQUENCE_FEATURE_ORDER

    model = GRURiskModel(
        input_size=len(SEQUENCE_FEATURE_ORDER),
        hidden_size=16,
        state_dim=len(STATE_CLASSES),
    )
    batch = torch.randn(4, 12, len(SEQUENCE_FEATURE_ORDER))
    outputs = model(batch)

    assert outputs["state_logits"].shape == (4, len(STATE_CLASSES))
    assert outputs["future_hesitation_logits"].shape == (4,)
    assert outputs["future_correction_logits"].shape == (4,)
