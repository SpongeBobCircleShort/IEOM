import importlib.util


def test_torch_model_forward_if_available() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch

    from hesitation.deep.model import TorchGRUMultiHead

    model = TorchGRUMultiHead(input_dim=8, hidden_dim=16, n_state_classes=6)
    x = torch.zeros((3, 20, 8), dtype=torch.float32)
    state, fh, fc = model(x)
    assert state.shape == (3, 6)
    assert fh.shape == (3, 1)
    assert fc.shape == (3, 1)
