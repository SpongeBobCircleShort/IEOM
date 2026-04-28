from hesitation.io.config import load_config


def test_base_config_loads() -> None:
    cfg = load_config("configs/base.yaml")
    assert cfg["frame_rate_hz"] > 0


def test_rules_config_has_expected_keys() -> None:
    cfg = load_config("configs/baseline/rules_v1.yaml")
    assert "thresholds" in cfg
    assert "risk" in cfg
