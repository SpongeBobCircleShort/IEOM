from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from hesitation.demo.app import _generate_synthetic_frames, _group_sessions


def test_demo_helpers_smoke() -> None:
    frames = _generate_synthetic_frames(n_sessions=2, seed=21)
    sessions = _group_sessions(frames)
    assert frames
    assert len(sessions) == 2
