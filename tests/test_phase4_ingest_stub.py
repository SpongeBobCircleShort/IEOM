from __future__ import annotations

import json
from pathlib import Path

from hesitation.ingest.stub import ingest_video_or_pose_stub
from tests.phase4_support import sample_session_frames


def test_ingest_jsonl_stub_loads_frames(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for frame in sample_session_frames(n_frames=8):
            handle.write(json.dumps(frame.model_dump()) + "\n")

    result = ingest_video_or_pose_stub(path)
    assert result.status == "loaded_frames"
    assert len(result.frames) == 8


def test_ingest_video_stub_returns_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not-a-real-video")

    result = ingest_video_or_pose_stub(path)
    assert result.status == "stub_only"
    assert "not implemented" in result.message.lower()
