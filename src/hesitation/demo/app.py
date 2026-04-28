from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from hesitation.ingest.stub import ingest_video_or_pose_stub
from hesitation.io.config import load_config
from hesitation.ml.pipeline import train_classical
from hesitation.schemas.events import FrameObservation
from hesitation.serving import (
    ADVISORY_NOTICE,
    ArtifactSpec,
    compare_report_sources,
    infer_from_frames,
    inspect_artifact_path,
    recommend_from_inference,
    supported_backends,
)
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import NoiseConfig, ScenarioConfig


DEMO_SCENARIO_CONFIGS = (
    "configs/simulation/default_scene.yaml",
    "configs/simulation/stress_scene.yaml",
    "configs/simulation/correction_heavy_scene.yaml",
    "configs/simulation/domain_gap_scene.yaml",
)


def _load_demo_scenario(path: str) -> ScenarioConfig:
    """Load one named demo scenario from the repo configs."""
    payload = load_config(path)
    noise_payload = payload.get("noise", {})
    return ScenarioConfig(**{**payload, "noise": NoiseConfig(**noise_payload)})


@st.cache_data(show_spinner=False)
def _build_demo_dataset(n_sessions: int, seed: int) -> tuple[list[FrameObservation], list[dict[str, object]]]:
    """Generate mixed-scenario demo sessions and their latent-state training rows."""
    scenarios = [_load_demo_scenario(path) for path in DEMO_SCENARIO_CONFIGS]
    frames: list[FrameObservation] = []
    rows: list[dict[str, object]] = []
    for index in range(n_sessions):
        scenario = scenarios[index % len(scenarios)]
        trajectory, latent = generate_session(
            session_id=f"{scenario.name}_session_{index}",
            scenario=scenario,
            frame_rate_hz=10,
            seed=seed + index,
        )
        frames.extend(trajectory.frames)
        for frame, state in zip(trajectory.frames, latent, strict=False):
            row = frame.model_dump()
            row["latent_state"] = state.value
            row["dataset_name"] = scenario.name
            rows.append(row)
    return frames, rows


def _generate_synthetic_frames(n_sessions: int, seed: int) -> list[FrameObservation]:
    """Generate synthetic frame observations for demo exploration."""
    frames, _ = _build_demo_dataset(n_sessions=n_sessions, seed=seed)
    return frames


@st.cache_resource(show_spinner=False)
def _build_demo_classical_artifact(n_sessions: int, seed: int) -> str:
    """Train a small classical artifact for the Streamlit demo."""
    _, rows = _build_demo_dataset(n_sessions=n_sessions, seed=seed)
    artifact_root = Path(tempfile.mkdtemp(prefix="hesitation_demo_"))
    data_path = artifact_root / "demo_synth.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    train_classical(
        input_path=str(data_path),
        output_dir=str(artifact_root),
        window_size=15,
        pause_speed_threshold=0.03,
        horizon_frames=10,
    )
    return str(artifact_root / "classical_model.json")


def _load_jsonl_frames(uploaded_file: Any) -> list[FrameObservation]:
    """Parse uploaded JSONL rows into validated frame observations."""
    rows = uploaded_file.getvalue().decode("utf-8").splitlines()
    return [FrameObservation.model_validate(json.loads(line)) for line in rows if line.strip()]


def _group_sessions(frames: list[FrameObservation]) -> dict[str, list[FrameObservation]]:
    """Group frame observations by session identifier."""
    grouped: dict[str, list[FrameObservation]] = {}
    for frame in frames:
        grouped.setdefault(frame.session_id, []).append(frame)
    for session_id in grouped:
        grouped[session_id] = sorted(grouped[session_id], key=lambda item: item.frame_idx)
    return grouped


def _render_inference_result(result: dict[str, Any], policy: dict[str, Any]) -> None:
    """Render the main inference and policy outputs."""
    state_col, risk_col, policy_col = st.columns(3)
    state_col.metric("Predicted State", result["predicted_state"])
    state_col.metric("Current Hesitation", f"{result['current_hesitation_probability']:.3f}")
    risk_col.metric("Future Hesitation", f"{result['future_hesitation_probability']:.3f}")
    risk_col.metric("Future Correction", f"{result['future_correction_probability']:.3f}")
    policy_col.metric("Robot Mode", policy["recommended_robot_mode"])
    policy_col.metric("Wait (ms)", int(policy["recommended_wait_time_ms"]))

    st.subheader("State Probabilities")
    st.json(result["state_probabilities"])
    st.subheader("Feature Window")
    st.json(result["feature_window"])
    st.subheader("Policy Recommendation")
    st.json(policy)


def _default_session_index(session_ids: list[str]) -> int:
    """Prefer a higher-risk demo session instead of the first normal one."""
    preferred_tokens = ("stress", "correction", "domain_gap")
    for index, session_id in enumerate(session_ids):
        if any(token in session_id for token in preferred_tokens):
            return index
    return 0


def _render_artifact_viewer() -> None:
    """Render report inspection and comparison widgets."""
    st.header("Artifact Viewer")
    inspect_path = st.text_input("Artifact or report path", value="")
    if inspect_path:
        try:
            inspection = inspect_artifact_path(inspect_path).to_dict()
            st.write("Artifact manifest")
            st.json({
                "json_files": inspection["json_files"],
                "csv_files": inspection["csv_files"],
                "markdown_files": inspection["markdown_files"],
                "chosen_report_path": inspection["chosen_report_path"],
            })
            if inspection["chosen_report_payload"] is not None:
                st.write("Chosen JSON report")
                st.json(inspection["chosen_report_payload"])
            if inspection["preview_rows"]:
                st.write("CSV preview")
                st.table(inspection["preview_rows"])
            if inspection["markdown_preview"]:
                st.write("Markdown preview")
                st.code(inspection["markdown_preview"], language="markdown")
        except Exception as exc:  # pragma: no cover - visual feedback path
            st.error(str(exc))

    left_path, right_path = st.columns(2)
    left_value = left_path.text_input("Compare left path", value="")
    right_value = right_path.text_input("Compare right path", value="")
    if left_value and right_value:
        try:
            comparison = compare_report_sources(left_value, right_value)
            st.write("Shared numeric metrics", comparison["shared_metric_count"])
            st.table(comparison["comparison_rows"][:50])
        except Exception as exc:  # pragma: no cover - visual feedback path
            st.error(str(exc))


def _render_ingestion_stub() -> None:
    """Render the optional video/pose ingestion stub controls."""
    st.header("Optional Video/Pose Stub")
    uploaded = st.file_uploader("Upload video or precomputed JSONL", type=["jsonl", "mp4", "mov", "avi", "mkv"], key="stub_upload")
    if uploaded is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as handle:
        handle.write(uploaded.getvalue())
        temp_path = handle.name

    result = ingest_video_or_pose_stub(temp_path).to_dict()
    st.json(result)


def main() -> None:
    """Run the Phase 4 Streamlit demo."""
    st.set_page_config(page_title="Hesitation Demo", layout="wide")
    st.title("Hesitation Phase 4 Demo")
    st.caption(ADVISORY_NOTICE)

    source = st.sidebar.selectbox("Input source", ["Synthetic sample", "Upload JSONL"])
    available_backends = supported_backends()
    default_backend_index = available_backends.index("classical") if "classical" in available_backends else 0
    backend = st.sidebar.selectbox("Backend", available_backends, index=default_backend_index)
    model_path = st.sidebar.text_input("Model path", value="")
    threshold_path = st.sidebar.text_input("Threshold path (deep only)", value="")
    pause_speed_threshold = st.sidebar.number_input("Pause speed threshold", min_value=0.0, value=0.03, step=0.01)

    demo_model_path: str | None = None
    if source == "Synthetic sample":
        synthetic_sessions = st.sidebar.slider("Synthetic sessions", min_value=1, max_value=8, value=4)
        synthetic_seed = st.sidebar.number_input("Synthetic seed", min_value=1, value=17, step=1)
        frames = _generate_synthetic_frames(
            n_sessions=synthetic_sessions,
            seed=synthetic_seed,
        )
        if backend == "classical" and not model_path:
            demo_model_path = _build_demo_classical_artifact(
                n_sessions=synthetic_sessions,
                seed=synthetic_seed,
            )
    else:
        uploaded = st.sidebar.file_uploader("Upload frame JSONL", type=["jsonl"])
        frames = _load_jsonl_frames(uploaded) if uploaded is not None else []

    if not frames:
        st.info("Provide synthetic or uploaded JSONL frames to run the demo.")
        _render_ingestion_stub()
        _render_artifact_viewer()
        return

    sessions = _group_sessions(frames)
    session_ids = sorted(sessions)
    session_id = st.selectbox("Session", session_ids, index=_default_session_index(session_ids))
    session_frames = sessions[session_id]
    st.write(f"Loaded {len(session_frames)} frames for `{session_id}`.")
    if demo_model_path:
        st.caption(f"Using demo classical artifact: `{demo_model_path}`")

    artifact = ArtifactSpec(
        backend=backend,
        model_path=model_path or demo_model_path or None,
        threshold_path=threshold_path or None,
        pause_speed_threshold=float(pause_speed_threshold),
    )

    if st.button("Run inference", type="primary"):
        try:
            result = infer_from_frames(session_frames, artifact)
            policy = recommend_from_inference(result).to_dict()
            _render_inference_result(result.to_dict(), policy)
        except Exception as exc:  # pragma: no cover - visual feedback path
            st.error(str(exc))

    _render_ingestion_stub()
    _render_artifact_viewer()


if __name__ == "__main__":
    main()
