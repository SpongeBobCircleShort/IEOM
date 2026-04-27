import random
from dataclasses import dataclass

from hesitation.schemas.events import FrameObservation
from hesitation.schemas.labels import HesitationState
from hesitation.schemas.trajectories import SessionTrajectory
from hesitation.simulation.noise import apply_jitter
from hesitation.simulation.scenario import ScenarioConfig


@dataclass(frozen=True)
class SyntheticFrame:
    observation: FrameObservation
    latent_state: HesitationState


def _sample_state(scenario: ScenarioConfig, rng: random.Random) -> HesitationState:
    roll = rng.random()
    cuts = [
        (scenario.strong_hesitation_prob, HesitationState.STRONG_HESITATION),
        (scenario.mild_hesitation_prob, HesitationState.MILD_HESITATION),
        (scenario.correction_rework_prob, HesitationState.CORRECTION_REWORK),
        (scenario.overlap_risk_prob, HesitationState.OVERLAP_RISK),
        (scenario.ready_for_robot_action_prob, HesitationState.READY_FOR_ROBOT_ACTION),
    ]
    running = 0.0
    for p, s in cuts:
        running += p
        if roll < running:
            return s
    return HesitationState.NORMAL_PROGRESS


def _inject_ambiguous_boundary(
    state: HesitationState,
    scenario: ScenarioConfig,
    rng: random.Random,
) -> HesitationState:
    if rng.random() > scenario.ambiguous_transition_prob:
        return state
    if state == HesitationState.NORMAL_PROGRESS:
        return HesitationState.MILD_HESITATION
    if state == HesitationState.MILD_HESITATION:
        return HesitationState.STRONG_HESITATION
    if state == HesitationState.STRONG_HESITATION:
        return HesitationState.MILD_HESITATION
    return state


def generate_session(
    session_id: str,
    scenario: ScenarioConfig,
    frame_rate_hz: int,
    seed: int,
) -> tuple[SessionTrajectory, list[HesitationState]]:
    rng = random.Random(seed)
    frames: list[FrameObservation] = []
    latent: list[HesitationState] = []
    dt_ms = int(1000 / frame_rate_hz)
    total_frames = int(scenario.n_steps * scenario.step_duration_mean_s * frame_rate_hz)
    x = 0.0
    y = 0.0

    correction_burst = rng.random() < scenario.correction_burst_prob
    burst_start = rng.randint(max(1, total_frames // 4), max(2, (3 * total_frames) // 4)) if total_frames > 4 else 0
    burst_len = max(2, total_frames // 12)

    for frame_idx in range(total_frames):
        state = _sample_state(scenario, rng)
        state = _inject_ambiguous_boundary(state, scenario, rng)
        if correction_burst and burst_start <= frame_idx < burst_start + burst_len:
            state = HesitationState.CORRECTION_REWORK
        latent.append(state)

        base_speed = 0.12 / scenario.domain_gap_scale
        if state == HesitationState.MILD_HESITATION:
            speed = base_speed * (0.55 + 0.15 * rng.random())
        elif state == HesitationState.STRONG_HESITATION:
            speed = base_speed * (0.15 + 0.2 * rng.random())
        elif state == HesitationState.CORRECTION_REWORK:
            speed = -base_speed * (0.25 + 0.35 * rng.random())
        else:
            speed = base_speed * (0.85 + 0.3 * rng.random())

        accel = speed - (frames[-1].hand_speed if frames else 0.0)
        x += speed / frame_rate_hz
        y += ((0.02 if frame_idx % 3 == 0 else -0.01) * scenario.domain_gap_scale)
        x = apply_jitter(x, scenario.noise.position_jitter_std * scenario.domain_gap_scale, rng)
        y = apply_jitter(y, scenario.noise.position_jitter_std * scenario.domain_gap_scale, rng)

        progress = max(0.0, min(1.0, x / scenario.n_steps))
        workspace_dist = abs(0.5 - x)
        is_dropout = rng.random() < scenario.noise.dropout_prob

        frames.append(
            FrameObservation(
                session_id=session_id,
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * dt_ms,
                task_step_id=min(scenario.n_steps - 1, int(progress * scenario.n_steps)),
                hand_x=x,
                hand_y=y,
                hand_speed=abs(speed),
                hand_accel=accel,
                distance_to_robot_workspace=max(0.0, workspace_dist),
                progress=progress,
                confidence=0.0 if is_dropout else 1.0,
                is_dropout=is_dropout,
            )
        )

    return SessionTrajectory(session_id=session_id, frame_rate_hz=frame_rate_hz, frames=frames), latent
