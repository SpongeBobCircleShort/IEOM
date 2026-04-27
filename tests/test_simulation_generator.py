from hesitation.schemas.labels import HesitationState
from hesitation.simulation.generator import generate_session
from hesitation.simulation.scenario import ScenarioConfig


def test_generator_is_deterministic_by_seed() -> None:
    scenario = ScenarioConfig()
    traj_a, latent_a = generate_session("s", scenario, frame_rate_hz=10, seed=11)
    traj_b, latent_b = generate_session("s", scenario, frame_rate_hz=10, seed=11)
    assert traj_a.frames[0].model_dump() == traj_b.frames[0].model_dump()
    assert latent_a[:20] == latent_b[:20]


def test_generator_outputs_known_states() -> None:
    scenario = ScenarioConfig()
    _, latent = generate_session("s", scenario, frame_rate_hz=10, seed=12)
    assert all(isinstance(s, HesitationState) for s in latent)
