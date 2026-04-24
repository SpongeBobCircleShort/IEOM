from dataclasses import dataclass, field


@dataclass(slots=True)
class NoiseConfig:
    position_jitter_std: float = 0.01
    dropout_prob: float = 0.01


@dataclass(slots=True)
class ScenarioConfig:
    name: str = "default"
    n_steps: int = 6
    step_duration_mean_s: float = 4.0
    step_duration_std_s: float = 1.0
    mild_hesitation_prob: float = 0.18
    strong_hesitation_prob: float = 0.08
    correction_rework_prob: float = 0.10
    overlap_risk_prob: float = 0.08
    ready_for_robot_action_prob: float = 0.20
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0")
