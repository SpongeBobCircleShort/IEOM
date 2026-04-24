import random


def apply_jitter(value: float, jitter_std: float, rng: random.Random) -> float:
    if jitter_std <= 0:
        return value
    return value + rng.gauss(0.0, jitter_std)
