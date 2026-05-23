"""Synthetic cohort for reproducible public evaluation."""

from __future__ import annotations

import random

from .features import feature_vector


def generate_cohort(size: int = 640, seed: int = 14) -> list[tuple[list[float], float]]:
    rng = random.Random(seed)
    data = []
    for _ in range(size):
        age = rng.uniform(18, 72)
        sex = "male" if rng.random() < 0.5 else "female"
        hb = rng.uniform(9.0, 17.0)
        complexion = rng.uniform(78, 154)
        red = complexion + hb * 3.6 + rng.gauss(0, 2.3)
        green = complexion + 20 - hb * 0.52 + rng.gauss(0, 2.0)
        blue = complexion + 11 - hb * 0.9 + rng.gauss(0, 2.2)
        target = hb + (0.22 if sex == "male" else -0.08) + (age - 42) * -0.003
        data.append((feature_vector(red, green, blue, age, sex), target))
    return data


def split(data: list[tuple[list[float], float]], ratio: float = 0.8):
    boundary = int(len(data) * ratio)
    return data[:boundary], data[boundary:]
