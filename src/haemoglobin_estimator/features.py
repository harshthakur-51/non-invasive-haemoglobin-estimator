"""RGB-derived feature engineering."""

from __future__ import annotations


def feature_vector(red: float, green: float, blue: float, age: float, sex: str) -> list[float]:
    if min(red, green, blue) <= 0:
        raise ValueError("RGB channel measurements must be positive")
    encoded_sex = 1.0 if sex.lower() == "male" else 0.0
    chroma_total = red + green + blue
    return [
        red / chroma_total,
        green / chroma_total,
        blue / chroma_total,
        red / green,
        red / blue,
        (red - green) / chroma_total,
        age / 100.0,
        encoded_sex,
    ]
