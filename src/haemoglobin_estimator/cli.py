"""Evaluate or demonstrate RGB-based haemoglobin estimation."""

from __future__ import annotations

import argparse

from .dataset import generate_cohort, split
from .features import feature_vector
from .model import fit, mae, predict, rmse


def train_model():
    training, testing = split(generate_cohort())
    weights = fit([row for row, _ in training], [target for _, target in training])
    actual = [target for _, target in testing]
    estimated = predict(weights, [row for row, _ in testing])
    return weights, actual, estimated


def evaluate() -> dict[str, float]:
    _, actual, estimated = train_model()
    metrics = {"mae": mae(actual, estimated), "rmse": rmse(actual, estimated)}
    print("Synthetic held-out evaluation")
    print(f"MAE:  {metrics['mae']:.3f} g/dL")
    print(f"RMSE: {metrics['rmse']:.3f} g/dL")
    print("\nResearch demonstration only; not for clinical use.")
    return metrics


def estimate(red: float, green: float, blue: float, age: float, sex: str) -> float:
    weights, _, _ = train_model()
    value = predict(weights, [feature_vector(red, green, blue, age, sex)])[0]
    print(f"Demonstration estimate: {value:.2f} g/dL")
    print("Not clinically validated; do not use for medical decisions.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("evaluate")
    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--red", type=float, required=True)
    prediction.add_argument("--green", type=float, required=True)
    prediction.add_argument("--blue", type=float, required=True)
    prediction.add_argument("--age", type=float, required=True)
    prediction.add_argument("--sex", choices=("female", "male"), required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate()
    else:
        estimate(args.red, args.green, args.blue, args.age, args.sex)


if __name__ == "__main__":
    main()
