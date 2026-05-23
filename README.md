# Non-Invasive Haemoglobin Estimator

A research demonstration of a facial RGB feature pipeline for estimating
haemoglobin concentration. The project uses a transparent regression model with
skin-region colour ratios, age, and sex encoded as features.

**Important:** This project is not a medical device, has not been clinically
validated, and must not be used for diagnosis, screening, or treatment decisions.

## What is included

- Feature engineering from mean facial RGB measurements.
- Synthetic cohort generator for public, reproducible evaluation.
- Ridge regression fit and held-out MAE/RMSE report.
- CLI prediction example for supplied RGB observations.

The starter intentionally avoids publishing private subject data. Later, an
approved data adapter and an OpenCV webcam capture layer can replace the synthetic
input path.

## Run

```bash
python -m haemoglobin_estimator.cli evaluate
python -m haemoglobin_estimator.cli predict --red 162 --green 118 --blue 103 --age 24 --sex female
```

Install with `python -m pip install -e .`, or set `PYTHONPATH=src`.

## Test

```bash
python -m unittest discover -s tests -v
```

## Responsible extension

Before using measured participant data, add consent and de-identification
documentation, dataset provenance, train/test subject separation, skin-tone
stratified reporting, and clinically meaningful error analysis.
