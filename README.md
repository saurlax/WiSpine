# WiSpine

WiFi CSI based spine detection research project.

## Setup

This project is configured for `uv`.

```powershell
uv sync
uv run pytest
```

Put raw and processed experiment data under `data/`; those files are ignored by git by default.

## File-level classifier

The classifier uses antenna phase-ratio CSI features:

```text
angle(H_rx / H_reference_rx)
```

Each recording is summarized into robust file-level statistics and classified by ExtraTrees.
Train and evaluate the model with:

```powershell
uv run python scripts/train_file_classifier.py --output-dir outputs/file_phase_ratio
uv run python scripts/evaluate_file_classifier.py --model-dir outputs/file_phase_ratio --split test
```

This evaluation keeps every source file wholly within one split, so overlapping windows from the
same recording cannot leak across train and test. It assumes each subject is represented in the
training set. Use a subject-held-out evaluation before claiming performance on unseen people.

Outputs include:

- `model.joblib`: trained ExtraTrees classifier.
- `metrics.json`: train, validation, and test reports plus split metadata.
- `evaluation_test.json`: independently generated evaluation report.

Preprocessed windows are cached under `artifacts/phase_ratio_windows/` by default.
