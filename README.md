# WiSpine

WiFi CSI based spine detection research project.

## Setup

This project is configured for `uv`.

```powershell
uv sync
uv run pytest
```

Put raw and processed experiment data under `data/`; those files are ignored by git by default.

## Train TCN classifier

The first training pipeline uses antenna phase-ratio CSI features:

```text
angle(H_rx / H_reference_rx)
```

Run a quick smoke test:

```powershell
uv run python scripts/train_tcn.py --max-files-per-class 4 --max-windows-per-file 1 --window-size 128 --stride 128 --epochs 1 --output-dir outputs/tcn_smoke
```

Run a fuller experiment:

```powershell
uv run python scripts/train_tcn.py --epochs 30 --window-size 512 --stride 256 --output-dir outputs/tcn_phase_ratio
```

Outputs include:

- `best_model.pt`: best validation checkpoint.
- `metrics.json`: train/validation history, test report, confusion matrix, and split metadata.
- `normalization.npz`: feature normalization statistics.

Preprocessed windows are cached under `artifacts/phase_ratio_windows/` by default.
