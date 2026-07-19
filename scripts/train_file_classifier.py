from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

from wispine.dataset import build_window_arrays, sample_to_json, split_samples
from wispine.features import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    CSISample,
    list_csi_samples,
    summarize_phase_ratio_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a file-level classifier on phase-ratio CSI statistics."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/file_phase_ratio")
    parser.add_argument("--cache-dir", default="artifacts/phase_ratio_windows")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--reference-rx", type=int, default=0)
    parser.add_argument("--scaled", action="store_true")
    parser.add_argument("--no-unwrap", action="store_true")
    parser.add_argument("--estimators", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def build_file_features(
    samples: list[CSISample],
    *,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
    reference_rx: int,
    scaled: bool,
    unwrap: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index:03d}/{len(samples):03d}] summarizing {sample.path}")
        windows, _ = build_window_arrays(
            [sample],
            window_size=window_size,
            stride=stride,
            max_windows_per_file=max_windows_per_file,
            reference_rx=reference_rx,
            scaled=scaled,
            unwrap=unwrap,
            cache_dir=cache_dir,
        )
        features.append(summarize_phase_ratio_windows(windows))
        labels.append(sample.label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def subject_names(samples: list[CSISample]) -> list[str]:
    return sorted({sample.path.stem.split("-", maxsplit=1)[0] for sample in samples})


def evaluate(model: ExtraTreesClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    predictions = model.predict(x)
    target_names = [INDEX_TO_CLASS[index] for index in range(len(INDEX_TO_CLASS))]
    return {
        "file_count": int(len(y)),
        "report": classification_report(
            y,
            predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    samples = list_csi_samples(args.data_dir, max_files_per_class=args.max_files_per_class)
    split = split_samples(
        samples,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    common = {
        "window_size": args.window_size,
        "stride": args.stride,
        "max_windows_per_file": args.max_windows_per_file,
        "reference_rx": args.reference_rx,
        "scaled": args.scaled,
        "unwrap": not args.no_unwrap,
        "cache_dir": cache_dir,
    }
    train_x, train_y = build_file_features(split.train, **common)
    val_x, val_y = build_file_features(split.val, **common)
    test_x, test_y = build_file_features(split.test, **common)

    model = ExtraTreesClassifier(
        n_estimators=args.estimators,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train_x, train_y)
    joblib.dump(model, output_dir / "model.joblib")

    split_subjects = {
        "train": subject_names(split.train),
        "val": subject_names(split.val),
        "test": subject_names(split.test),
    }
    metadata = {
        "args": vars(args),
        "classes": CLASS_TO_INDEX,
        "feature_count": int(train_x.shape[1]),
        "feature_extractor": "phase_ratio_file_statistics_v1",
        "model": "ExtraTreesClassifier",
        "sklearn_version": sklearn.__version__,
        "evaluation_scope": "known-subject file holdout",
        "split": {
            "train": [sample_to_json(sample) for sample in split.train],
            "val": [sample_to_json(sample) for sample in split.val],
            "test": [sample_to_json(sample) for sample in split.test],
        },
        "split_subjects": split_subjects,
        "train": evaluate(model, train_x, train_y),
        "val": evaluate(model, val_x, val_y),
        "test": evaluate(model, test_x, test_y),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"val": metadata["val"], "test": metadata["test"]}, indent=2))
    print(f"saved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
