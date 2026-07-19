from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from wispine.features import CSISample, load_phase_ratio_sequence, make_windows


@dataclass(frozen=True)
class DatasetSplit:
    train: list[CSISample]
    val: list[CSISample]
    test: list[CSISample]


def split_samples(
    samples: list[CSISample],
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> DatasetSplit:
    labels = [sample.label for sample in samples]
    class_count = len(set(labels))
    test_size = max(test_size, class_count / len(samples))
    train_val, test = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [sample.label for sample in train_val]
    val_size = max(val_size, class_count / len(samples))
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )
    return DatasetSplit(train=train, val=val, test=test)


def build_window_arrays(
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
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for index, sample in enumerate(samples, start=1):
        cache_path = make_cache_path(
            sample,
            cache_dir=cache_dir,
            window_size=window_size,
            stride=stride,
            max_windows_per_file=max_windows_per_file,
            reference_rx=reference_rx,
            scaled=scaled,
            unwrap=unwrap,
        )
        if cache_path is not None and cache_path.exists():
            print(f"[{index:03d}/{len(samples):03d}] loading cache {cache_path}")
            windows = np.load(cache_path)["windows"]
        else:
            print(f"[{index:03d}/{len(samples):03d}] extracting {sample.path}")
            sequence = load_phase_ratio_sequence(
                sample.path,
                reference_rx=reference_rx,
                scaled=scaled,
                unwrap=unwrap,
            )
            windows = make_windows(
                sequence,
                window_size=window_size,
                stride=stride,
                max_windows=max_windows_per_file,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, windows=windows)
        labels = np.full((windows.shape[0],), sample.label, dtype=np.int64)
        all_windows.append(windows)
        all_labels.append(labels)

    return np.concatenate(all_windows), np.concatenate(all_labels)


def make_cache_path(
    sample: CSISample,
    *,
    cache_dir: Path | None,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
    reference_rx: int,
    scaled: bool,
    unwrap: bool,
) -> Path | None:
    if cache_dir is None:
        return None

    key = "|".join(
        [
            str(sample.path.resolve()),
            str(sample.path.stat().st_mtime_ns),
            str(window_size),
            str(stride),
            str(max_windows_per_file),
            str(reference_rx),
            str(scaled),
            str(unwrap),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / sample.class_name / f"{sample.path.stem}-{digest}.npz"


def sample_to_json(sample: CSISample) -> dict[str, str | int]:
    return {
        "path": str(sample.path),
        "label": sample.label,
        "class_name": sample.class_name,
    }
