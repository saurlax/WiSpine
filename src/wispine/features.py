from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wispine.csi import read_phase_ratio

CLASS_TO_INDEX = {
    "kyphosis": 0,
    "normal": 1,
    "scoliosis": 2,
}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}


@dataclass(frozen=True)
class CSISample:
    path: Path
    label: int
    class_name: str


def list_csi_samples(
    data_dir: str | Path,
    *,
    max_files_per_class: int | None = None,
) -> list[CSISample]:
    """List CSI .dat files arranged as data/class_name/*.dat."""
    root = Path(data_dir)
    samples: list[CSISample] = []

    for class_name, label in CLASS_TO_INDEX.items():
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")

        files = sorted(class_dir.glob("*.dat"))
        if max_files_per_class is not None:
            files = files[:max_files_per_class]

        samples.extend(CSISample(path=file, label=label, class_name=class_name) for file in files)

    if not samples:
        raise FileNotFoundError(f"No .dat files found under {root}")

    return samples


def load_phase_ratio_sequence(
    path: str | Path,
    *,
    reference_rx: int = 0,
    scaled: bool = False,
    unwrap: bool = True,
    expected_rx_count: int = 3,
    expected_tx_count: int = 3,
) -> np.ndarray:
    """Load one CSI file as a frames x features phase-ratio sequence."""
    ratio = read_phase_ratio(path, reference_rx=reference_rx, scaled=scaled)
    sequence = ratio.phase_ratio

    if unwrap:
        sequence = np.unwrap(sequence, axis=0)

    frame_count, subcarrier_count = sequence.shape[:2]
    fixed = np.zeros(
        (
            frame_count,
            subcarrier_count,
            expected_rx_count - 1,
            expected_tx_count,
        ),
        dtype=np.float32,
    )

    for source_rx_index, rx_index in enumerate(ratio.compared_rx):
        if rx_index > reference_rx:
            target_rx_index = rx_index - 1
        else:
            target_rx_index = rx_index
        if target_rx_index >= fixed.shape[2]:
            continue

        tx_count = min(sequence.shape[3], expected_tx_count)
        fixed[:, :, target_rx_index, :tx_count] = sequence[:, :, source_rx_index, :tx_count]

    return fixed.reshape(frame_count, -1)


def make_windows(
    sequence: np.ndarray,
    *,
    window_size: int,
    stride: int,
    max_windows: int | None = None,
) -> np.ndarray:
    """Cut a frames x features sequence into fixed-size windows."""
    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence shape frames x features, got {sequence.shape}.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    frame_count, feature_count = sequence.shape
    if frame_count < window_size:
        padded = np.zeros((window_size, feature_count), dtype=np.float32)
        padded[:frame_count] = sequence
        return padded[None, :, :]

    starts = range(0, frame_count - window_size + 1, stride)
    windows = [sequence[start : start + window_size] for start in starts]
    if max_windows is not None:
        windows = windows[:max_windows]

    return np.asarray(windows, dtype=np.float32)
