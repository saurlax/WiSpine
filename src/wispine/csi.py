from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from CSIKit.reader import get_reader
from CSIKit.util.csitools import get_CSI


@dataclass(frozen=True)
class CSIMetrics:
    amplitude: np.ndarray
    phase: np.ndarray
    timestamps: np.ndarray
    frame_count: int
    subcarrier_count: int


@dataclass(frozen=True)
class CSIPhaseRatio:
    phase_ratio: np.ndarray
    timestamps: np.ndarray
    frame_count: int
    subcarrier_count: int
    reference_rx: int
    compared_rx: tuple[int, ...]


def read_amplitude_phase(path: str | Path, *, scaled: bool = False) -> CSIMetrics:
    """Read an Intel 5300 CSI .dat file and return amplitude and phase arrays.

    Arrays are shaped as frames x subcarriers x rx_antennas x tx_antennas.
    """
    csi_path = Path(path)
    reader = get_reader(str(csi_path))
    csi_data = reader.read_file(str(csi_path), scaled=scaled)

    amplitude, frame_count, subcarrier_count = get_CSI(
        csi_data,
        metric="amplitude",
        extract_as_dBm=False,
        squeeze_output=False,
    )
    phase, _, _ = get_CSI(
        csi_data,
        metric="phase",
        squeeze_output=False,
    )

    return CSIMetrics(
        amplitude=amplitude,
        phase=phase,
        timestamps=np.asarray(csi_data.timestamps),
        frame_count=frame_count,
        subcarrier_count=subcarrier_count,
    )


def read_phase_ratio(
    path: str | Path,
    *,
    reference_rx: int = 0,
    scaled: bool = False,
    eps: float = 1e-8,
) -> CSIPhaseRatio:
    """Read CSI and return antenna phase-ratio features.

    The returned array is shaped as frames x subcarriers x compared_rx x tx_antennas.
    Each value is angle(H_rx / H_reference_rx), computed from the complex CSI.
    """
    csi_path = Path(path)
    reader = get_reader(str(csi_path))
    csi_data = reader.read_file(str(csi_path), scaled=scaled)

    csi, frame_count, subcarrier_count = get_CSI(
        csi_data,
        metric="complex",
        squeeze_output=False,
    )

    if csi.ndim != 4:
        raise ValueError(f"Expected CSI shape frames x subcarriers x rx x tx, got {csi.shape}.")

    rx_count = csi.shape[2]
    if not 0 <= reference_rx < rx_count:
        raise ValueError(f"reference_rx must be in [0, {rx_count - 1}], got {reference_rx}.")

    compared_rx = tuple(index for index in range(rx_count) if index != reference_rx)
    reference = csi[:, :, reference_rx : reference_rx + 1, :]
    compared = csi[:, :, compared_rx, :]

    ratio = compared / (reference + eps)
    phase_ratio = np.angle(ratio)

    return CSIPhaseRatio(
        phase_ratio=phase_ratio,
        timestamps=np.asarray(csi_data.timestamps),
        frame_count=frame_count,
        subcarrier_count=subcarrier_count,
        reference_rx=reference_rx,
        compared_rx=compared_rx,
    )
