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
