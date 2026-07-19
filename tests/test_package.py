import numpy as np

from wispine import __version__
from wispine.features import summarize_phase_ratio_windows


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_summarize_phase_ratio_windows() -> None:
    windows = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

    features = summarize_phase_ratio_windows(windows)

    assert features.shape == (27,)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[:3], windows.reshape(-1, 3).mean(axis=0))
