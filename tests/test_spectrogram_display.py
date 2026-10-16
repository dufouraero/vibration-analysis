import numpy as np
import pytest

from vibration_analysis.plots import spectrogram_color_limits


def test_auto_color_headroom_preserves_psd():
    density = np.array([0.0, 0.01, 100.0])
    original = density.copy()
    assert spectrogram_color_limits(density) == pytest.approx((-34, 26))
    np.testing.assert_array_equal(density, original)
    assert spectrogram_color_limits(density, upper_db=30, lower_db=-20) == (-20, 30)
    assert spectrogram_color_limits(density, upper_db=30) == (-30, 30)
    assert np.isfinite(spectrogram_color_limits(np.zeros(3))).all()


@pytest.mark.parametrize(
    "settings", [{"headroom_db": -1}, {"upper_db": float("nan")}, {"lower_db": 10, "upper_db": 0}]
)
def test_invalid_color_limits(settings):
    with pytest.raises(ValueError, match="Spectrogram"):
        spectrogram_color_limits(np.ones(3), **settings)
