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


def test_paired_frequency_views_axes_rms_and_spectrogram_colours(tmp_path, monkeypatch):
    from pathlib import Path

    from matplotlib.figure import Figure

    from vibration_analysis.analysis import analyze
    from vibration_analysis.plots import psd_band_rms, save_plots

    time = np.arange(512) / 256
    values = np.sin(2 * np.pi * 20 * time) + 3 * np.sin(2 * np.pi * 80 * time)
    result = analyze(values, 256)
    captured = {}

    def capture(fig, filename, **kwargs):
        axis = fig.axes[0]
        captured[Path(filename).name] = {
            "x": axis.get_xlim(),
            "y": axis.get_ylim(),
            "title": axis.get_title(),
            "colour": axis.images[0].get_clim() if axis.images else None,
        }

    monkeypatch.setattr(Figure, "savefig", capture)
    save_plots(
        result,
        time,
        values,
        tmp_path,
        unit="g",
        title="two tones",
        plots={"time", "fft", "psd", "waterfall", "spectrogram"},
        fmin=10,
        fmax=40,
        dpi=35,
    )
    assert len(captured) == 11
    for name in ["fft", "psd_linear_x", "waterfall"]:
        assert captured[name + "_broadband.png"]["x"] == (0, 128)
        assert captured[name + "_trimmed.png"]["x"] == (10, 40)
    assert captured["psd_log_x_broadband.png"]["x"] == (2, 128)
    assert captured["psd_log_x_trimmed.png"]["x"] == (10, 40)
    assert captured["spectrogram_broadband.png"]["y"] == (0, 128)
    assert captured["spectrogram_trimmed.png"]["y"] == (10, 40)
    assert (
        captured["spectrogram_broadband.png"]["colour"]
        == captured["spectrogram_trimmed.png"]["colour"]
    )
    assert "Full-band RMS = 2.236 g" in captured["psd_linear_x_trimmed.png"]["title"]
    assert "Band RMS (10–40 Hz) = 0.7071 g" in captured["psd_linear_x_trimmed.png"]["title"]
    assert psd_band_rms(result, 10, 40) == pytest.approx(1 / np.sqrt(2))
