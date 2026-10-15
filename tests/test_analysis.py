import numpy as np
import pytest
from scipy import signal

from vibration_analysis.analysis import analyze


@pytest.mark.parametrize("n", [4095, 4096])
def test_peak_scaling_and_phase(n):
    fs = n
    time = np.arange(n) / fs
    x = 3 * np.cos(2 * np.pi * 100 * time + 0.4)
    result = analyze(x, fs, window="boxcar")
    assert result.amplitude[100] == pytest.approx(3, abs=1e-10)
    assert result.phase[100] == pytest.approx(0.4, abs=1e-10)
    assert result.summary["welch_rms"] == pytest.approx(3 / np.sqrt(2))


def test_dc_and_nyquist_are_not_doubled():
    x = 3 + 2 * (-1.0) ** np.arange(1024)
    result = analyze(x, 1024, detrend="none", window="boxcar")
    assert result.amplitude[0] == pytest.approx(3)
    assert result.amplitude[-1] == pytest.approx(2)
    assert result.summary["welch_rms"] == pytest.approx(np.sqrt(13))


@pytest.mark.parametrize("window", ["hann", "flattop", "boxcar"])
@pytest.mark.parametrize("detrend", ["constant", "linear", "none"])
def test_welch_and_frame_psd_match_scipy(window, detrend):
    x = np.random.default_rng(4).normal(size=10000)
    result = analyze(x, 1024, window=window, detrend=detrend, segment_seconds=0.5)
    mode = False if detrend == "none" else detrend
    f, expected = signal.welch(
        x, fs=1024, window=window, nperseg=512, noverlap=256, detrend=mode, scaling="density"
    )
    np.testing.assert_allclose(result.frequency, f)
    np.testing.assert_allclose(result.psd, expected, rtol=1e-12, atol=1e-28)
    _, _, density = signal.spectrogram(
        x, fs=1024, window=window, nperseg=512, noverlap=256, detrend=mode, scaling="density"
    )
    np.testing.assert_allclose(result.window_psd.T, density, rtol=1e-12, atol=1e-28)


def test_band_rms_and_window_correction():
    t = np.arange(8192) / 1024
    x = 2 * np.sin(2 * np.pi * 100 * t) + 3 * np.sin(2 * np.pi * 300 * t)
    result = analyze(x, 1024, bands=[(90, 110), (290, 310)])
    assert result.amplitude[800] == pytest.approx(2)
    assert result.summary["bands"][0]["rms"] == pytest.approx(np.sqrt(2))
    assert result.summary["bands"][1]["rms"] == pytest.approx(3 / np.sqrt(2))


def test_white_noise_parseval():
    x = np.random.default_rng(3).normal(size=65536)
    result = analyze(x, 4096)
    assert result.summary["welch_rms"] == pytest.approx(np.std(x), rel=0.01)


@pytest.mark.parametrize(
    "options", [{"overlap": 1}, {"segment_seconds": 0}, {"sample_rate": 0}, {"bands": [(0, 600)]}]
)
def test_invalid_settings(options):
    kwargs = {"sample_rate": 1024, **options}
    with pytest.raises(ValueError):
        analyze(np.ones(1024), **kwargs)


def test_short_record_and_zero_signal():
    result = analyze(np.zeros(16), 1024)
    assert result.summary["segment_was_shortened"]
    assert result.summary["segment_count"] == 1
    assert result.summary["welch_rms"] == 0


@pytest.mark.parametrize("order", [2, 4, 6])
def test_highpass_suppresses_drift_preserves_fft_and_matches_scipy(order):
    fs = 1024
    t = np.arange(fs * 12) / fs
    x = 4 * np.sin(2 * np.pi * 2 * t) + 2 * np.sin(2 * np.pi * 100 * t)
    raw = analyze(x, fs)
    filtered = analyze(x, fs, psd_highpass_hz=10, psd_filter_order=order)
    sos = signal.butter(order, 10, fs=fs, btype="highpass", output="sos")
    filtered_x = signal.sosfiltfilt(sos, x)
    _, expected = signal.welch(
        filtered_x,
        fs=fs,
        window="hann",
        nperseg=1024,
        noverlap=512,
        detrend="constant",
        scaling="density",
    )
    np.testing.assert_allclose(filtered.psd, expected, rtol=1e-10, atol=1e-24)
    assert filtered.psd[2] < raw.psd[2] * 1e-4
    assert filtered.psd[100] == pytest.approx(raw.psd[100], rel=0.001)
    np.testing.assert_array_equal(filtered.amplitude, raw.amplitude)
    np.testing.assert_array_equal(filtered.window_amplitude, raw.window_amplitude)
    assert filtered.summary["raw_rms"] == raw.summary["raw_rms"]
    assert filtered.summary["welch_rms"] < raw.summary["welch_rms"]
    assert filtered.summary["psd_filter_order_per_pass"] == order
    assert filtered.summary["psd_filter_effective_order"] == 2 * order


def test_independent_spectrogram_matches_scipy():
    fs = 1000
    x = np.random.default_rng(9).normal(size=4000)
    result = analyze(x, fs, segment_seconds=1, spectrogram_seconds=0.1, spectrogram_overlap=0.8)
    f, t, density = signal.spectrogram(
        x, fs=fs, window="hann", nperseg=100, noverlap=80, detrend="constant", scaling="density"
    )
    np.testing.assert_allclose(result.spectrogram_frequency, f)
    np.testing.assert_allclose(result.spectrogram_time, t)
    np.testing.assert_allclose(result.spectrogram_psd.T, density, rtol=1e-12, atol=1e-28)
    assert len(result.spectrogram_time) > len(result.time) * 10
    changed = analyze(x, fs, spectrogram_seconds=0.2)
    np.testing.assert_array_equal(result.psd, changed.psd)


@pytest.mark.parametrize(
    "settings",
    [
        {"psd_highpass_hz": -1},
        {"psd_highpass_hz": 512},
        {"psd_highpass_hz": float("nan")},
        {"spectrogram_overlap": 1},
        {"spectrogram_seconds": 0},
    ],
)
def test_invalid_filter_and_spectrogram_settings(settings):
    with pytest.raises(ValueError):
        analyze(np.ones(1024), 1024, **settings)


def test_filter_rejects_too_short_record():
    with pytest.raises(ValueError, match="too short"):
        analyze(np.ones(4), 1024, psd_highpass_hz=10)
