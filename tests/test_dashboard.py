"""Numerical and persistence checks for interactive region comparisons."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from vibration_analysis.dashboard import DEFAULTS, DISPLAY, axis_bounds, create_app, demo
from vibration_analysis.dashboard_analysis import Settings, _spectrogram, compute_region
from vibration_analysis.dashboard_data import cache_recording, display_indices
from vibration_analysis.dashboard_service import DashboardService, Dataset, validate_region
from vibration_analysis.dashboard_views import comparison_figures, export_report, overview
from vibration_analysis.io import Recording
from vibration_analysis.telemetry import Trace


@pytest.fixture
def service(tmp_path):
    instance = DashboardService(tmp_path / "cache")
    yield instance
    instance.close()


def synthetic(fs=2000, duration=4):
    t = np.arange(round(fs * duration)) / fs
    x = 2 * np.sin(2 * np.pi * 100 * t) + 0.5 * np.cos(2 * np.pi * 300 * t) + 3
    return Dataset(
        "d",
        "Test",
        Recording(t, {"g": x}, fs, 0),
        {"unit": "g", "input": "test.csv"},
        "g",
        None,
        None,
        None,
        None,
        0,
        [],
    )


def region(dataset, start=0, end=None):
    return validate_region(dataset, {"start": start, "end": end or dataset.duration})


@pytest.mark.parametrize("fs", [1706.6666667, 17066.666667])
def test_rounded_csv_cache_matches_original_and_reuses_cache(tmp_path, monkeypatch, fs):
    path = tmp_path / "headerless.csv"
    t = np.arange(12000) / fs
    x = np.sin(2 * np.pi * 120 * t)
    np.savetxt(path, np.column_stack([t, x, x * 0.01]), delimiter=",", fmt="%.6f")
    config = {"input": str(path), "scale": 2}
    cached = cache_recording(config, tmp_path / "cache")
    assert isinstance(cached.recording.time, np.memmap)
    np.testing.assert_allclose(cached.recording.channels["1"], np.round(x, 6) * 2, atol=1e-14)
    assert cached.recording.sample_rate == pytest.approx(fs, rel=2e-6)
    assert len(cached.recording.channels) == 2
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Cache reread original CSV"))
    reopened = cache_recording(config, tmp_path / "cache")
    assert reopened.directory == cached.directory


def test_csv_formats_and_channel_named_time(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text(
        "metadata\nt;time\n"
        + "".join(f"{i / 100:.2f};{i / 10:.1f}\n".replace(".", ",") for i in range(100))
    )
    result = cache_recording(
        {
            "input": str(path),
            "skiprows": 1,
            "delimiter": ";",
            "decimal": ",",
            "time_column": "t",
            "channels": ["time"],
        },
        tmp_path / "cache",
    )
    np.testing.assert_allclose(result.recording.time, np.arange(100) / 100)
    np.testing.assert_allclose(result.recording.channels["time"], np.arange(100) / 10)


def test_headerless_daq_trailing_empty_column_is_ignored(tmp_path):
    path = tmp_path / "daq.csv"
    path.write_text("".join(f"{i / 1000:.6f},{np.sin(i):.6f},\n" for i in range(1000)))
    result = cache_recording({"input": str(path)}, tmp_path / "cache")
    assert list(result.recording.channels) == ["1"]
    assert len(result.recording.time) == 1000


def test_dropped_sample_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    t = np.delete(np.arange(1000) / 1000, 500)
    pd.DataFrame({"time": t, "g": np.ones(len(t))}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="[Uu]neven sampling"):
        cache_recording({"input": str(path)}, tmp_path / "cache")
    pd.DataFrame({"time": np.arange(100) / 100, "g": [1.0] * 99 + [np.inf]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError):
        cache_recording({"input": str(path)}, tmp_path / "cache")


def test_minmax_display_preserves_short_peaks():
    x = np.zeros(100000)
    x[12345], x[98765] = 50, -49
    indices = display_indices(x, budget=800)
    assert {12345, 98765}.issubset(indices)
    assert len(indices) <= 800
    assert np.all(np.diff(indices) > 0)


def test_shared_row_zoom_bounds():
    assert axis_bounds({"xaxis3.range[0]": 12, "xaxis3.range[1]": 15}) == (12, 15)
    assert axis_bounds({"xaxis2.range": [20, 40]}) == (20, 40)
    assert axis_bounds({"xaxis.autorange": True}) is None


def test_telemetry_decimation_does_not_create_false_gaps():
    dataset = synthetic(duration=180)
    times = np.arange(18000) / 100
    dataset.origin = 0
    dataset.raw_rpm = Trace(times, np.sin(times) + 4500, "Raw RPM", "rpm.csv", "rpm")
    figure = overview(dataset, "g", [])
    assert len(figure.data[1].x) <= 6000
    assert np.isfinite(figure.data[1].y).all()
    dataset.raw_rpm.seconds[9000:] += 5
    figure = overview(dataset, "g", [])
    assert np.isnan(figure.data[1].y).sum() == 1


@pytest.mark.parametrize("highpass", [0, 10])
@pytest.mark.parametrize("fs", [1700, 17000])
def test_region_welch_matches_scipy_and_fft_amplitude(fs, highpass):
    dataset = synthetic(fs)
    before = dataset.recording.channels["g"].copy()
    settings = Settings(highpass_hz=highpass)
    result = compute_region(dataset, region(dataset, 0.5, 3.5), settings)
    x = before[round(fs * 0.5) : round(fs * 3.5)]
    filtered = (
        signal.sosfiltfilt(signal.butter(4, highpass, fs=fs, btype="highpass", output="sos"), x)
        if highpass
        else x
    )
    f, reference = signal.welch(
        filtered,
        fs=fs,
        window="hann",
        nperseg=round(fs * 0.5),
        noverlap=round(fs * 0.25),
        detrend="constant",
    )
    np.testing.assert_allclose(result.frequency, f)
    np.testing.assert_allclose(result.psd, reference, atol=1e-14, rtol=1e-9)
    assert result.fft_amplitude[np.argmin(abs(f - 100))] == pytest.approx(2, abs=1e-10)
    assert result.summary["raw_rms"] == pytest.approx(np.sqrt(np.mean(x**2)))
    assert result.summary["welch_rms"] == pytest.approx(np.sqrt(reference.sum() * (f[1] - f[0])))
    assert result.summary["psd_bin_hz"] == 2
    np.testing.assert_array_equal(before, dataset.recording.channels["g"])


def test_full_fft_and_changed_interval_recalculate(service):
    dataset = synthetic()
    dataset.recording.channels["g"][4000:] *= 3
    service.datasets[dataset.id] = dataset
    rows = [region(dataset, 0, 2), region(dataset, 2, 4)]
    bundle = service.compare(rows, {"highpass_hz": 0, "fft_mode": "full"})
    a, b = service.bundles[bundle]
    assert a.summary["samples"] == 4000
    assert a.summary["fft_bin_hz"] == 0.5
    assert b.summary["raw_rms"] == pytest.approx(a.summary["raw_rms"] * 3)
    assert max(b.fft_amplitude) == pytest.approx(max(a.fft_amplitude) * 3)
    assert service.bundles[service.compare(rows, a.settings)][0] is a
    with pytest.raises(ValueError, match="four original"):
        validate_region(dataset, {"start": 0, "end": 0.001})


def test_spectrogram_pooling_averages_linear_power():
    dataset = synthetic(fs=2000, duration=2)
    x = dataset.recording.channels["g"]
    f, t, density, frames, _ = _spectrogram(
        x, 2000, 200, 0.8, "hann", "constant", lambda _: None, max_times=7, max_frequencies=11
    )
    rf, rt, reference = signal.spectrogram(
        x, fs=2000, window="hann", nperseg=200, noverlap=160, detrend="constant", scaling="density"
    )
    assert frames == len(rt)
    assert density.shape == (7, 11)
    fg = np.arange(len(rf)) * len(f) // len(rf)
    tg = np.arange(len(rt)) * len(t) // len(rt)
    for i in range(7):
        for j in range(11):
            assert density[i, j] == pytest.approx(reference[np.ix_(fg == j, tg == i)].mean())


def test_demo_suggestions_session_roundtrip_and_source_protection(service, tmp_path):
    identifier = demo(service, lambda _: None)
    dataset = service.datasets[identifier]
    rows = service.suggest(identifier, DEFAULTS)
    assert len(rows) == 2
    assert (
        dataset.mrpm is not None and dataset.raw_rpm is not None and dataset.electrical is not None
    )
    assert {
        round(
            dataset.mrpm[1][(dataset.mrpm[0] >= r["start"]) & (dataset.mrpm[0] < r["end"])].mean()
        )
        for r in rows
    } == {4500, 5400}
    payload = service.session(rows, {}, DISPLAY)
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload))
    restored = service.restore(path)
    assert len(restored["regions"]) == 2
    assert restored["dataset"] != identifier
    pd.DataFrame({"time": [0, 1, 2, 3], "g": [0, 0, 0, 0]}).to_csv(
        dataset.config["input"], index=False
    )
    with pytest.raises(ValueError, match="changed since"):
        service.restore(path)
    service.datasets[identifier] = replace(dataset, mrpm=None)
    with pytest.raises(ValueError, match="Raw RPM is only"):
        service.suggest(identifier, DEFAULTS)


def test_common_band_comparison_and_portable_export(service, tmp_path):
    datasets = [synthetic(1700), synthetic(17000)]
    datasets[1].id = "d2"
    rows = []
    for dataset in datasets:
        service.datasets[dataset.id] = dataset
        rows.append(region(dataset))
    bundle = service.compare(rows, {"highpass_hz": 0})
    results = service.bundles[bundle]
    display = {**DISPLAY, "fmin": 10, "fmax": 850, "fft_layout": "stacked", "psd_axis": "log"}
    figures, stats, note = comparison_figures(service, results, display)
    assert stats[0]["band_rms"] == stats[1]["band_rms"]
    assert figures["psd"].layout.xaxis.type == "log"
    assert figures["fft"].layout.yaxis2.range == figures["fft"].layout.yaxis.range
    assert len(figures["fft"].data) == 2
    assert "original samples" in note
    path = export_report(service, results, figures, stats, {}, tmp_path)
    document = path.read_text(encoding="utf-8")
    assert "plotly.js" in document
    assert '<script src="https://' not in document
    assert (path.parent / "statistics.csv").is_file()
    with np.load(path.parent / "region_02_spectra.npz") as numerical:
        assert numerical["frequency_hz"][-1] == 8500
        np.testing.assert_allclose(numerical["psd"], results[1].psd)
    assert overview(datasets[0], "g", rows).layout.dragmode == "select"


def test_app_serves_local_assets_and_valid_callback_graph(service):
    app = create_app(service=service)
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200 and b"Vibration workspace" in layout.data
    assert client.get("/_dash-dependencies").status_code == 200
    assert client.get("/assets/wave.svg").status_code == 200
    assert client.get("/assets/style.css").status_code == 200


@pytest.mark.parametrize(
    "values",
    [
        {"overlap": 1},
        {"highpass_hz": -1},
        {"segment_seconds": 0},
        {"filter_order": 2.5},
        {"fft_mode": "other"},
    ],
)
def test_invalid_analysis_parameters(values):
    with pytest.raises(ValueError):
        Settings.from_dict(values)
