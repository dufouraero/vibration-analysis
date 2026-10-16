import json

import numpy as np
import pandas as pd
import pytest

from vibration_analysis.cli import main
from vibration_analysis.io import Recording, read_csv, select_active_channels, select_interval


def test_semicolon_milliseconds_decimal_comma(tmp_path):
    path = tmp_path / "sensor.csv"
    path.write_text("metadata\ntime;accel\n0,0;1,0\n1,0;2,0\n2,0;3,0\n3,0;4,0\n")
    rec = read_csv(
        path, time_column="time", time_unit="ms", delimiter=";", decimal=",", skiprows=1, scale=2
    )
    assert rec.sample_rate == pytest.approx(1000)
    np.testing.assert_allclose(rec.channels["accel"], [2, 4, 6, 8])


def test_headerless_without_time(tmp_path):
    path = tmp_path / "sensor.csv"
    path.write_text("1,2\n2,3\n3,4\n4,5\n")
    rec = read_csv(path, header=False, channels=["1"], sample_rate=100)
    np.testing.assert_allclose(rec.time, [0, 0.01, 0.02, 0.03])
    assert list(rec.channels) == ["1"]


def test_datalogger_export_columns(tmp_path):
    path = tmp_path / "logger.csv"
    path.write_text("Sample,Time (s),Sensor (g),\n0,0,1,\n1,0.001,2,\n2,0.002,3,\n3,0.003,4,\n")
    rec = read_csv(path)
    assert list(rec.channels) == ["Sensor (g)"]
    assert rec.sample_rate == pytest.approx(1000)


def test_quantized_timestamps_do_not_bias_sample_rate(tmp_path):
    fs = 1706.6666666667
    path = tmp_path / "rounded.csv"
    t = np.round(np.arange(20000) / fs, 6)
    pd.DataFrame({"time_s": t, "accel": np.ones(len(t))}).to_csv(path, index=False)
    assert read_csv(path).sample_rate == pytest.approx(fs, abs=0.001)


def test_active_channel_selection_ignores_unused_noise_and_single_burst():
    fs = 128
    t = np.arange(fs * 120) / fs
    rng = np.random.default_rng(2)
    active = 4 * np.sin(2 * np.pi * 20 * t)
    unused = 0.02 * rng.normal(size=len(t))
    unused[:8] = 50  # An isolated startup/handling burst must not dominate selection.
    other_sensor = np.sin(2 * np.pi * 30 * t)
    rec = Recording(
        t, {"unused": unused, "connected": active, "second_connected": other_sensor}, fs, 0
    )
    selected, report = select_active_channels(rec)
    assert list(selected.channels) == ["connected", "second_connected"]
    assert report["excluded"] == ["unused"]


def test_zero_activity_does_not_arbitrarily_discard_channels():
    t = np.arange(1280) / 128
    rec = Recording(t, {"one": np.zeros(len(t)), "two": np.zeros(len(t))}, 128, 0)
    selected, report = select_active_channels(rec)
    assert len(selected.channels) == 2
    assert report["excluded"] == []


def test_datetime(tmp_path):
    path = tmp_path / "sensor.csv"
    pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=8, freq="1ms"), "accel": np.ones(8)}
    ).to_csv(path, index=False)
    rec = read_csv(path, time_unit="datetime")
    assert rec.sample_rate == pytest.approx(1000)


@pytest.mark.parametrize(
    "times", [[0, 0.01, 0.02, 0.04], [0, 0.01, 0.01, 0.02], [0, 0.02, 0.01, 0.03]]
)
def test_reject_bad_time(tmp_path, times):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time": times, "accel": np.ones(4)}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_csv(path)


def test_reject_missing_channel_and_rate_mismatch(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("time,accel\n0,1\n.01,\n.02,3\n.03,4\n")
    with pytest.raises(ValueError, match="missing"):
        read_csv(path, channels=["accel"])
    with pytest.raises(ValueError, match="disagrees"):
        read_csv(path, sample_rate=200, time_column="time")


def test_demo_and_all_exports(tmp_path):
    source = tmp_path / "demo.csv"
    output = tmp_path / "results"
    assert main(["demo", "--output", str(source), "--duration", "2"]) == 0
    assert (
        main(
            [
                "analyze",
                str(source),
                "--output",
                str(output),
                "--channels",
                "accel_steady_g",
                "--band",
                "90:110",
                "--dpi",
                "50",
            ]
        )
        == 0
    )
    folder = output / "01_accel_steady_g"
    expected = {
        "fft.png",
        "psd_linear_x.png",
        "psd_log_x.png",
        "spectrogram.png",
        "waterfall.png",
        "time.png",
        "fft.csv",
        "psd.csv",
        "time_frequency.npz",
        "summary.json",
    }
    assert {p.name for p in folder.iterdir()} == expected
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["bands"][0]["rms"] == pytest.approx(0.75 / np.sqrt(2), rel=0.05)
    data = np.load(folder / "time_frequency.npz")
    assert data["psd"].shape == (7, 1025)
    assert data["spectrogram_psd"].shape[0] > data["psd"].shape[0]


def test_cli_filtered_spectrogram_exports_settings(tmp_path):
    source = tmp_path / "demo.csv"
    assert main(["demo", "--output", str(source), "--duration", "2"]) == 0
    output = tmp_path / "filtered"
    assert (
        main(
            [
                "analyze",
                str(source),
                "--channels",
                "accel_steady_g",
                "--output",
                str(output),
                "--psd-highpass-hz",
                "10",
                "--plots",
                "psd",
                "spectrogram",
                "--spectrogram-interpolation",
                "nearest",
                "--dpi",
                "40",
            ]
        )
        == 0
    )
    summary = json.loads((output / "01_accel_steady_g/summary.json").read_text())
    assert summary["psd_highpass_hz"] == 10
    assert summary["spectrogram_display"]["interpolation"] == "nearest"


def test_zero_short_record_plots(tmp_path):
    path = tmp_path / "zero.csv"
    pd.DataFrame({"a": np.zeros(16)}).to_csv(path, index=False)
    assert (
        main(
            [
                "analyze",
                str(path),
                "--sample-rate",
                "1024",
                "--output",
                str(tmp_path / "out"),
                "--dpi",
                "40",
            ]
        )
        == 0
    )


def test_cli_invalid_band_returns_error(tmp_path):
    assert main(["analyze", str(tmp_path / "unused.csv"), "--band", "bad"]) == 2


def test_interval_trim_preserves_source_time_and_all_channels():
    t = np.arange(2000) / 100
    recording = Recording(t, {"x": t * 2, "y": t * 3}, 100, 0)
    trimmed, info = select_interval(
        recording, start_seconds=2, end_seconds=12, trim_start_seconds=1, trim_end_seconds=2
    )
    assert len(trimmed.time) == 700
    assert trimmed.time[0] == 3
    assert trimmed.time[-1] == 9.99
    np.testing.assert_array_equal(trimmed.channels["y"], recording.channels["y"][300:1000])
    assert info["retained_start_sample"] == 300
    assert info["retained_end_sample_exclusive"] == 1000
    assert info["retained_duration_seconds"] == 7


@pytest.mark.parametrize(
    "settings",
    [
        {"trim_start_seconds": -1},
        {"trim_end_seconds": float("nan")},
        {"trim_start_seconds": 5, "trim_end_seconds": 5},
        {"start_seconds": 5, "end_seconds": 4},
        {"end_seconds": 11},
    ],
)
def test_invalid_interval_and_trim(settings):
    recording = Recording(np.arange(1000) / 100, {"x": np.ones(1000)}, 100, None)
    with pytest.raises(ValueError):
        select_interval(recording, **settings)


def test_cli_trim_controls_statistics_and_source_time_exports(tmp_path):
    fs = 1000
    time = np.arange(4000) / fs
    values = np.sin(2 * np.pi * 50 * time)
    values[:1000] += 100
    source = tmp_path / "trim.csv"
    pd.DataFrame({"time_s": time, "signal": values}).to_csv(source, index=False)
    output = tmp_path / "output"
    assert (
        main(
            [
                "analyze",
                str(source),
                "--output",
                str(output),
                "--trim-start-seconds",
                "1",
                "--trim-end-seconds",
                "1",
                "--plots",
                "psd",
                "--dpi",
                "40",
            ]
        )
        == 0
    )
    summary = json.loads((output / "01_signal/summary.json").read_text())
    assert summary["samples"] == 2000
    assert summary["raw_rms"] == pytest.approx(1 / np.sqrt(2))
    assert summary["analysis_interval"]["retained_duration_seconds"] == pytest.approx(2)
    spectra = np.load(output / "01_signal/time_frequency.npz")
    assert spectra["time_s"][0] == pytest.approx(1.25)
    assert spectra["spectrogram_time_s"][0] == pytest.approx(1.05)


def test_microsecond_rounding_at_high_sample_rate(tmp_path):
    fs = 17066.6666666667
    time = np.round(np.arange(100000) / fs, 6)
    path = tmp_path / "high_rate.csv"
    pd.DataFrame({"time_s": time, "accel": np.sin(2 * np.pi * 100 * time)}).to_csv(
        path, index=False
    )
    rec = read_csv(path)
    assert rec.timing_jitter > 0.01
    assert rec.sample_rate == pytest.approx(fs, rel=1e-6)
    assert len(rec.time) == len(time)
    np.testing.assert_array_equal(rec.time, time)
    # A lost sample must not be explained away as decimal rounding.
    pd.DataFrame({"time_s": np.delete(time, 50000), "accel": np.ones(len(time) - 1)}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="Uneven"):
        read_csv(path)


def test_rounded_timestamps_with_clock_drift_still_fail(tmp_path):
    fs = 17066.6666666667
    time = np.arange(100000) / fs
    time = np.round(time + 0.0001 * np.sin(np.arange(len(time)) / 1000), 6)
    path = tmp_path / "drift.csv"
    pd.DataFrame({"time_s": time, "accel": np.ones(len(time))}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Uneven"):
        read_csv(path)
