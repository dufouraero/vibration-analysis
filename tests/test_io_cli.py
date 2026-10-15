import json

import numpy as np
import pandas as pd
import pytest

from vibration_analysis.cli import main
from vibration_analysis.io import read_csv


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
    assert data["psd"].shape == (3, 2049)
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
