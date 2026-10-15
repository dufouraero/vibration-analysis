import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from vibration_analysis.batch import run_rpm_analysis
from vibration_analysis.cli import main
from vibration_analysis.io import Recording
from vibration_analysis.operating_points import (
    detect_runs_and_holds,
    estimate_offset,
    read_rpm,
    source_origin,
)


def test_rpm_timestamp_units_and_timezone(tmp_path):
    path = tmp_path / "rpm.csv"
    path.write_text(
        "Time,RPM\n2026-10-14T16:00:00+02:00,0\n2026-10-14T16:00:05+02:00,3000\n2026-10-14T16:00:10+02:00,3000\n"
    )
    time, rpm, meta = read_rpm(path)
    np.testing.assert_allclose(np.diff(time), 5)
    assert time[0] == pd.Timestamp("2026-10-14T14:00:00Z").timestamp()
    assert meta["median_interval_seconds"] == 5
    assert rpm[1] == 3000


def test_filename_origin():
    value = source_origin("Job1_2026-10-14T14-58-37-482.csv")
    assert value == pd.Timestamp("2026-10-14T14:58:37.482Z").timestamp()


def test_batch_refuses_existing_results(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing results")
    with pytest.raises(ValueError, match="preserve previous"):
        run_rpm_analysis(Namespace(output=tmp_path), None, [])
    assert sentinel.read_text() == "existing results"


def test_ramp_then_plateau_is_not_split_into_rpm_bins():
    time = np.arange(0, 305, 5, dtype=float)
    rpm = np.where(
        time < 20,
        0,
        np.where(time < 100, 2000, np.where(time < 200, 2000 + (time - 100) * 20, 4000)),
    )
    rpm[time >= 290] = 0
    runs = detect_runs_and_holds(time, rpm)
    assert len(runs) == 1
    assert len(runs[0]["holds"]) == 2
    first, second = runs[0]["holds"]
    assert first["end"] <= 105
    assert second["start"] >= 190
    assert second["end"] - second["start"] >= 60


def test_telemetry_gap_splits_tests():
    time = np.r_[np.arange(0, 70, 5), np.arange(150, 230, 5)].astype(float)
    runs = detect_runs_and_holds(time, np.full(len(time), 3000))
    assert len(runs) == 2


def test_auto_alignment_recovers_known_delay():
    fs = 128
    t = np.arange(360 * fs) / fs
    active = ((t >= 60) & (t < 150)) | ((t >= 210) & (t < 300))
    x = np.where(active, 3, 0.001) * np.sin(2 * np.pi * 20 * t)
    recording = Recording(t, {"sensor": x}, fs, 0)
    reference = np.arange(0, 335, 5, dtype=float)
    rpm = np.where(
        ((reference >= 30) & (reference < 120)) | ((reference >= 180) & (reference < 270)), 4000, 0
    )
    offset, info = estimate_offset(recording, reference, rpm, 0, 500)
    assert offset == pytest.approx(30, abs=3)
    assert info["balanced_agreement"] > 0.95
    with pytest.raises(ValueError, match="both running and downtime"):
        estimate_offset(recording, reference, np.full(len(rpm), 4000), 0, 500)


def test_batch_folder_structure_trim_and_comparison_exports(tmp_path, capsys):
    fs = 128
    time = np.arange(180 * fs) / fs
    values = np.sin(2 * np.pi * 20 * time)
    source = tmp_path / "vibration.csv"
    pd.DataFrame({"time_s": time, "accel": values}).to_csv(source, index=False)
    rt = np.arange(0, 180, 5)
    speed = np.where(
        rt < 20, 0, np.where(rt < 60, 2000, np.where(rt < 90, 2000 + (rt - 60) * 100 / 3, 3000))
    )
    speed[rt >= 155] = 0
    origin = pd.Timestamp("2026-10-14T16:00:00Z")
    rpm_path = tmp_path / "rpm.csv"
    pd.DataFrame({"Time": origin + pd.to_timedelta(rt, unit="s"), "RPM": speed}).to_csv(
        rpm_path, index=False
    )
    output = tmp_path / "out"
    assert (
        main(
            [
                "analyze",
                str(source),
                "--rpm-csv",
                str(rpm_path),
                "--vibration-start",
                str(origin),
                "--rpm-alignment",
                "Manual",
                "--output",
                str(output),
                "--trim-start-seconds",
                "2",
                "--trim-end-seconds",
                "2",
                "--rpm-stability-seconds",
                "10",
                "--minimum-hold-seconds",
                "20",
                "--plots",
                "fft",
                "psd",
                "--dpi",
                "35",
                "--gui-progress",
            ]
        )
        == 0
    )
    test = output / "test_01"
    assert (test / "whole_test/01_accel/fft.png").is_file()
    holds = list((test / "operating_points").iterdir())
    assert len(holds) == 2
    summary = json.loads((holds[0] / "01_accel/summary.json").read_text())
    assert summary["analysis_interval"]["trim_start_seconds"] == 2
    assert summary["operating_context"]["mean_rpm"] > 0
    assert (test / "comparisons/01_accel/fft_rpm_waterfall.png").is_file()
    assert (test / "comparisons/01_accel/psd_comparison.png").is_file()
    data = np.load(test / "comparisons/01_accel/operating_point_spectra.npz")
    assert data["mean_window_peak_amplitude"].shape[0] == 2
    assert (output / "alignment.json").is_file()
    assert (output / "detection_overview.png").is_file()
    from vibration_analysis.progress import parse

    events = [event for line in capsys.readouterr().out.splitlines() if (event := parse(line))]
    assert any("Test 1/1" in event["message"] for event in events)
    assert any("Point 2/2" in event["message"] for event in events)
    assert events[-1]["complete"]
