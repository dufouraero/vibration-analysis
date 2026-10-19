import json

import numpy as np
import pandas as pd
import pytest

from vibration_analysis.telemetry import Trace, read_electrical, save_operating_conditions


@pytest.mark.parametrize("fit", [True, False])
def test_root_fits_telemetry_union_but_test_keeps_requested_bounds(tmp_path, monkeypatch, fit):
    from matplotlib import dates as mdates
    from matplotlib.figure import Figure

    captured = []
    monkeypatch.setattr(
        Figure, "savefig", lambda fig, *a, **k: captured.append(fig.axes[0].get_xlim())
    )
    rpm = Trace(
        np.array([99.0, 102.0, 104.0, 106.0, 111.0]),
        np.arange(5.0),
        "Engine speed [RPM]",
        "rpm.csv",
        "rpm",
    )
    volts = Trace(
        np.array([101.0, 108.0]), np.array([57.0, 56.0]), "Voltage [V]", "volts.csv", "voltage"
    )
    save_operating_conditions(tmp_path, [rpm, volts], bounds=(100, 110), fit_to_data=fit)
    meta = json.loads((tmp_path / "operating_conditions.json").read_text())
    expected = [101, 108] if fit else [100, 110]
    assert meta["clock_bounds_epoch_s"] == expected
    assert meta["requested_clock_bounds_epoch_s"] == [100, 110]
    np.testing.assert_allclose(
        captured[0],
        mdates.date2num(pd.to_datetime(expected, unit="s", utc=True).to_pydatetime()),
        atol=1e-12,
    )
    assert pd.read_csv(tmp_path / "operating_conditions_rpm.csv")["value"].tolist() == [1, 2, 3]
    assert meta["traces"][1]["samples"] == 2


def test_no_telemetry_in_window_preserves_requested_bounds(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", lambda *a, **k: None)
    rpm = Trace(np.array([1.0, 2.0, 3.0]), np.zeros(3), "Engine speed [RPM]", "rpm.csv", "rpm")
    save_operating_conditions(tmp_path, [rpm], bounds=(100, 110), fit_to_data=True)
    meta = json.loads((tmp_path / "operating_conditions.json").read_text())
    assert meta["clock_bounds_epoch_s"] == [100, 110]
    assert meta["traces"][0]["samples"] == 0


def test_electrical_prefers_voltage_and_can_select_current(tmp_path):
    path = tmp_path / "electrical.csv"
    pd.DataFrame(
        {
            "Time": pd.date_range("2026-10-15 15:00:00", periods=4, freq="2s"),
            "Current [A]": [-2, 1, 10, 20],
            "Voltage [V]": [52, 51, 50, 49],
        }
    ).to_csv(path, index=False)
    voltage = read_electrical(path)
    assert voltage.label == "Voltage [V]"
    assert voltage.values.tolist() == [52, 51, 50, 49]
    assert pd.Timestamp(voltage.seconds[0], unit="s", tz="UTC").hour == 13
    current = read_electrical(path, value_column="Current [A]", quantity="current")
    assert current.values[0] == -2
    with pytest.raises(ValueError, match="not voltage"):
        read_electrical(path, value_column="Current [A]", quantity="voltage")


def test_current_only_export_is_not_labelled_voltage(tmp_path):
    path = tmp_path / "Current and Voltage.csv"
    path.write_text(
        "Time,current rig\n2026-10-15 15:00:00,0\n2026-10-15 15:00:02,10\n2026-10-15 15:00:02,10\n"
    )
    trace = read_electrical(path)
    assert trace.label == "Current [A]"
    assert len(trace.seconds) == 3
    path.write_text(path.read_text().replace("15:00:02,10\n", "15:00:02,11\n", 1))
    trace = read_electrical(path)
    assert trace.values.tolist() == [0, 11, 10]
    assert trace.seconds[-1] == trace.seconds[-2]


def test_operating_conditions_native_times_window_and_optional_rows(tmp_path):
    origin = pd.Timestamp("2026-10-15T13:00:00Z").timestamp()
    rpm = Trace(
        origin + np.arange(0, 12, 2),
        np.arange(6) * 1000.0,
        "Engine speed [RPM]",
        "rpm.csv",
        "raw RPM",
    )
    voltage = Trace(
        origin + np.arange(0, 12, 3), np.arange(4) + 50.0, "Voltage [V]", "voltage.csv", "voltage"
    )
    save_operating_conditions(tmp_path, [rpm, voltage], bounds=(origin + 2, origin + 10), dpi=35)
    meta = json.loads((tmp_path / "operating_conditions.json").read_text())
    assert [row["samples"] for row in meta["traces"]] == [4, 3]
    assert (tmp_path / "operating_conditions.png").is_file()
    electrical = pd.read_csv(tmp_path / "operating_conditions_electrical.csv")
    assert electrical["value"].tolist() == [51, 52, 53]
    assert "+02:00" in electrical["timestamp"].iloc[0]
    save_operating_conditions(tmp_path / "rpm_only", [rpm, None], dpi=35)
    single = json.loads((tmp_path / "rpm_only/operating_conditions.json").read_text())
    assert len(single["traces"]) == 1
    save_operating_conditions(tmp_path / "none", [None], dpi=35)
    assert not (tmp_path / "none").exists()


def test_unknown_electrical_quantity_needs_explicit_selection(tmp_path):
    path = tmp_path / "unknown.csv"
    path.write_text("Time,value\n2026-10-15 15:00:00,50\n2026-10-15 15:00:02,49\n")
    with pytest.raises(ValueError, match="ambiguous"):
        read_electrical(path)
    assert read_electrical(path, quantity="voltage").label == "Voltage [V]"


def test_electrical_only_cli_run_without_vibration_clock(tmp_path):
    from vibration_analysis.cli import main

    source = tmp_path / "vibration.csv"
    t = np.arange(256) / 128
    pd.DataFrame({"time_s": t, "sensor": np.sin(2 * np.pi * 20 * t)}).to_csv(source, index=False)
    electrical = tmp_path / "voltage.csv"
    pd.DataFrame(
        {
            "Time": pd.date_range("2026-10-15 15:00:00", periods=3, freq="s"),
            "Voltage [V]": [52, 51, 50],
        }
    ).to_csv(electrical, index=False)
    output = tmp_path / "output"
    assert (
        main(
            [
                "analyze",
                str(source),
                "--electrical-csv",
                str(electrical),
                "--output",
                str(output),
                "--plots",
                "fft",
                "--dpi",
                "35",
            ]
        )
        == 0
    )
    meta = json.loads((output / "operating_conditions.json").read_text())
    assert len(meta["traces"]) == 1
    assert meta["traces"][0]["label"] == "Voltage [V]"
    assert meta["clock_bounds_epoch_s"] is None


def test_raw_rpm_only_does_not_trigger_detection_or_alignment(tmp_path):
    from vibration_analysis.cli import main

    t = np.arange(256) / 128
    sensor = tmp_path / "sensor.csv"
    pd.DataFrame({"time_s": t, "sensor": np.sin(2 * np.pi * 20 * t)}).to_csv(sensor, index=False)
    raw = tmp_path / "raw_rpm.csv"
    pd.DataFrame(
        {
            "Time": pd.date_range("2026-10-15 15:00:00", periods=4, freq="s"),
            "val_rpm": [0, 6000, 3000, 0],
        }
    ).to_csv(raw, index=False)
    output = tmp_path / "output"
    assert (
        main(
            [
                "analyze",
                str(sensor),
                "--raw-rpm-csv",
                str(raw),
                "--output",
                str(output),
                "--plots",
                "fft",
                "--dpi",
                "35",
            ]
        )
        == 0
    )
    assert not (output / "alignment.json").exists()
    assert not list(output.glob("test_*"))
    meta = json.loads((output / "operating_conditions.json").read_text())
    assert len(meta["traces"]) == 1
    assert meta["traces"][0]["source"] == str(raw.resolve())
    assert pd.read_csv(output / "operating_conditions_rpm.csv")["value"].tolist() == [
        0,
        6000,
        3000,
        0,
    ]


def test_duplicate_voltage_values_are_preserved_in_plot_export(tmp_path):
    path = tmp_path / "voltage.csv"
    path.write_text(
        "Time,voltage\n2026-10-15 17:00:12,57.0\n2026-10-15 17:00:14,56.8\n2026-10-15 17:00:14,56.7\n"
    )
    trace = read_electrical(path)
    output = tmp_path / "output"
    save_operating_conditions(output, [trace], dpi=35)
    data = pd.read_csv(output / "operating_conditions_electrical.csv")
    assert data["value"].tolist() == [57, 56.8, 56.7]
    assert data["timestamp"].iloc[-1] == data["timestamp"].iloc[-2]
    metadata = json.loads((output / "operating_conditions.json").read_text())
    assert metadata["traces"][0]["repeated_timestamp_rows"] == 1
    path.write_text("Time,voltage\n2026-10-15 17:00:14,56.8\n2026-10-15 17:00:12,56.7\n")
    with pytest.raises(ValueError, match="increasing"):
        read_electrical(path)
