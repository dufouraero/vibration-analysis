import pytest

from vibration_analysis.cli import parser
from vibration_analysis.gui import build_arguments


def test_form_maps_to_cli_without_shell(tmp_path):
    csv = tmp_path / "recording with spaces.csv"
    csv.write_text("time,x,y\n0,1,2\n")
    values = {
        "input": str(csv),
        "output": str(tmp_path / "results"),
        "time_column": "time",
        "channels": "x, y",
        "time_unit": "ms",
        "delimiter": "Semicolon",
        "decimal": ",",
        "no_header": "1",
        "bands": "100:500, 500:1000",
        "scale": "9.80665",
        "psd_highpass_hz": "10",
        "spectrogram_seconds": "0.1",
        "spectrogram_overlap": "0.8",
        "spectrogram_range_db": "60",
        "spectrogram_headroom_db": "8",
        "spectrogram_lower_db": "-50",
        "spectrogram_upper_db": "10",
        "spectrogram_rpm": "0",
        "spectrogram_colormap": "viridis",
        "spectrogram_interpolation": "bilinear",
        "waterfall_max_frames": "0",
        "psd_filter_order": "6",
        "start_seconds": "100",
        "end_seconds": "160",
        "trim_start_seconds": "5",
        "trim_end_seconds": "5",
    }
    args = parser().parse_args(build_arguments(values, ["fft", "psd"]))
    assert args.input == csv
    assert args.channels == ["x", "y"]
    assert args.band == ["100:500", "500:1000"]
    assert args.delimiter == ";"
    assert args.plots == ["fft", "psd"]
    assert args.no_header
    assert args.scale == 9.80665
    assert args.psd_highpass_hz == 10
    assert args.spectrogram_headroom_db == 8
    assert args.spectrogram_lower_db == -50 and args.spectrogram_upper_db == 10
    assert args.no_spectrogram_rpm
    assert args.spectrogram_seconds == 0.1
    assert args.spectrogram_interpolation == "bilinear"
    assert args.waterfall_max_frames == 0
    assert args.psd_filter_order == 6
    assert args.start_seconds == 100 and args.end_seconds == 160
    assert args.trim_start_seconds == 5 and args.trim_end_seconds == 5
    assert args.segment_seconds == 0.5


def test_form_requires_csv_and_plots(tmp_path):
    with pytest.raises(ValueError, match="CSV"):
        build_arguments({"input": "", "output": str(tmp_path)}, ["fft"])
    csv = tmp_path / "sample.csv"
    csv.touch()
    with pytest.raises(ValueError, match="plot"):
        build_arguments({"input": str(csv), "output": str(tmp_path)}, [])


def test_rpm_form_controls(tmp_path):
    csv = tmp_path / "data.csv"
    csv.touch()
    rpm = tmp_path / "rpm.csv"
    rpm.touch()
    args = parser().parse_args(
        build_arguments(
            {
                "input": str(csv),
                "output": str(tmp_path),
                "rpm_csv": str(rpm),
                "rpm_alignment": "Manual",
                "rpm_offset_seconds": "30",
                "vibration_start": "2026-10-14T14:58:37.482",
                "minimum_hold_seconds": "45",
            },
            ["fft"],
        )
    )
    assert args.rpm_csv == rpm
    assert args.rpm_offset_seconds == 30
    assert args.minimum_hold_seconds == 45


def test_gui_default_active_selection_and_all_override(tmp_path):
    csv = tmp_path / "sensor.csv"
    csv.touch()
    settings = {"input": str(csv), "output": str(tmp_path)}
    assert parser().parse_args(build_arguments(settings, ["fft"])).active_channels
    settings["channel_mode"] = "All channels"
    assert not parser().parse_args(build_arguments(settings, ["fft"])).active_channels
    settings.update(channel_mode="Auto activity", channels="quiet sensor")
    args = parser().parse_args(build_arguments(settings, ["fft"]))
    assert args.channels == ["quiet sensor"]
    assert not args.active_channels


def test_grafana_window_form_and_missing_rpm_validation(tmp_path):
    csv = tmp_path / "vibration.csv"
    csv.touch()
    settings = {
        "input": str(csv),
        "output": str(tmp_path),
        "grafana_start": "16:01:00",
        "grafana_end": "16:09:30",
        "grafana_timezone": "Europe/Zurich",
    }
    with pytest.raises(ValueError, match="RPM CSV"):
        build_arguments(settings, ["fft"])
    settings["rpm_csv"] = str(csv)
    args = parser().parse_args(build_arguments(settings, ["fft"]))
    assert args.grafana_start == "16:01:00"
    assert args.grafana_end == "16:09:30"
    assert args.grafana_timezone == "Europe/Zurich"


def test_optional_electrical_csv_and_waterfall_limit(tmp_path):
    source = tmp_path / "sensor.csv"
    source.touch()
    electrical = tmp_path / "voltage.csv"
    electrical.touch()
    args = parser().parse_args(
        build_arguments(
            {
                "input": str(source),
                "output": str(tmp_path / "results"),
                "electrical_csv": str(electrical),
                "electrical_column": "Voltage [V]",
                "electrical_quantity": "voltage",
                "waterfall2d_fmax": "500",
                "fmax": "2000",
                "vibration_start": "2026-10-15 15:00:00",
            },
            ["waterfall"],
        )
    )
    assert args.electrical_csv == electrical
    assert args.electrical_column == "Voltage [V]"
    assert args.electrical_quantity == "voltage"
    assert args.waterfall2d_fmax == 500 and args.fmax == 2000
    assert args.vibration_start == "2026-10-15 15:00:00"


def test_mrpm_detection_and_raw_rpm_plot_are_separate_inputs(tmp_path):
    source, mrpm, raw = (tmp_path / name for name in ["sensor.csv", "mrpm.csv", "raw_rpm.csv"])
    for file in (source, mrpm, raw):
        file.touch()
    args = parser().parse_args(
        build_arguments(
            {
                "input": str(source),
                "output": str(tmp_path),
                "rpm_csv": str(mrpm),
                "raw_rpm_csv": str(raw),
                "raw_rpm_column": "val_rpm",
            },
            ["fft"],
        )
    )
    assert args.rpm_csv == mrpm
    assert args.raw_rpm_csv == raw
    assert args.raw_rpm_column == "val_rpm"
