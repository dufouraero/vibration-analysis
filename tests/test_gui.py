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
        "spectrogram_colormap": "viridis",
        "spectrogram_interpolation": "bilinear",
        "waterfall_max_frames": "0",
        "psd_filter_order": "6",
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
    assert args.spectrogram_seconds == 0.1
    assert args.spectrogram_interpolation == "bilinear"
    assert args.waterfall_max_frames == 0
    assert args.psd_filter_order == 6


def test_form_requires_csv_and_plots(tmp_path):
    with pytest.raises(ValueError, match="CSV"):
        build_arguments({"input": "", "output": str(tmp_path)}, ["fft"])
    csv = tmp_path / "sample.csv"
    csv.touch()
    with pytest.raises(ValueError, match="plot"):
        build_arguments({"input": str(csv), "output": str(tmp_path)}, [])
