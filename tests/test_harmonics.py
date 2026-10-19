import numpy as np
import pytest

from vibration_analysis.cli import parser
from vibration_analysis.gui import build_arguments
from vibration_analysis.harmonics import parse_orders, track_orders


def test_rpm_guides_seventh_order_instead_of_unrelated_larger_peak():
    f = np.arange(0, 1002, 2.0)
    reports = []
    for expected, height in [(560, 2), (630, 5), (700, 30)]:
        a = np.exp(-0.5 * ((f - expected) / 2) ** 2) * height
        a += 100 * np.exp(-0.5 * ((f - 740) / 2) ** 2)
        reports.append({"frequency": f, "mean_amplitude": a})
    table = track_orders(reports, [4800, 5400, 6000], [7], 5)
    np.testing.assert_allclose(table.peak_hz, [560, 630, 700])
    np.testing.assert_allclose(table.peak_amplitude, [2, 5, 30])


def test_missing_out_of_band_and_ambiguous_candidates_not_invented():
    report = {"frequency": np.arange(0, 102, 2.0), "mean_amplitude": np.ones(51)}
    table = track_orders([report], [6000], [0.5, 2], 5)
    assert list(table.status) == ["no_local_peak", "outside_bandwidth"]
    assert table.peak_amplitude.isna().all()
    assert track_orders([report], [600], [1, 2], 6).status.iloc[0] == "ambiguous_search_width"


def test_fractional_orders_and_fft_bin_tolerance():
    f = np.arange(0, 102, 2.0)
    a = np.zeros(len(f))
    a[13] = 4
    table = track_orders([{"frequency": f, "mean_amplitude": a}], [3000], [0.5], 0.1)
    assert table.peak_hz.iloc[0] == 26
    assert table.detuning_hz.iloc[0] == 1
    assert table.search_half_width_hz.iloc[0] == 2
    assert parse_orders("7, 0.5, 1,7") == [0.5, 1, 7]
    assert parse_orders("") == []


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "seven", "1,,2"])
def test_invalid_order_settings(value):
    with pytest.raises(ValueError):
        parse_orders(value)


def test_gui_passes_empty_orders_to_disable_and_fractional_orders(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("time,x\n0,1\n")
    settings = {
        "input": str(path),
        "output": str(tmp_path),
        "harmonic_orders": "",
        "harmonic_tolerance_hz": "3",
    }
    args = parser().parse_args(build_arguments(settings, ["fft"]))
    assert args.harmonic_orders == ""
    assert args.harmonic_tolerance_hz == 3
