"""Timestamped operating-condition plots, preserving native logger samples."""

import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass
class Trace:
    seconds: np.ndarray
    values: np.ndarray
    label: str
    source: str
    column: str


def read_electrical(
    path, *, timezone="Europe/Zurich", time_column=None, value_column=None, quantity="auto"
):
    frame = pd.read_csv(path)
    if time_column is None:
        names = [c for c in frame if c.lower().strip() in {"time", "timestamp", "datetime"}]
        if len(names) != 1:
            raise ValueError("Select the electrical CSV time column.")
        time_column = names[0]

    def kind(column):
        name = column.lower()
        if "volt" in name or "[v]" in name:
            return "voltage"
        if "current" in name or "[a]" in name:
            return "current"
        return None

    numeric = [c for c in frame if c != time_column and pd.api.types.is_numeric_dtype(frame[c])]
    if value_column is None:
        preferred = [
            c for c in numeric if kind(c) == ("voltage" if quantity == "auto" else quantity)
        ]
        if len(preferred) == 1:
            value_column = preferred[0]
        elif len(numeric) == 1:
            value_column = numeric[0]
        else:
            raise ValueError(
                "Select the electrical CSV value column; multiple or no matching columns found."
            )
    detected = kind(value_column)
    selected_quantity = detected if quantity == "auto" else quantity
    if selected_quantity not in {"voltage", "current"}:
        raise ValueError("Electrical quantity is ambiguous; select Voltage or Current explicitly.")
    if quantity != "auto" and detected is not None and detected != quantity:
        raise ValueError(f"Selected column contains {detected}, not {quantity}.")
    raw_time = frame[time_column]
    if pd.Timestamp(raw_time.iloc[0]).tzinfo is None:
        stamps = pd.to_datetime(raw_time, errors="raise").dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        )
    else:
        stamps = pd.to_datetime(raw_time, utc=True, errors="raise")
    seconds = stamps.dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    values = pd.to_numeric(frame[value_column], errors="raise").to_numpy(dtype=float)
    if len(values) < 2 or stamps.isna().any() or not np.isfinite(values).all():
        raise ValueError(
            "Electrical CSV needs at least two finite observations with valid timestamps."
        )
    delta = np.diff(seconds)
    if np.any(delta < 0):
        raise ValueError("Electrical timestamps must be in increasing order.")
    # Plot-only data can contain distinct readings with the same exported clock
    # value. Preserve every row: neither choose one nor invent subsecond times.
    if not np.any(delta > 0):
        raise ValueError("Electrical CSV needs at least two distinct timestamps.")
    return Trace(
        seconds,
        values,
        "Voltage [V]" if selected_quantity == "voltage" else "Current [A]",
        str(Path(path).resolve()),
        value_column,
    )


def save_operating_conditions(
    directory, traces, *, timezone="Europe/Zurich", bounds=None, dpi=160, fit_to_data=False
):
    """One row per available source; time is the Grafana wall clock, without resampling."""
    from matplotlib import dates as mdates

    from .plots import plt

    traces = [trace for trace in traces if trace is not None]
    if not traces:
        return
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(traces),
        1,
        sharex=True,
        squeeze=False,
        figsize=(12, 2.5 * len(traces) + 1),
        layout="constrained",
    )
    metadata = {
        "timezone": timezone,
        "clock_bounds_epoch_s": bounds,
        "requested_clock_bounds_epoch_s": bounds,
        "fit_to_available_telemetry": fit_to_data,
        "traces": [],
    }
    data_start, data_end = np.inf, -np.inf
    for index, (axis, trace) in enumerate(zip(axes[:, 0], traces)):
        selected = (
            np.ones(len(trace.seconds), dtype=bool)
            if bounds is None
            else ((trace.seconds >= bounds[0]) & (trace.seconds < bounds[1]))
        )
        time, values = trace.seconds[selected], trace.values[selected]
        dates = pd.to_datetime(time, unit="s", utc=True).tz_convert(timezone)
        if len(time):
            data_start = min(data_start, float(time[0]))
            data_end = max(data_end, float(time[-1]))
            intervals = np.diff(trace.seconds)
            cadence = np.median(intervals[intervals > 0])
            gap = np.flatnonzero(np.diff(time) > 3 * cadence) + 1
            x = mdates.date2num(dates.to_pydatetime())
            axis.plot(
                np.insert(x, gap, np.nan), np.insert(values, gap, np.nan), lw=0.9, color=f"C{index}"
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No observations in this time window",
                transform=axis.transAxes,
                ha="center",
            )
        axis.set_ylabel(trace.label)
        axis.grid(alpha=0.2)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        name = "rpm" if trace.label == "Engine speed [RPM]" else "electrical"
        pd.DataFrame({"timestamp": dates.astype(str), "value": values}).to_csv(
            directory / f"operating_conditions_{name}.csv", index=False
        )
        metadata["traces"].append(
            {
                "source": trace.source,
                "column": trace.column,
                "label": trace.label,
                "samples": int(selected.sum()),
                "repeated_timestamp_rows": int(np.sum(np.diff(time) == 0)),
                "duplicate_policy": "preserve every exported observation at its timestamp; no averaging",
            }
        )
    zone = ZoneInfo(timezone)
    locator = mdates.AutoDateLocator(tz=zone)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=zone))
    axes[-1, 0].set_xlabel(f"Grafana local time [{timezone}]")
    plot_bounds = bounds
    if fit_to_data and bounds is not None and np.isfinite([data_start, data_end]).all():
        plot_bounds = (data_start, data_end)
        if data_start == data_end:
            plot_bounds = (max(bounds[0], data_start - 0.5), min(bounds[1], data_end + 0.5))
        metadata["clock_bounds_epoch_s"] = plot_bounds
    if plot_bounds is not None:
        axes[-1, 0].set_xlim(
            mdates.date2num(pd.to_datetime(plot_bounds, unit="s", utc=True).to_pydatetime())
        )
    axes[0, 0].set_title("Operating conditions")
    fig.savefig(directory / "operating_conditions.png", dpi=dpi)
    plt.close(fig)
    (directory / "operating_conditions.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
