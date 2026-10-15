"""CSV parsing deliberately rejects data that would silently corrupt a spectrum."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Recording:
    time: np.ndarray
    channels: dict[str, np.ndarray]
    sample_rate: float
    timing_jitter: float | None


def read_csv(
    path: str | Path,
    *,
    time_column: str | None = None,
    channels: list[str] | None = None,
    sample_rate: float | None = None,
    delimiter: str | None = None,
    decimal: str = ".",
    skiprows: int = 0,
    header: bool = True,
    time_unit: str = "s",
    jitter_tolerance: float = 0.01,
    scale: float = 1.0,
) -> Recording:
    """Read numeric channels. Headerless columns are named '0', '1', etc.

    Numeric timestamps are elapsed units; datetime timestamps are parsed as UTC.
    Jitter is max interval deviation / median interval. No resampling or row dropping.
    """
    if not np.isfinite(jitter_tolerance) or not 0 <= jitter_tolerance < 1:
        raise ValueError("Timing tolerance must be finite and between 0 (inclusive) and 1.")
    if not np.isfinite(scale):
        raise ValueError("Calibration scale must be finite.")
    if sample_rate is not None and (not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("Sample rate must be finite and positive.")
    if skiprows < 0:
        raise ValueError("Skiprows must be non-negative.")
    if delimiter is None:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for _ in range(skiprows):
                stream.readline()
            sample = stream.read(65536)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","  # A single-column file has no delimiter to sniff.
    if len(delimiter) != 1:
        raise ValueError("Delimiter must be a single character.")
    frame = pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        decimal=decimal,
        skiprows=skiprows,
        header=0 if header else None,
        encoding="utf-8-sig",
    )
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Column names must be unique after stripping whitespace.")
    if len(frame) < 4:
        raise ValueError("At least four samples are required.")
    if time_column is None and sample_rate is None:
        candidates = [c for c in frame if c.lower() in {"time", "time_s", "timestamp", "t"}]
        if len(candidates) != 1:
            raise ValueError("Specify --time-column or --sample-rate; time column is ambiguous.")
        time_column = candidates[0]
    jitter = None
    if time_column is not None:
        if time_column not in frame:
            raise ValueError(f"Time column {time_column!r} not found. Available: {list(frame)}")
        raw = frame[time_column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
            if time_unit not in factors:
                raise ValueError("Numeric time unit must be s, ms, us, or ns.")
            # Subtract origin before conversion to preserve precision for large counters.
            time = (numeric - numeric.iloc[0]).to_numpy(dtype=float) * factors[time_unit]
        elif time_unit == "datetime":
            stamps = pd.to_datetime(raw, errors="raise", utc=True)
            time = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy()
        else:
            raise ValueError("Time must be numeric; use --time-unit datetime for date strings.")
        intervals = np.diff(time)
        if not np.isfinite(time).all() or np.any(intervals <= 0):
            raise ValueError("Timestamps must be finite, strictly increasing and unique.")
        interval = float(np.median(intervals))
        jitter = float(np.max(np.abs(intervals - interval)) / interval)
        if jitter > jitter_tolerance:
            raise ValueError(
                f"Uneven sampling ({jitter:.2%} maximum interval deviation). "
                "Check dropped samples/time units or resample explicitly before analysis."
            )
        inferred_rate = 1.0 / interval
        if sample_rate is not None and not np.isclose(
            sample_rate, inferred_rate, rtol=max(jitter_tolerance, 1e-6)
        ):
            raise ValueError("Specified sample rate disagrees with the timestamp sample rate.")
        sample_rate = inferred_rate if sample_rate is None else sample_rate
    else:
        time = np.arange(len(frame)) / sample_rate
    if channels is None:
        channels = [
            c for c in frame if c != time_column and pd.api.types.is_numeric_dtype(frame[c])
        ]
    if not channels:
        raise ValueError("No numeric channels found. Select them explicitly with --channels.")
    values = {}
    for name in channels:
        if name not in frame or name == time_column:
            raise ValueError(f"Invalid signal column {name!r}. Available: {list(frame)}")
        data = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float) * scale
        if not np.isfinite(data).all():
            raise ValueError(f"Channel {name!r} contains missing or non-finite values.")
        values[name] = data
    return Recording(time, values, float(sample_rate), jitter)
