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


def select_active_channels(recording, relative_threshold=0.05):
    """Suggest active inputs from sustained AC activity, not single handling spikes.

    Score = median of the strongest 1% of one-second AC-RMS blocks (at least three,
    when available). Retain inputs
    above a fraction of the strongest score. This is not hardware connection detection.
    """
    if not np.isfinite(relative_threshold) or not 0 < relative_threshold <= 1:
        raise ValueError("Active-channel relative threshold must lie in (0, 1].")
    block = min(len(recording.time), max(4, round(recording.sample_rate)))
    scores = {}
    for name, values in recording.channels.items():
        usable = len(values) // block * block
        rms = np.std(values[:usable].reshape(-1, block), axis=1)
        strongest_count = min(len(rms), max(3, int(np.ceil(len(rms) * 0.01))))
        scores[name] = float(np.median(np.partition(rms, -strongest_count)[-strongest_count:]))
    strongest = max(scores.values())
    included = sorted(
        (name for name in scores if scores[name] >= strongest * relative_threshold),
        key=lambda name: scores[name],
        reverse=True,
    )
    report = {
        "method": "relative sustained vibration activity",
        "relative_threshold": relative_threshold,
        "activity_score": scores,
        "included": included,
        "excluded": [name for name in scores if name not in included],
        "note": "Amplitude-based suggestion. Select quiet sensors manually if they are required.",
    }
    return Recording(
        recording.time,
        {name: recording.channels[name] for name in included},
        recording.sample_rate,
        recording.timing_jitter,
    ), report


def select_interval(
    recording, *, start_seconds=0.0, end_seconds=None, trim_start_seconds=0.0, trim_end_seconds=0.0
):
    """Select a half-open interval on the original elapsed timeline, then trim it.

    Preserve original time coordinates so exported spectra can be traced to the source.
    No samples are modified or resampled.
    """
    duration = float(recording.time[-1] + 1 / recording.sample_rate)
    end = duration if end_seconds is None else end_seconds
    settings = (start_seconds, end, trim_start_seconds, trim_end_seconds)
    if not all(np.isfinite(value) for value in settings):
        raise ValueError("Interval limits and trims must be finite.")
    if not 0 <= start_seconds < end <= duration + 1e-8:
        raise ValueError(f"Interval must satisfy 0 <= start < end <= {duration:.6g} seconds.")
    if trim_start_seconds < 0 or trim_end_seconds < 0:
        raise ValueError("Start and end trim must be non-negative.")
    low, high = start_seconds + trim_start_seconds, end - trim_end_seconds
    if low >= high:
        raise ValueError("Trimming removes the entire selected interval; reduce the trims.")
    first = int(np.searchsorted(recording.time, low, side="left"))
    last = int(np.searchsorted(recording.time, high, side="left"))
    if last - first < 4:
        raise ValueError("Fewer than four samples remain after interval selection and trimming.")
    selected = Recording(
        recording.time[first:last],
        {key: values[first:last] for key, values in recording.channels.items()},
        recording.sample_rate,
        recording.timing_jitter,
    )
    metadata = {
        "source_samples": len(recording.time),
        "selected_start_seconds": start_seconds,
        "selected_end_seconds_exclusive": end,
        "trim_start_seconds": trim_start_seconds,
        "trim_end_seconds": trim_end_seconds,
        "retained_start_sample": first,
        "retained_end_sample_exclusive": last,
        "retained_first_time_seconds": float(selected.time[0]),
        "retained_last_time_seconds": float(selected.time[-1]),
        "retained_duration_seconds": (last - first) / recording.sample_rate,
        "time_coordinates": "elapsed seconds from first source CSV sample",
    }
    return selected, metadata


def _timestamps_match_rounded_grid(time):
    """Accept decimal export rounding only when the entire clock follows one grid.

    Check in blocks to avoid another recording-sized allocation. Missing samples
    or drifting clocks must still fail, even if individual steps look plausible.
    """
    spacing = (time[-1] - time[0]) / (len(time) - 1)
    probe = time[:4096]
    quantum = None
    epsilon = max(1e-12, float(np.max(np.abs(time[[0, -1]]))) * np.finfo(float).eps * 8)
    for decimals in range(13):
        candidate = 10.0**-decimals
        if candidate >= spacing / 10 or candidate <= epsilon * 4:
            continue
        if np.max(np.abs(probe - np.rint(probe / candidate) * candidate)) <= epsilon:
            quantum = candidate
            break
    if quantum is None:
        return False
    for first in range(0, len(time), 250000):
        block = time[first : first + 250000]
        if np.max(np.abs(block - np.rint(block / quantum) * quantum)) > epsilon:
            return False
        expected = time[0] + np.arange(first, first + len(block)) * spacing
        if np.max(np.abs(block - expected)) > quantum + epsilon:
            return False
    return True


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
        engine="c",
        float_precision="round_trip",
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
        candidates = [
            c
            for c in frame
            if c.lower() in {"time", "time_s", "timestamp", "t", "time (s)", "time [s]"}
        ]
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
        if jitter > jitter_tolerance and not _timestamps_match_rounded_grid(time):
            raise ValueError(
                f"Uneven sampling ({jitter:.2%} maximum interval deviation). "
                "Check dropped samples/time units or resample explicitly before analysis."
            )
        # Rounded CSV timestamps can alternate between two interval lengths.
        # Endpoint-average spacing avoids bias from their more frequent rounded value.
        inferred_rate = (len(time) - 1) / (time[-1] - time[0])
        if sample_rate is not None and not np.isclose(
            sample_rate, inferred_rate, rtol=max(jitter_tolerance, 1e-6)
        ):
            raise ValueError("Specified sample rate disagrees with the timestamp sample rate.")
        sample_rate = inferred_rate if sample_rate is None else sample_rate
    else:
        time = np.arange(len(frame)) / sample_rate
    if channels is None:
        channels = [
            c
            for c in frame
            if c != time_column
            and c.lower() not in {"sample", "sample index"}
            and not (c.startswith("Unnamed:") and frame[c].isna().all())
            and pd.api.types.is_numeric_dtype(frame[c])
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
