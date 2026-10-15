"""RPM-based run/hold suggestions. Clock alignment is an estimate, saved for review."""

import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd


def read_rpm(path, time_column=None, value_column=None, timezone=None):
    frame = pd.read_csv(path)
    if time_column is None:
        candidates = [c for c in frame if c.lower().strip() in {"time", "timestamp", "datetime"}]
        if len(candidates) != 1:
            raise ValueError("Specify the RPM timestamp column.")
        time_column = candidates[0]
    if value_column is None:
        candidates = [
            c for c in frame if c != time_column and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Specify the RPM value column; more than one numeric column is present."
            )
        value_column = candidates[0]
    explicit_zone = pd.Timestamp(frame[time_column].iloc[0]).tzinfo is not None
    if explicit_zone:
        time = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    else:
        time = pd.to_datetime(frame[time_column], errors="raise").dt.tz_localize(
            timezone or "UTC", ambiguous="raise", nonexistent="raise"
        )
    seconds = time.dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    rpm = pd.to_numeric(frame[value_column], errors="raise").to_numpy(dtype=float)
    if len(rpm) < 3 or time.isna().any() or not np.isfinite(rpm).all() or np.any(rpm < 0):
        raise ValueError(
            "RPM log needs at least three finite, non-negative values and valid times."
        )
    if np.any(np.diff(seconds) <= 0):
        raise ValueError("RPM timestamps must be unique and strictly increasing.")
    return (
        seconds,
        rpm,
        {
            "time_column": time_column,
            "value_column": value_column,
            "median_interval_seconds": float(np.median(np.diff(seconds))),
            "timestamps_have_timezone": explicit_zone,
            "naive_timestamp_timezone": timezone or "UTC (nominal clock)",
        },
    )


def grafana_window(
    start,
    end,
    *,
    rpm_seconds,
    origin,
    offset,
    timestamps_have_timezone=False,
    timezone="Europe/Zurich",
):
    """Map Grafana wall-clock bounds onto aligned source CSV elapsed seconds.

    Naive exports retain their nominal wall clock, matching read_rpm. Aware exports
    use UTC instants. Time-only entries use the first RPM date in that clock.
    """
    try:
        clock = pd.Timestamp(rpm_seconds[0], unit="s", tz="UTC")
        local_clock = clock.tz_convert(timezone)
        date = (local_clock if timestamps_have_timezone else clock).date().isoformat()

        def convert(value):
            if not value or not value.strip():
                return None
            text = value.strip()
            if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", text):
                text = date + " " + text
            stamp = pd.Timestamp(text)
            if pd.isna(stamp):
                raise ValueError("Timestamp is missing.")
            if timestamps_have_timezone:
                stamp = (
                    stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
                    if stamp.tzinfo is None
                    else stamp
                )
                instant = stamp.timestamp()
            else:
                if stamp.tzinfo is not None:
                    stamp = stamp.tz_convert(timezone).tz_localize(None)
                instant = stamp.tz_localize("UTC").timestamp()
            return instant - origin + offset

        low, high = convert(start), convert(end)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(
            "Invalid Grafana time window or timezone. Use YYYY-MM-DD HH:MM:SS or HH:MM:SS and a valid zone such as Europe/Zurich."
        ) from exc
    if low is not None and high is not None and low >= high:
        raise ValueError(
            "Grafana window end must be after its start; include full dates for overnight windows."
        )
    return (
        low,
        high,
        {
            "requested_start": start or None,
            "requested_end": end or None,
            "timezone": timezone,
            "rpm_timestamps_have_timezone": timestamps_have_timezone,
            "source_start_seconds": low,
            "source_end_seconds_exclusive": high,
            "time_only_date": date,
        },
    )


def source_origin(path, supplied=None, timezone=None):
    if supplied:
        stamp = pd.Timestamp(supplied)
    else:
        match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})(?:-(\d{3}))?", Path(path).stem)
        if not match:
            raise ValueError("Enter the vibration start timestamp to align with the RPM log.")
        date, clock = match[1].split("T")
        stamp = pd.Timestamp(date + "T" + clock.replace("-", ":") + "." + (match[2] or "000"))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone or "UTC", ambiguous="raise", nonexistent="raise")
    return stamp.timestamp()


def _groups(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def estimate_offset(recording, rpm_seconds, rpm, origin, running_rpm):
    """Match starts/stops using balanced binary agreement within +/-300 seconds.

    Positive offset maps an RPM timestamp later on the CSV elapsed timeline.
    Require observable running/downtime and reject weak alignments.
    """
    values = next(iter(recording.channels.values()))
    bins = np.floor(recording.time).astype(int)
    count = np.bincount(bins)
    sums = np.bincount(bins, weights=values)
    squares = np.bincount(bins, weights=values**2)
    valid = count > 0
    rms = np.sqrt(np.maximum(0, squares[valid] / count[valid] - (sums[valid] / count[valid]) ** 2))
    centers = np.flatnonzero(valid) + 0.5
    threshold = max(float(np.percentile(rms, 25)) * 10, float(np.percentile(rms, 99)) * 0.05, 1e-12)
    expected = rpm >= running_rpm
    if expected.all() or not expected.any():
        raise ValueError(
            "Auto alignment requires both running and downtime in RPM; use Manual mode."
        )
    scores = []
    offsets = np.arange(-300, 301, dtype=float)
    for offset in offsets:
        query = rpm_seconds - origin + offset
        inside = (query >= centers[0]) & (query <= centers[-1])
        if inside.mean() < 0.8 or not np.any(expected & inside) or not np.any(~expected & inside):
            scores.append(-1)
            continue
        observed = np.interp(query[inside], centers, rms) > threshold
        target = expected[inside]
        scores.append(0.5 * (np.mean(observed[target]) + np.mean(~observed[~target])))
    scores = np.asarray(scores)
    best = float(scores.max())
    if best < 0.9:
        raise ValueError(
            "No reliable start/stop alignment found within +/-300 s. Set the offset manually."
        )
    near = offsets[scores >= best - 0.003]
    if np.ptp(near) > 15 or abs(float(near.mean())) > 295:
        raise ValueError("Automatic clock offset is ambiguous; set it manually using known events.")
    offset = float(np.median(near))
    return offset, {
        "method": "automatic start/stop matching",
        "balanced_agreement": best,
        "near_best_offset_range_seconds": [float(near.min()), float(near.max())],
        "vibration_running_threshold": threshold,
        "reference_channel": next(iter(recording.channels)),
        "note": "Heuristic suggestion, not clock synchronization. Inspect detection_overview.png.",
    }


def detect_runs_and_holds(
    time,
    rpm,
    *,
    running_rpm=500,
    minimum_run=30,
    minimum_hold=30,
    stability_window=20,
    slope_limit=2,
    scatter_limit=40,
):
    """Find running intervals, then local low-slope, low-scatter plateau candidates.

    Gaps break runs. The rolling fit excludes missing intervals and large RPM steps.
    Plateau edges expand by half the fit window; caller applies guard trims afterward.
    """
    params = [running_rpm, minimum_run, minimum_hold, stability_window, slope_limit, scatter_limit]
    if not all(np.isfinite(v) and v > 0 for v in params):
        raise ValueError("RPM detection thresholds and durations must be finite and positive.")
    dt = float(np.median(np.diff(time)))
    if stability_window < 2 * dt:
        raise ValueError("RPM stability window must span at least two telemetry intervals.")
    gap = np.diff(time) > 2 * dt
    active = rpm >= running_rpm
    # Split on gaps without losing the samples either side of them.
    boundaries = np.r_[0, np.flatnonzero(gap) + 1, len(time)]
    runs = []
    for block_start, block_end in pairwise(boundaries):
        for lo, hi in _groups(active[block_start:block_end]):
            lo, hi = lo + block_start, hi + block_start
            start = time[lo] - (dt / 2 if lo == block_start else (time[lo] - time[lo - 1]) / 2)
            end = time[hi - 1] + (dt / 2 if hi == block_end else (time[hi] - time[hi - 1]) / 2)
            if end - start < minimum_run:
                continue
            stable = np.zeros(hi - lo, dtype=bool)
            half = stability_window / 2
            for local, i in enumerate(range(lo, hi)):
                left = max(lo, np.searchsorted(time, time[i] - half))
                right = min(hi, np.searchsorted(time, time[i] + half, side="right"))
                if right - left < 3 or time[right - 1] - time[left] < stability_window * 0.9:
                    continue
                t = time[left:right] - time[i]
                y = rpm[left:right]
                fit = np.polyfit(t, y, 1)
                scatter = np.sqrt(np.mean((y - np.polyval(fit, t)) ** 2))
                stable[local] = abs(fit[0]) <= slope_limit and scatter <= scatter_limit
            holds = []
            for first, last in _groups(stable):
                a = max(start, time[lo + first] - half)
                b = min(end, time[lo + last - 1] + half)
                if b - a >= minimum_hold:
                    # Adjacent local fits must not create overlapping reports.
                    if holds and a < holds[-1]["end"]:
                        a = holds[-1]["end"]
                    if b - a >= minimum_hold:
                        holds.append({"start": float(a), "end": float(b)})
            runs.append({"start": float(start), "end": float(end), "holds": holds})
    return runs
