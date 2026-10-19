"""Disk-backed CSV import and bounded, peak-preserving browser representations."""

import csv
import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import Recording, _timestamps_match_rounded_grid, read_csv

CACHE_VERSION = 1


def source_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def clean_config(config):
    """Normalise form values before cache hashing or persistence."""
    source = source_identity(config["input"])
    channels = config.get("channels") or []
    if isinstance(channels, str):
        channels = channels.split(",")
    result = {
        "input": source["path"],
        "header": config.get("header", "auto"),
        "time_column": str(config.get("time_column") or "").strip() or None,
        "time_unit": config.get("time_unit", "s"),
        "sample_rate": float(config["sample_rate"]) if config.get("sample_rate") else None,
        "channels": [str(c).strip() for c in channels if str(c).strip()],
        "delimiter": config.get("delimiter") or "auto",
        "decimal": config.get("decimal", "."),
        "skiprows": int(config.get("skiprows") or 0),
        "scale": float(config.get("scale") if config.get("scale") is not None else 1),
        "unit": str(config.get("unit") or "g").strip(),
        "active_channels": bool(config.get("active_channels", True)),
    }
    if result["skiprows"] < 0:
        raise ValueError("Metadata rows cannot be negative.")
    if not np.isfinite(result["scale"]):
        raise ValueError("Calibration multiplier must be finite.")
    if result["sample_rate"] is not None and (
        not np.isfinite(result["sample_rate"]) or result["sample_rate"] <= 0
    ):
        raise ValueError("Sample rate must be positive and finite.")
    return result, source


def display_indices(values, first=0, last=None, budget=8000):
    """Chronological bucket minima/maxima: preserve peaks, never average samples."""
    last = len(values) if last is None else last
    if last - first <= budget:
        return np.arange(first, last)
    edges = np.linspace(first, last, max(2, budget // 2), dtype=np.int64)
    indices = [first, last - 1]
    for a, b in itertools.pairwise(edges):
        block = values[a:b]
        if len(block):
            indices.extend((a + int(np.argmin(block)), a + int(np.argmax(block))))
    return np.unique(indices)


def activity_envelope(recording):
    """One-second AC RMS without a recording-sized grouping allocation."""
    values = next(iter(recording.channels.values()))
    block = max(4, round(recording.sample_rate))
    centers, levels = [], []
    for first in range(0, len(values) - block + 1, block * 256):
        chunk = values[first : min(len(values), first + block * 256)]
        chunk = chunk[: len(chunk) // block * block].reshape(-1, block)
        levels.extend(np.std(chunk, axis=1))
        centers.extend(
            recording.time[first + np.arange(len(chunk)) * block]
            + block / (2 * recording.sample_rate)
        )
    return np.asarray(centers), np.asarray(levels)


@dataclass
class CachedRecording:
    recording: Recording
    config: dict
    source: dict
    directory: Path


def _load_cache(directory):
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    channels = {
        name: np.load(directory / filename, mmap_mode="r")
        for name, filename in meta["channel_files"].items()
    }
    recording = Recording(
        np.load(directory / "time.npy", mmap_mode="r"),
        channels,
        meta["sample_rate"],
        meta["timing_jitter"],
    )
    return CachedRecording(recording, meta["config"], meta["source"], directory)


def cache_recording(config, cache_root, progress=lambda _: None):
    """Two-pass, bounded-memory import; subsequent loads memory-map cached arrays.

    The count pass avoids assuming one physical line equals one CSV record.
    Import remains cancellable between chunks through the progress callback.
    """
    config, identity = clean_config(config)
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        json.dumps([CACHE_VERSION, identity, config], sort_keys=True).encode()
    ).hexdigest()[:24]
    directory = cache_root / key
    if (directory / "metadata.json").is_file():
        progress("Opening cached recording…")
        return _load_cache(directory)
    delimiter = config["delimiter"]
    with Path(config["input"]).open(encoding="utf-8-sig", newline="") as stream:
        for _ in range(config["skiprows"]):
            stream.readline()
        sample = stream.read(65536)
    if delimiter == "auto":
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    header = config["header"]
    if header == "auto":
        row = next(csv.reader(sample.splitlines(), delimiter=delimiter))
        try:
            for item in row:
                if item.strip():
                    float(item.replace(config["decimal"], "."))
            header = False
        except ValueError:
            header = True
    else:
        header = header in (True, "yes")
    time_column = config["time_column"]
    if not header and time_column is None and config["sample_rate"] is None:
        time_column = "0"
    kwargs = {
        "sep": delimiter,
        "engine": "c",
        "float_precision": "round_trip",
        "decimal": config["decimal"],
        "skiprows": config["skiprows"],
        "header": 0 if header else None,
        "encoding": "utf-8-sig",
        "chunksize": 250000,
    }
    count, first_frame = 0, None
    nonempty_columns = set()
    for frame in pd.read_csv(config["input"], **kwargs):
        nonempty_columns.update(str(c).strip() for c in frame.columns[frame.notna().any()])
        if first_frame is None:
            first_frame = frame.copy()
        count += len(frame)
        progress(f"Inspecting CSV: {count:,} rows counted…")
    if first_frame is None or count < 4:
        raise ValueError("The vibration CSV needs at least four samples.")
    first_frame.columns = [str(c).strip() for c in first_frame]
    # Many DAQ exports end every row with a comma. Ignore only columns proven
    # empty across the entire file, never a sensor that happens to start idle.
    first_frame = first_frame.loc[:, [c for c in first_frame if c in nonempty_columns]]
    # Reuse the existing importer for format detection and first-block validation.
    with tempfile.TemporaryDirectory(prefix="import-", dir=cache_root) as temporary:
        temporary = Path(temporary)
        preview = temporary / "preview.csv"
        first_frame.to_csv(preview, index=False)
        initial = read_csv(
            preview,
            time_column=time_column,
            time_unit=config["time_unit"],
            sample_rate=config["sample_rate"],
            channels=config["channels"] or None,
            scale=config["scale"],
            delimiter=",",
        )
        if time_column is None and config["sample_rate"] is None:
            time_column = next(
                c
                for c in first_frame
                if c.lower() in {"time", "time_s", "timestamp", "t", "time (s)", "time [s]"}
            )
        names = list(initial.channels)
        numeric_origin = None
        clock_origin = None
        if time_column is not None:
            if config["time_unit"] == "datetime":
                clock_origin = pd.Timestamp(first_frame[time_column].iloc[0])
                clock_origin = (
                    clock_origin.tz_localize("UTC") if clock_origin.tzinfo is None else clock_origin
                )
            else:
                numeric_origin = float(first_frame[time_column].iloc[0])
        channel_files = {name: f"channel_{i}.npy" for i, name in enumerate(names)}
        mapped = {
            None: np.lib.format.open_memmap(
                temporary / "time.npy", mode="w+", dtype="float64", shape=(count,)
            )
        }
        mapped.update(
            {
                name: np.lib.format.open_memmap(
                    temporary / filename, mode="w+", dtype="float64", shape=(count,)
                )
                for name, filename in channel_files.items()
            }
        )
        first = 0
        previous = None
        minimum, maximum = np.inf, 0.0
        try:
            for frame in pd.read_csv(config["input"], **kwargs):
                frame.columns = [str(c).strip() for c in frame]
                stop = first + len(frame)
                if stop > count:
                    raise ValueError(
                        "The source CSV changed during import. Retry with a stable file."
                    )
                if time_column is None:
                    time = np.arange(first, stop) / config["sample_rate"]
                elif clock_origin is not None:
                    stamps = pd.to_datetime(frame[time_column], utc=True, errors="raise")
                    time = (stamps - clock_origin).dt.total_seconds().to_numpy()
                else:
                    factors = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
                    time = (
                        pd.to_numeric(frame[time_column], errors="raise").to_numpy()
                        - numeric_origin
                    ) * factors[config["time_unit"]]
                if not np.isfinite(time).all():
                    raise ValueError("Timestamps must be finite.")
                steps = np.diff(time) if previous is None else np.diff(np.r_[previous, time])
                if len(steps):
                    minimum = min(minimum, float(steps.min()))
                    maximum = max(maximum, float(steps.max()))
                if minimum <= 0:
                    raise ValueError("Timestamps must be strictly increasing and unique.")
                mapped[None][first:stop] = time
                for name in names:
                    values = (
                        pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
                        * config["scale"]
                    )
                    if not np.isfinite(values).all():
                        raise ValueError(f"Channel {name!r} contains missing or nonfinite values.")
                    mapped[name][first:stop] = values
                previous = time[-1]
                first = stop
                progress(f"Caching original samples: {first:,} / {count:,}…")
            if first != count or source_identity(config["input"]) != identity:
                raise ValueError("The source CSV changed during import. Retry with a stable file.")
            time = mapped[None]
            typical = float(np.median(np.diff(time[: min(count, 250000)])))
            jitter = max(abs(minimum - typical), abs(maximum - typical)) / typical
            if jitter > 0.01 and not _timestamps_match_rounded_grid(time):
                raise ValueError(
                    f"Uneven sampling ({jitter:.2%}). Check missing samples or timestamp precision."
                )
            rate = (count - 1) / float(time[-1] - time[0])
            if config["sample_rate"] is not None:
                if not np.isclose(rate, config["sample_rate"], rtol=0.01):
                    raise ValueError("Sample rate disagrees with CSV timestamps.")
                rate = config["sample_rate"]
            meta = {
                "config": config,
                "source": identity,
                "sample_rate": rate,
                "timing_jitter": jitter,
                "channel_files": channel_files,
                "samples": count,
            }
            (temporary / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            preview.unlink()
        finally:
            for array in mapped.values():
                array.flush()
                array._mmap.close()
        if not directory.exists():
            os.replace(temporary, directory)
    progress("Recording cached; original samples are retained.")
    return _load_cache(directory)
