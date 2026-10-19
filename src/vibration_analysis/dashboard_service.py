"""Local dashboard state, background jobs, named regions and reproducible sessions."""

import hashlib
import json
import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dashboard_analysis import Settings, compute_region
from .dashboard_data import activity_envelope, cache_recording, source_identity
from .io import Recording
from .operating_points import detect_runs_and_holds, read_rpm, source_origin
from .telemetry import Trace, read_electrical


@dataclass
class Dataset:
    id: str
    name: str
    recording: Recording
    config: dict
    default_channel: str
    mrpm: tuple | None
    raw_rpm: Trace | None
    electrical: Trace | None
    origin: float | None
    offset: float
    notes: list

    @property
    def duration(self):
        return float(self.recording.time[-1] + 1 / self.recording.sample_rate)


def automatic_offset(centers, rms, rpm_seconds, rpm, origin, threshold_rpm=500):
    threshold = max(float(np.percentile(rms, 25)) * 10, float(np.percentile(rms, 99)) * 0.05, 1e-12)
    active = rpm >= threshold_rpm
    if active.all() or not active.any():
        raise ValueError(
            "Auto alignment needs running and downtime in mRPM. Select Manual alignment and enter the offset."
        )
    offsets = np.arange(-300, 301, dtype=float)
    scores = []
    for offset in offsets:
        query = rpm_seconds - origin + offset
        inside = (query >= centers[0]) & (query <= centers[-1])
        if inside.mean() < 0.8 or not np.any(active & inside) or not np.any(~active & inside):
            scores.append(-1)
            continue
        observed = np.interp(query[inside], centers, rms) > threshold
        target = active[inside]
        scores.append(0.5 * (observed[target].mean() + (~observed[~target]).mean()))
    scores = np.asarray(scores)
    best = scores.max()
    near = offsets[scores >= best - 0.003]
    if best < 0.9 or np.ptp(near) > 15 or abs(near.mean()) > 295:
        raise ValueError(
            "Automatic mRPM alignment is ambiguous. Choose Manual and enter an offset checked against the timeline."
        )
    return float(np.median(near))


def validate_region(dataset, region):
    channel = str(region.get("channel") or dataset.default_channel)
    if channel not in dataset.recording.channels:
        raise ValueError("Choose a valid sensor channel.")
    start, end = float(region["start"]), float(region["end"])
    if (
        not np.isfinite(start)
        or not np.isfinite(end)
        or not 0 <= start < end <= dataset.duration + 1e-8
    ):
        raise ValueError(f"Region must satisfy 0 ≤ start < end ≤ {dataset.duration:.6g} s.")
    if (
        np.searchsorted(dataset.recording.time, end)
        - np.searchsorted(dataset.recording.time, start)
        < 4
    ):
        raise ValueError("Select at least four original samples.")
    return {
        "id": str(region.get("id") or uuid.uuid4().hex[:10]),
        "dataset": dataset.id,
        "recording": dataset.name,
        "channel": channel,
        "name": str(region.get("name") or "Selected region").strip()[:100],
        "start": start,
        "end": end,
    }


class DashboardService:
    def __init__(self, cache_root):
        self.cache_root = Path(cache_root).resolve()
        self.datasets = OrderedDict()
        self.jobs = {}
        self.bundles = OrderedDict()
        self.spectra = OrderedDict()
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vibration-dashboard")

    def submit(self, kind, function, *args):
        identifier = uuid.uuid4().hex
        with self.lock:
            self.jobs[identifier] = {
                "state": "running",
                "kind": kind,
                "message": "Starting…",
                "cancel": False,
            }

        def progress(message):
            with self.lock:
                if self.jobs[identifier]["cancel"]:
                    raise InterruptedError("Cancelled; original data was not changed.")
                self.jobs[identifier]["message"] = message

        def worker():
            try:
                result = function(*args, progress=progress)
                with self.lock:
                    self.jobs[identifier].update(state="done", result=result, message="Ready.")
            except Exception as exc:
                logging.getLogger(__name__).exception("Dashboard background job failed")
                with self.lock:
                    self.jobs[identifier].update(
                        state="error", message=f"{type(exc).__name__}: {exc}"
                    )

        self.executor.submit(worker)
        return identifier

    def job(self, identifier):
        with self.lock:
            return dict(self.jobs[identifier])

    def cancel(self, identifier):
        with self.lock:
            if identifier in self.jobs:
                self.jobs[identifier]["cancel"] = True

    def options(self):
        with self.lock:
            return [{"label": d.name, "value": d.id} for d in self.datasets.values()]

    def load(self, config, progress=lambda _: None):
        config = dict(config)
        for key in ("input", "mrpm_csv", "raw_rpm_csv", "electrical_csv"):
            config[key] = str(config.get(key) or "").strip().strip('"')
        if not config["input"]:
            raise ValueError("Choose a vibration CSV first.")
        for key, expected in config.get("saved_sources", {}).items():
            if source_identity(config[key]) != expected:
                raise ValueError(
                    f"{key} changed since this session was saved. Load it as a new recording."
                )
        timezone = config.get("timezone") or "Europe/Zurich"
        mrpm = raw = electrical = None
        origin = None
        notes = []
        if config["mrpm_csv"]:
            progress("Reading mRPM analysis log…")
            rt, speed, _ = read_rpm(
                config["mrpm_csv"],
                config.get("mrpm_time_column") or None,
                config.get("mrpm_column") or None,
                timezone=timezone,
            )
            mrpm = rt, speed
        if config["raw_rpm_csv"]:
            rt, speed, columns = read_rpm(
                config["raw_rpm_csv"],
                config.get("raw_rpm_time_column") or None,
                config.get("raw_rpm_column") or None,
                timezone=timezone,
            )
            raw = Trace(rt, speed, "Raw RPM", config["raw_rpm_csv"], columns["value_column"])
        if config["electrical_csv"]:
            electrical = read_electrical(
                config["electrical_csv"],
                timezone=timezone,
                time_column=config.get("electrical_time_column") or None,
                value_column=config.get("electrical_column") or None,
                quantity=config.get("electrical_quantity") or "auto",
            )
        try:
            origin = source_origin(
                Path(config["input"]), config.get("vibration_start") or None, timezone=timezone
            )
        except ValueError:
            if mrpm is not None or config.get("vibration_start"):
                raise
            if raw is not None or electrical is not None:
                notes.append(
                    "Telemetry cannot share the vibration timeline without a recording start timestamp. Reload with a timestamp to align it."
                )
        cached = cache_recording(config, self.cache_root, progress)
        recording = cached.recording
        preferred = next(iter(recording.channels))
        envelope = None
        if cached.config["active_channels"] or mrpm is not None:
            scores = {}
            for name, values in recording.channels.items():
                progress(f"Checking activity in {name}…")
                centers, rms = activity_envelope(
                    Recording(
                        recording.time,
                        {name: values},
                        recording.sample_rate,
                        recording.timing_jitter,
                    )
                )
                if not len(rms):
                    continue
                strongest = min(len(rms), max(3, int(np.ceil(len(rms) * 0.01))))
                scores[name] = float(np.median(np.partition(rms, -strongest)[-strongest:]))
                if name == preferred:
                    envelope = centers, rms
            if scores:
                preferred = max(scores, key=scores.get)
                envelope = activity_envelope(
                    Recording(
                        recording.time,
                        {preferred: recording.channels[preferred]},
                        recording.sample_rate,
                        recording.timing_jitter,
                    )
                )
        offset = float(config.get("offset") or 0)
        if not np.isfinite(offset):
            raise ValueError("Alignment offset must be finite.")
        if mrpm is not None:
            if config.get("alignment", "Auto") == "Auto":
                if envelope is None or not len(envelope[0]):
                    raise ValueError("Recording is too short for automatic alignment; use Manual.")
                progress("Matching mRPM starts/stops to vibration…")
                offset = automatic_offset(
                    *envelope, *mrpm, origin, float(config.get("running_rpm") or 500)
                )
                notes.append(
                    f"Suggested alignment offset: {offset:+g} s. Verify the telemetry on the overview."
                )
            mrpm = mrpm[0] - origin + offset, mrpm[1]
        if origin is not None:
            for trace in (raw, electrical):
                if trace is not None:
                    trace.seconds = trace.seconds - origin + offset
        merged = {
            **cached.config,
            **config,
            "unit": cached.config["unit"],
            "scale": cached.config["scale"],
        }
        merged["saved_sources"] = {
            key: source_identity(config[key])
            for key in ("input", "mrpm_csv", "raw_rpm_csv", "electrical_csv")
            if config[key]
        }
        identifier = uuid.uuid4().hex[:12]
        dataset = Dataset(
            identifier,
            str(config.get("name") or Path(config["input"]).stem),
            recording,
            merged,
            preferred,
            mrpm,
            raw,
            electrical,
            origin,
            offset,
            notes,
        )
        with self.lock:
            self.datasets[identifier] = dataset
        return identifier

    def suggest(self, dataset_id, config):
        dataset = self.datasets[dataset_id]
        if dataset.mrpm is None:
            raise ValueError(
                "Load an mRPM CSV to suggest operating regions. Raw RPM is only for plotting."
            )
        runs = detect_runs_and_holds(
            *dataset.mrpm,
            running_rpm=float(config.get("running_rpm") or 500),
            minimum_run=float(config.get("minimum_run") or 30),
            minimum_hold=float(config.get("minimum_hold") or 30),
            stability_window=float(config.get("stability_window") or 20),
            slope_limit=float(config.get("slope_limit") or 2),
            scatter_limit=float(config.get("scatter_limit") or 40),
        )
        trim_start, trim_end = (
            float(config.get("trim_start") or 0),
            float(config.get("trim_end") or 0),
        )
        if min(trim_start, trim_end) < 0 or not np.isfinite([trim_start, trim_end]).all():
            raise ValueError("Hold trims must be finite and nonnegative.")
        regions = []
        for test, run in enumerate(runs, 1):
            for hold in run["holds"]:
                start, end = (
                    max(0, hold["start"]) + trim_start,
                    min(dataset.duration, hold["end"]) - trim_end,
                )
                if end <= start:
                    continue
                t, rpm = dataset.mrpm
                inside = (t >= start) & (t < end)
                if inside.sum() < 2:
                    continue
                regions.append(
                    validate_region(
                        dataset,
                        {
                            "name": f"Test {test} · {rpm[inside].mean():.0f} RPM",
                            "channel": dataset.default_channel,
                            "start": start,
                            "end": end,
                        },
                    )
                )
        return regions

    def compare(self, regions, settings_dict, progress=lambda _: None):
        settings = Settings.from_dict(settings_dict)
        if not regions:
            raise ValueError("Select at least one region using the table checkboxes.")
        results = []
        units = set()
        for index, row in enumerate(regions, 1):
            dataset = self.datasets[row["dataset"]]
            region = validate_region(dataset, row)
            units.add(dataset.config["unit"])
            if len(units) > 1:
                raise ValueError(
                    "Regions must use the same calibrated unit. Reload with consistent unit/calibration settings."
                )
            key = hashlib.sha256(
                json.dumps([region, asdict(settings)], sort_keys=True).encode()
            ).hexdigest()
            progress(f"Region {index}/{len(regions)}: {region['name']}…")
            if key in self.spectra:
                result = self.spectra.pop(key)
                self.spectra[key] = result
            else:

                def region_progress(message, index=index, region=region):
                    progress(f"{index}/{len(regions)} · {region['name']} · {message}")

                result = compute_region(dataset, region, settings, region_progress)
                self.spectra[key] = result
                while len(self.spectra) > 16:
                    self.spectra.popitem(last=False)
            results.append(result)
        identifier = uuid.uuid4().hex
        with self.lock:
            self.bundles[identifier] = results
            while len(self.bundles) > 3:
                self.bundles.popitem(last=False)
        return identifier

    def session(self, regions, settings, display):
        validated = [validate_region(self.datasets[row["dataset"]], row) for row in regions]
        ids = {row["dataset"] for row in validated}
        return {
            "schema": 1,
            "application": "vibration-analysis interactive dashboard",
            "datasets": [
                {"id": d.id, "config": d.config} for d in self.datasets.values() if d.id in ids
            ],
            "regions": validated,
            "settings": asdict(Settings.from_dict(settings)),
            "display": display,
        }

    def restore(self, path, progress=lambda _: None):
        payload = json.loads(Path(str(path).strip().strip('"')).read_text(encoding="utf-8"))
        if payload.get("schema") != 1 or not isinstance(payload.get("datasets"), list):
            raise ValueError("Unsupported session file.")
        mapping = {}
        for entry in payload["datasets"]:
            mapping[entry["id"]] = self.load(entry["config"], progress)
        rows = []
        for row in payload["regions"]:
            rows.append(
                validate_region(
                    self.datasets[mapping[row["dataset"]]],
                    {**row, "dataset": mapping[row["dataset"]]},
                )
            )
        Settings.from_dict(payload["settings"])
        return {
            "dataset": next(iter(mapping.values()), None),
            "regions": rows,
            "settings": payload["settings"],
            "display": payload.get("display", {}),
        }

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["cancel"] = True
        self.executor.shutdown(wait=False, cancel_futures=True)
