"""Command-line entry points. Use `vibration --help` for all CSV options."""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import signal

from .analysis import analyze
from .io import read_csv, select_active_channels, select_interval
from .progress import emit


def parser():
    root = argparse.ArgumentParser(description="FFT, PSD, waterfall and spectrogram from CSV.")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate reproducible synthetic example data.")
    demo.add_argument("--output", type=Path, default=Path("data/demo.csv"))
    demo.add_argument("--duration", type=float, default=8)
    demo.add_argument("--sample-rate", type=float, default=4096)
    demo.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("analyze", help="Analyze one or more CSV signal columns.")
    run.add_argument("input", type=Path)
    run.add_argument("--gui-progress", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument(
        "--mrpm-csv",
        "--rpm-csv",
        dest="rpm_csv",
        type=Path,
        help="mRPM log for alignment, test/hold detection and operating-point means (legacy --rpm-csv also supported).",
    )
    run.add_argument(
        "--electrical-csv",
        type=Path,
        help="Optional Grafana voltage/current CSV for the operating-condition time plot.",
    )
    run.add_argument(
        "--raw-rpm-csv",
        type=Path,
        help="Optional raw RPM log for operating-condition time plot ONLY; never used for detection or alignment.",
    )
    run.add_argument("--raw-rpm-time-column")
    run.add_argument("--raw-rpm-column")
    run.add_argument("--electrical-time-column")
    run.add_argument("--electrical-column")
    run.add_argument(
        "--electrical-quantity", choices=["auto", "voltage", "current"], default="auto"
    )
    run.add_argument(
        "--waterfall2d-fmax",
        type=float,
        help="Maximum frequency of the selected-range 2D RPM waterfall; blank uses plot maximum. A broadband version is always saved too.",
    )
    run.add_argument("--rpm-time-column")
    run.add_argument(
        "--harmonic-orders",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Shaft orders to track against mRPM, including fractions; blank disables.",
    )
    run.add_argument(
        "--harmonic-tolerance-hz",
        type=float,
        default=5,
        help="Half-width around order × RPM/60; at least one FFT bin is searched.",
    )
    run.add_argument(
        "--grafana-start",
        help="Optional local Grafana window start: YYYY-MM-DD HH:MM:SS or HH:MM:SS.",
    )
    run.add_argument("--grafana-end", help="Optional local Grafana window end, exclusive.")
    run.add_argument(
        "--grafana-timezone",
        default="Europe/Zurich",
        help="Grafana display timezone (default Europe/Zurich).",
    )
    run.add_argument("--rpm-column")
    run.add_argument(
        "--vibration-start", help="Timestamp matching the RPM clock; default from filename."
    )
    run.add_argument("--rpm-alignment", choices=["Auto", "Manual"], default="Auto")
    run.add_argument(
        "--rpm-offset-seconds",
        type=float,
        default=0,
        help="Manual: add to RPM elapsed times to map them onto the vibration timeline.",
    )
    run.add_argument("--running-rpm", type=float, default=500)
    run.add_argument("--minimum-run-seconds", type=float, default=30)
    run.add_argument("--minimum-hold-seconds", type=float, default=30)
    run.add_argument("--rpm-stability-seconds", type=float, default=20)
    run.add_argument(
        "--rpm-slope-limit",
        type=float,
        default=2,
        help="Maximum local RPM drift per second for a hold.",
    )
    run.add_argument(
        "--rpm-scatter-limit",
        type=float,
        default=40,
        help="Maximum RPM RMS scatter about the local fitted trend.",
    )
    run.add_argument(
        "--start-seconds", type=float, default=0, help="Interval start on the CSV elapsed timeline."
    )
    run.add_argument(
        "--end-seconds", type=float, help="Interval end, exclusive; default end of recording."
    )
    run.add_argument(
        "--trim-start-seconds",
        type=float,
        default=0,
        help="Discard seconds after interval start; GUI default 5.",
    )
    run.add_argument(
        "--trim-end-seconds",
        type=float,
        default=0,
        help="Discard seconds before interval end; GUI default 5.",
    )
    run.add_argument(
        "--time-column", help="Numeric or datetime column; auto-detect time/time_s/timestamp/t."
    )
    run.add_argument("--time-unit", choices=["s", "ms", "us", "ns", "datetime"], default="s")
    run.add_argument(
        "--sample-rate", type=float, help="Hz; required if no time column is available."
    )
    run.add_argument("--channels", nargs="+", help="Default: all numeric columns except time.")
    run.add_argument(
        "--active-channels",
        action="store_true",
        help="Automatically exclude inputs with negligible activity relative to the strongest; explicit --channels overrides.",
    )
    run.add_argument("--delimiter", help="Default: auto-detect. Use ';' for semicolon CSVs.")
    run.add_argument("--decimal", choices=[".", ","], default=".")
    run.add_argument(
        "--skiprows", type=int, default=0, help="Metadata lines before the header/data."
    )
    run.add_argument("--no-header", action="store_true", help="Columns are then named 0, 1, 2, ...")
    run.add_argument(
        "--timing-tolerance",
        type=float,
        default=0.01,
        help="Max relative interval deviation (default 0.01 = 1%%).",
    )
    run.add_argument(
        "--scale", type=float, default=1, help="Multiply signal columns by calibration factor."
    )
    run.add_argument(
        "--unit", default="g", help="Unit AFTER calibration; labeling only (default g)."
    )
    run.add_argument("--window", choices=["hann", "hamming", "flattop", "boxcar"], default="hann")
    run.add_argument("--detrend", choices=["constant", "linear", "none"], default="constant")
    run.add_argument(
        "--segment-seconds",
        type=float,
        default=0.5,
        help="Welch PSD and waterfall window duration (default 0.5 s).",
    )
    run.add_argument(
        "--overlap", type=float, default=0.5, help="Fraction of segment length, 0 <= overlap < 1."
    )
    run.add_argument(
        "--band",
        action="append",
        default=[],
        metavar="LOW:HIGH",
        help="Band RMS in Hz; repeatable, e.g. --band 100:500.",
    )
    run.add_argument(
        "--fmin",
        type=float,
        default=0,
        help="Trimmed-view minimum frequency, Hz; broadband views still start at zero.",
    )
    run.add_argument(
        "--fmax",
        type=float,
        help="Trimmed-view maximum frequency, Hz (default Nyquist); broadband images are also saved.",
    )
    run.add_argument(
        "--plots",
        nargs="+",
        choices=["all", "time", "fft", "psd", "waterfall", "spectrogram"],
        default=["all"],
    )
    run.add_argument("--dpi", type=int, default=160)
    run.add_argument(
        "--waterfall-max-frames",
        type=int,
        default=60,
        help="Maximum displayed waterfall spectra; 0 shows all (default 60).",
    )
    run.add_argument(
        "--psd-highpass-hz",
        type=float,
        default=0,
        help="High-pass cutoff for PSD/spectrogram only; 0 disables (GUI default 10 Hz).",
    )
    run.add_argument(
        "--psd-filter-order",
        type=int,
        default=4,
        help="Butterworth order per pass; forward-backward filtering doubles the roll-off order (default 4).",
    )
    run.add_argument(
        "--spectrogram-seconds",
        type=float,
        default=0.1,
        help="Independent spectrogram window duration (default 0.1 s).",
    )
    run.add_argument("--spectrogram-overlap", type=float, default=0.8)
    run.add_argument(
        "--spectrogram-range-db",
        type=float,
        default=60,
        help="Color range below maximum PSD (default 60 dB).",
    )
    run.add_argument(
        "--spectrogram-colormap", choices=["viridis", "magma", "turbo"], default="viridis"
    )
    run.add_argument(
        "--spectrogram-interpolation", choices=["bilinear", "nearest"], default="bilinear"
    )
    run.add_argument(
        "--spectrogram-headroom-db",
        type=float,
        default=6,
        help="Automatic upper limit above peak PSD (default 6 dB).",
    )
    run.add_argument(
        "--spectrogram-lower-db", type=float, help="Fixed lower PSD color limit, dB re 1 unit²/Hz."
    )
    run.add_argument(
        "--spectrogram-upper-db",
        type=float,
        help="Fixed upper PSD color limit; blank uses peak plus headroom.",
    )
    run.add_argument(
        "--no-spectrogram-rpm",
        action="store_true",
        help="Hide the mean RPM operating-point labels.",
    )
    return root


def make_demo(args):
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("Demo duration must be finite and positive.")
    if not np.isfinite(args.sample_rate) or args.sample_rate < 2048:
        raise ValueError("Demo sample rate must be at least 2048 Hz for its fixed tones.")
    time = np.arange(round(args.duration * args.sample_rate)) / args.sample_rate
    if len(time) < 4:
        raise ValueError("Demo must contain at least four samples.")
    rng = np.random.default_rng(args.seed)
    steady = (
        0.75 * np.sin(2 * np.pi * 100 * time)
        + 2 * np.sin(2 * np.pi * 500 * time + 0.4)
        + 1.5 * np.sin(2 * np.pi * 700 * time + 1.2)
        + 0.15 * rng.normal(size=len(time))
    )
    sweep = 0.7 * signal.chirp(time, f0=80, f1=800, t1=args.duration)
    impact = 4 * np.exp(-(((time - 0.6 * args.duration) / 0.003) ** 2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_s": time,
            "accel_steady_g": steady,
            "accel_sweep_g": sweep + impact + 0.08 * rng.normal(size=len(time)),
        }
    ).to_csv(args.output, index=False)
    print(f"Created {args.output.resolve()} ({len(time)} samples, synthetic acceleration in g).")


def run_analysis(args):
    if (args.grafana_start or args.grafana_end) and args.rpm_csv is None:
        raise ValueError(
            "A Grafana clock-time window requires an mRPM CSV for alignment. Otherwise use elapsed interval limits."
        )
    args.raw_rpm_trace = None
    if args.raw_rpm_csv is not None:
        from .operating_points import read_rpm
        from .telemetry import Trace

        seconds, values, columns = read_rpm(
            args.raw_rpm_csv,
            args.raw_rpm_time_column,
            args.raw_rpm_column,
            timezone=args.grafana_timezone,
        )
        args.raw_rpm_trace = Trace(
            seconds,
            values,
            "Engine speed [RPM]",
            str(args.raw_rpm_csv.resolve()),
            columns["value_column"],
        )
        emit(args, "Raw RPM CSV loaded for the operating-condition time plot only.")
    args.electrical_trace = None
    if args.electrical_csv is not None:
        from .telemetry import read_electrical

        args.electrical_trace = read_electrical(
            args.electrical_csv,
            timezone=args.grafana_timezone,
            time_column=args.electrical_time_column,
            value_column=args.electrical_column,
            quantity=args.electrical_quantity,
        )
        emit(
            args, f"Electrical CSV: {args.electrical_trace.label} ({args.electrical_trace.column})."
        )
    if args.electrical_trace is not None:
        repeated = int(np.sum(np.diff(args.electrical_trace.seconds) == 0))
        if repeated:
            emit(
                args,
                f"Electrical import: retained {repeated} additional reading(s) sharing exported timestamps; no averaging.",
            )
    if args.waterfall2d_fmax is not None and (
        not np.isfinite(args.waterfall2d_fmax) or args.waterfall2d_fmax <= args.fmin
    ):
        raise ValueError("2D waterfall max frequency must be finite and greater than plot minimum.")
    from .harmonics import parse_orders

    parse_orders(args.harmonic_orders)
    if not np.isfinite(args.harmonic_tolerance_hz) or args.harmonic_tolerance_hz < 0:
        raise ValueError("Order search tolerance must be finite and nonnegative.")
    emit(args, "Reading the vibration CSV…")
    bands = []
    for band in args.band:
        pair = band.split(":")
        if len(pair) != 2:
            raise ValueError("Bands must be LOW:HIGH, for example 100:500.")
        bands.append(tuple(map(float, pair)))
    read_started = perf_counter()
    recording = read_csv(
        args.input,
        time_column=args.time_column,
        channels=args.channels,
        sample_rate=args.sample_rate,
        delimiter=args.delimiter,
        decimal=args.decimal,
        skiprows=args.skiprows,
        header=not args.no_header,
        time_unit=args.time_unit,
        jitter_tolerance=args.timing_tolerance,
        scale=args.scale,
    )
    emit(
        args,
        f"CSV loaded: {len(recording.time):,} samples, "
        f"{len(recording.channels)} channel(s), in {perf_counter() - read_started:.1f} s.",
    )
    if args.waterfall2d_fmax is not None and args.waterfall2d_fmax > recording.sample_rate / 2 * (
        1 + 1e-10
    ):
        raise ValueError("2D waterfall max frequency must be at or below Nyquist.")
    available_count = len(recording.channels)
    args.channel_selection_report = None
    if args.active_channels and args.channels is None:
        emit(args, "Identifying active sensor channels…", history=False)
        recording, args.channel_selection_report = select_active_channels(recording)
        emit(
            args,
            f"Auto selection: using {len(recording.channels)} of {available_count} channels. "
            f"Excluded {available_count - len(recording.channels)} with negligible relative activity.",
        )
        print("Channel selection details: " + json.dumps(args.channel_selection_report), flush=True)
    else:
        emit(
            args,
            f"Using {len(recording.channels)} sensor channel(s). Results use one subfolder per sensor.",
        )
    print("Selected sensors: " + ", ".join(recording.channels), flush=True)
    if args.rpm_csv is not None:
        from .batch import run_rpm_analysis

        return run_rpm_analysis(args, recording, bands)
    recording, interval = select_interval(
        recording,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        trim_start_seconds=args.trim_start_seconds,
        trim_end_seconds=args.trim_end_seconds,
    )
    if args.raw_rpm_trace is not None or args.electrical_trace is not None:
        from .operating_points import source_origin
        from .telemetry import save_operating_conditions

        try:
            origin = source_origin(args.input, args.vibration_start, timezone=args.grafana_timezone)
        except ValueError:
            if args.vibration_start:
                raise
            origin = None
        bounds = (
            None
            if origin is None
            else (
                origin + interval["retained_first_time_seconds"],
                origin + interval["retained_last_time_seconds"] + 1 / recording.sample_rate,
            )
        )
        if origin is None:
            emit(
                args,
                "Operating-condition plot uses the full optional logs: no vibration start timestamp is available for clipping.",
            )
        save_operating_conditions(
            args.output,
            [args.raw_rpm_trace, args.electrical_trace],
            timezone=args.grafana_timezone,
            bounds=bounds,
            fit_to_data=True,
            dpi=args.dpi,
        )
    emit(args, "Generating plots and numerical results…")
    reports = export_recording(args, recording, interval, bands)
    emit(args, "Complete — results are ready.", complete=True)
    return reports


def export_recording(args, recording, interval, bands, context=None, rpm_points=None):
    print(
        f"Retained CSV interval: {interval['retained_first_time_seconds']:.6g}–"
        f"{interval['retained_last_time_seconds']:.6g} s "
        f"({interval['retained_duration_seconds']:.6g} s, {len(recording.time)} samples)."
    )
    fmax = recording.sample_rate / 2 if args.fmax is None else args.fmax
    if not (
        np.isfinite(args.fmin)
        and np.isfinite(fmax)
        and 0 <= args.fmin < fmax <= recording.sample_rate / 2 * (1 + 1e-10)
    ):
        raise ValueError("Require 0 <= fmin < fmax <= Nyquist.")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive.")
    if args.waterfall_max_frames < 0:
        raise ValueError("Waterfall maximum frames must be 0 (all) or a positive integer.")
    if not np.isfinite(args.spectrogram_range_db) or not 0 < args.spectrogram_range_db <= 200:
        raise ValueError("Spectrogram color range must be between 0 and 200 dB.")
    from .plots import psd_band_rms, spectrogram_color_limits

    spectrogram_color_limits(
        np.array([1.0]),
        args.spectrogram_range_db,
        args.spectrogram_headroom_db,
        args.spectrogram_lower_db,
        args.spectrogram_upper_db,
    )
    plots = set(args.plots)
    if "all" in plots:
        plots = {"time", "fft", "psd", "waterfall", "spectrogram"}
    # Preflight all channel names and settings before writing output.
    names = [
        re.sub(r"[^a-zA-Z0-9_.-]+", "_", c).strip("._")[:80] or "channel"
        for c in recording.channels
    ]
    reports = []
    # Numeric prefix prevents sanitized-name collisions and Windows reserved names.
    for index, ((name, values), safe) in enumerate(zip(recording.channels.items(), names), 1):
        prefix = getattr(args, "progress_context", "Generating results")
        emit(args, f"{prefix} · Sensor {index}/{len(names)}: {name}", history=False)
        result = analyze(
            values,
            recording.sample_rate,
            window=args.window,
            detrend=args.detrend,
            segment_seconds=args.segment_seconds,
            overlap=args.overlap,
            bands=bands,
            psd_highpass_hz=args.psd_highpass_hz,
            psd_filter_order=args.psd_filter_order,
            spectrogram_seconds=args.spectrogram_seconds,
            spectrogram_overlap=args.spectrogram_overlap,
        )
        # Keep the source time reference in both figures and numerical exports.
        result.time += recording.time[0]
        result.spectrogram_time += recording.time[0]
        if result.summary["segment_was_shortened"]:
            print(f"Warning: {name}: segment shortened to the record length.", file=sys.stderr)
        if result.summary["segment_count"] < 2:
            print(
                f"Warning: {name}: only one segment; no Welch averaging or time evolution.",
                file=sys.stderr,
            )
        if not np.any((result.frequency >= args.fmin) & (result.frequency <= fmax)):
            raise ValueError("Plot frequency range contains no segment bins; use longer segments.")
        if "psd" in plots and fmax <= max(args.fmin, result.frequency[1]):
            raise ValueError("Log PSD plot needs a positive frequency range above the first bin.")
        if "spectrogram" in plots and not np.any(
            (result.spectrogram_frequency >= args.fmin) & (result.spectrogram_frequency <= fmax)
        ):
            raise ValueError(
                "Spectrogram display range contains no bins; use a longer spectrogram window."
            )
        directory = args.output / f"{index:02d}_{safe}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "frequency_hz": result.fft_frequency,
                "peak_amplitude": result.amplitude,
                "phase_rad": result.phase,
            }
        ).to_csv(directory / "fft.csv", index=False)
        pd.DataFrame({"frequency_hz": result.frequency, "psd": result.psd}).to_csv(
            directory / "psd.csv", index=False
        )
        np.savez_compressed(
            directory / "time_frequency.npz",
            frequency_hz=result.frequency,
            time_s=result.time,
            peak_amplitude=result.window_amplitude,
            psd=result.window_psd,
            spectrogram_frequency_hz=result.spectrogram_frequency,
            spectrogram_time_s=result.spectrogram_time,
            spectrogram_psd=result.spectrogram_psd,
        )
        summary = {
            "source": str(args.input.resolve()),
            "channel": name,
            "unit": args.unit,
            "calibration_scale": args.scale,
            "timing_jitter": recording.timing_jitter,
            "analysis_interval": interval,
            "operating_context": context,
            "channel_selection": getattr(args, "channel_selection_report", None),
            "csv_options": {
                k: getattr(args, k)
                for k in (
                    "time_column",
                    "time_unit",
                    "sample_rate",
                    "delimiter",
                    "decimal",
                    "skiprows",
                    "no_header",
                    "timing_tolerance",
                )
            },
            **result.summary,
            "frequency_plot_views_hz": {
                "broadband": [0, recording.sample_rate / 2],
                "trimmed": [args.fmin, fmax],
            },
            "psd_display_rms": {
                "broadband": result.summary["welch_rms"],
                "trimmed": psd_band_rms(result, args.fmin, fmax),
            },
            "waterfall_display": {
                "max_frames": args.waterfall_max_frames,
                "displayed_frames": len(result.time)
                if args.waterfall_max_frames == 0
                else min(args.waterfall_max_frames, len(result.time)),
                "available_frames": len(result.time),
            },
            "spectrogram_display": {
                "range_db": args.spectrogram_range_db,
                "headroom_db": args.spectrogram_headroom_db,
                "requested_lower_db": args.spectrogram_lower_db,
                "requested_upper_db": args.spectrogram_upper_db,
                "operating_points": rpm_points,
                "rpm_panel": bool(
                    rpm_points and not args.no_spectrogram_rpm and "spectrogram" in plots
                ),
                "limits_db": spectrogram_color_limits(
                    result.spectrogram_psd,
                    args.spectrogram_range_db,
                    args.spectrogram_headroom_db,
                    args.spectrogram_lower_db,
                    args.spectrogram_upper_db,
                )
                if "spectrogram" in plots
                else None,
                "colormap": args.spectrogram_colormap,
                "interpolation": args.spectrogram_interpolation,
                "shared_color_limits_between_views": True,
            },
        }
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        from .plots import save_plots

        save_plots(
            result,
            recording.time,
            values,
            directory,
            unit=args.unit,
            title=f"{args.input.stem} | {name}",
            plots=plots,
            fmin=args.fmin,
            fmax=fmax,
            dpi=args.dpi,
            spectrogram_range_db=args.spectrogram_range_db,
            spectrogram_colormap=args.spectrogram_colormap,
            spectrogram_interpolation=args.spectrogram_interpolation,
            waterfall_max_frames=args.waterfall_max_frames,
            spectrogram_headroom_db=args.spectrogram_headroom_db,
            spectrogram_lower_db=args.spectrogram_lower_db,
            spectrogram_upper_db=args.spectrogram_upper_db,
            rpm_points=None if args.no_spectrogram_rpm else rpm_points,
        )
        print(
            f"{name}: raw RMS={summary['raw_rms']:.5g}, Welch RMS={summary['welch_rms']:.5g} "
            f"{args.unit}; peak={summary['peak_absolute']:.5g} {args.unit} -> {directory.resolve()}"
        )
        reports.append(
            {
                "channel": name,
                "folder": directory.name,
                "frequency": result.frequency,
                "mean_amplitude": result.window_amplitude.mean(axis=0),
                "psd": result.psd,
                "summary": summary,
            }
        )
    return reports


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        make_demo(args) if args.command == "demo" else run_analysis(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
