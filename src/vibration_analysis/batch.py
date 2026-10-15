"""Organize whole-test, per-hold and comparison exports from an aligned RPM log."""

import json
from argparse import Namespace

import numpy as np
import pandas as pd

from .io import select_interval
from .operating_points import detect_runs_and_holds, estimate_offset, read_rpm, source_origin
from .progress import emit


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _overview(args, recording, time, rpm, runs):
    from .plots import plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
    axes[0].plot(time / 60, rpm, ".-", color="#2563eb", markersize=3, lw=1)
    axes[0].set(
        ylabel="Engine speed [RPM]",
        title="Suggested tests and holds — verify alignment and boundaries",
    )
    x = next(iter(recording.channels.values()))
    n = max(1, round(recording.sample_rate))
    whole = len(x) // n * n
    blocks = x[:whole].reshape(-1, n)
    rms = np.std(blocks, axis=1)
    centers = recording.time[np.arange(len(rms)) * n] + n / (2 * recording.sample_rate)
    axes[1].plot(centers / 60, np.maximum(rms, 1e-15), color="#ea580c", lw=0.7)
    axes[1].set(
        yscale="log", ylabel="Reference-channel AC RMS", xlabel="Source CSV elapsed time [min]"
    )
    axes[0].set_xlim(max(recording.time[0], time[0]) / 60, min(recording.time[-1], time[-1]) / 60)
    for index, run in enumerate(runs, 1):
        for ax in axes:
            ax.axvspan(run["start"] / 60, run["end"] / 60, alpha=0.05, color="black")
            for hold in run["holds"]:
                ax.axvspan(hold["start"] / 60, hold["end"] / 60, alpha=0.15, color="#0f766e")
        axes[0].text(
            run["start"] / 60,
            0.96,
            f" Test {index}",
            transform=axes[0].get_xaxis_transform(),
            va="top",
        )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(args.output / "detection_overview.png", dpi=args.dpi)
    plt.close(fig)


def _comparisons(args, root, points, unit):
    from .plots import plt

    if not points:
        return
    root.mkdir(parents=True, exist_ok=True)
    for channel_index in range(len(points[0]["reports"])):
        reports = [point["reports"][channel_index] for point in points]
        directory = root / reports[0]["folder"]
        directory.mkdir(exist_ok=True)
        frequency = reports[0]["frequency"]
        amplitude = np.array(
            [np.interp(frequency, r["frequency"], r["mean_amplitude"]) for r in reports]
        )
        psd = np.array([np.interp(frequency, r["frequency"], r["psd"]) for r in reports])
        means = np.array([point["metadata"]["mean_rpm"] for point in points])
        np.savez_compressed(
            directory / "operating_point_spectra.npz",
            frequency_hz=frequency,
            mean_rpm=means,
            mean_window_peak_amplitude=amplitude,
            psd=psd,
        )
        table = pd.DataFrame([point["metadata"] for point in points])
        table["welch_rms"] = [r["summary"]["welch_rms"] for r in reports]
        table.to_csv(directory / "operating_points.csv", index=False)
        selected = (frequency >= args.fmin) & (frequency <= (args.fmax or frequency[-1]))
        if any(p in args.plots for p in ("all", "fft", "waterfall")):
            fig = plt.figure(figsize=(11, 6), layout="constrained")
            ax = fig.add_subplot(111, projection="3d")
            for i, spectrum in enumerate(amplitude):
                ax.plot(
                    frequency[selected], np.full(selected.sum(), i + 1), spectrum[selected], lw=0.9
                )
            ax.set(
                xlabel="Frequency [Hz]",
                ylabel="Point / RPM",
                zlabel=f"Mean window peak amplitude [{unit}]",
                title=f"{reports[0]['channel']} | FFT comparison (unfiltered)\n{args.segment_seconds:g} s {args.window} windows; magnitude averaged",
            )
            ax.set_yticks(
                np.arange(1, len(means) + 1),
                [f"{i + 1}: {value:.0f}" for i, value in enumerate(means)],
            )
            ax.tick_params(axis="y", labelsize=8)
            ax.yaxis.labelpad = 16
            ax.view_init(elev=28, azim=-65)
            fig.savefig(directory / "fft_rpm_waterfall.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
        if any(p in args.plots for p in ("all", "psd")):
            fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
            for i, density in enumerate(psd):
                ax.plot(
                    frequency[selected],
                    np.maximum(density[selected], 1e-30),
                    label=f"{i + 1}: {means[i]:.0f} RPM",
                    lw=0.9,
                )
            ax.set(
                yscale="log",
                xlabel="Frequency [Hz]",
                ylabel=f"PSD [({unit})²/Hz]",
                title=f"{reports[0]['channel']} | Operating-point PSDs | HPF {args.psd_highpass_hz:g} Hz",
            )
            ax.legend()
            fig.savefig(directory / "psd_comparison.png", dpi=args.dpi)
            plt.close(fig)


def run_rpm_analysis(args, recording, bands):
    from .cli import export_recording

    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(
            "RPM batch output must be a new or empty folder; choose a new output to preserve previous results."
        )
    # Validate limits and trim values before writing any batch output.
    select_interval(
        recording,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        trim_start_seconds=args.trim_start_seconds,
        trim_end_seconds=args.trim_end_seconds,
    )
    rpm_seconds, rpm, columns = read_rpm(args.rpm_csv, args.rpm_time_column, args.rpm_column)
    emit(args, "Aligning RPM and vibration timelines…")
    origin = source_origin(args.input, args.vibration_start)
    if args.rpm_alignment == "Auto":
        offset, alignment = estimate_offset(recording, rpm_seconds, rpm, origin, args.running_rpm)
    else:
        offset = args.rpm_offset_seconds
        if not np.isfinite(offset):
            raise ValueError("RPM offset must be finite.")
        alignment = {
            "method": "manual",
            "note": "Inspect detection_overview.png to verify alignment.",
        }
    time = rpm_seconds - origin + offset
    alignment.update(
        {
            "rpm_csv": str(args.rpm_csv.resolve()),
            "rpm_columns": columns,
            "vibration_origin_timestamp_epoch_s": origin,
            "offset_seconds_added_to_rpm_elapsed": offset,
            "mapping": "CSV elapsed = RPM timestamp - vibration origin + offset",
            "clock_note": "Naive timestamps are treated as a common nominal clock. Specify timezone offsets explicitly if sources use different zones.",
        }
    )
    print(f"RPM alignment offset: {offset:+g} s; {alignment['method']}.", flush=True)
    emit(args, f"Time alignment: {offset:+g} s. Check the overview after completion.")
    runs = detect_runs_and_holds(
        time,
        rpm,
        running_rpm=args.running_rpm,
        minimum_run=args.minimum_run_seconds,
        minimum_hold=args.minimum_hold_seconds,
        stability_window=args.rpm_stability_seconds,
        slope_limit=args.rpm_slope_limit,
        scatter_limit=args.rpm_scatter_limit,
    )
    end = recording.time[-1] + 1 / recording.sample_rate
    high = end if args.end_seconds is None else min(end, args.end_seconds)
    low = max(0, args.start_seconds)
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= end:
        raise ValueError("Selected analysis limits must lie within the recording.")
    runs = [
        r for r in runs if min(r["end"], high) - max(r["start"], low) >= args.minimum_run_seconds
    ]
    if not runs:
        raise ValueError(
            "No sustained engine runs overlap the selected vibration interval. Check alignment and detection settings."
        )
    emit(
        args,
        f"Found {len(runs)} tests and {sum(len(r['holds']) for r in runs)} candidate operating points.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _json(args.output / "alignment.json", alignment)
    _json(
        args.output / "detection_settings.json",
        {
            k: getattr(args, k)
            for k in (
                "running_rpm",
                "minimum_run_seconds",
                "minimum_hold_seconds",
                "rpm_stability_seconds",
                "rpm_slope_limit",
                "rpm_scatter_limit",
                "trim_start_seconds",
                "trim_end_seconds",
            )
        },
    )
    _overview(args, recording, time, rpm, runs)
    test_rows = []
    for index, run in enumerate(runs, 1):
        test_label = f"Test {index}/{len(runs)}"
        emit(args, f"{test_label} — generating whole-test plots.")
        start, stop = max(low, run["start"]), min(high, run["end"])
        test_root = args.output / f"test_{index:02d}"
        whole, interval = select_interval(recording, start_seconds=start, end_seconds=stop)
        context = {
            "test": index,
            "scope": "whole test; includes startup, transitions and shutdown",
            "alignment": alignment,
        }
        export_recording(
            Namespace(
                **{
                    **vars(args),
                    "output": test_root / "whole_test",
                    "progress_context": f"{test_label} · Whole test",
                }
            ),
            whole,
            interval,
            bands,
            context,
        )
        points, point_rows = [], []
        for candidate_index, hold in enumerate(run["holds"], 1):
            a, b = max(start, hold["start"]), min(stop, hold["end"])
            retained_a, retained_b = a + args.trim_start_seconds, b - args.trim_end_seconds
            row = {
                "candidate": candidate_index,
                "start_seconds": a,
                "end_seconds": b,
                "trimmed_start_seconds": retained_a,
                "trimmed_end_seconds": retained_b,
            }
            if retained_b - retained_a < max(args.segment_seconds * 2, 4 / recording.sample_rate):
                row.update(
                    status="skipped",
                    reason="Too short after clipping/trimming for two Welch windows",
                )
                point_rows.append(row)
                emit(args, f"Warning: {test_label}, point {candidate_index}: {row['reason']}.")
                continue
            inside = (time >= retained_a) & (time < retained_b)
            if inside.sum() < 2:
                row.update(
                    status="skipped", reason="Fewer than two RPM observations in retained interval"
                )
                point_rows.append(row)
                emit(args, f"Warning: {test_label}, point {candidate_index}: {row['reason']}.")
                continue
            row.update(
                mean_rpm=float(rpm[inside].mean()),
                rpm_std=float(rpm[inside].std()),
                minimum_rpm=float(rpm[inside].min()),
                maximum_rpm=float(rpm[inside].max()),
                retained_seconds=retained_b - retained_a,
                status="analysed",
                interpretation="candidate hold; warmup/load state not inferred from RPM",
            )
            point_root = (
                test_root
                / "operating_points"
                / f"point_{candidate_index:02d}_{row['mean_rpm']:.0f}rpm"
            )
            point_label = f"{test_label} · Point {candidate_index}/{len(run['holds'])} · {row['mean_rpm']:.0f} RPM"
            emit(args, f"{point_label} — generating plots.")
            retained, interval = select_interval(
                recording,
                start_seconds=a,
                end_seconds=b,
                trim_start_seconds=args.trim_start_seconds,
                trim_end_seconds=args.trim_end_seconds,
            )
            reports = export_recording(
                Namespace(**{**vars(args), "output": point_root, "progress_context": point_label}),
                retained,
                interval,
                bands,
                {"test": index, **row, "alignment": alignment},
            )
            points.append({"reports": reports, "metadata": row})
            point_rows.append(row)
        pd.DataFrame(point_rows).to_csv(test_root / "operating_points.csv", index=False)
        emit(args, f"{test_label} — saving operating-point comparisons.")
        _comparisons(args, test_root / "comparisons", points, args.unit)
        if not points:
            print(
                f"Warning: test {index}: no settled holds qualified; whole-test plots were exported."
            )
        test_rows.append(
            {
                "test": index,
                "start_seconds": start,
                "end_seconds": stop,
                "analysed_holds": len(points),
            }
        )
    pd.DataFrame(test_rows).to_csv(args.output / "tests.csv", index=False)
    print(f"Completed {len(runs)} tests. Review {args.output / 'detection_overview.png'}.")
    emit(
        args,
        f"Complete — {len(runs)} tests, {sum(row['analysed_holds'] for row in test_rows)} operating points. Results are ready.",
        complete=True,
    )
