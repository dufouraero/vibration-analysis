"""Headless plot exports; all numerical arrays are exported separately by the CLI."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def spectrogram_color_limits(density, range_db=60, headroom_db=6, lower_db=None, upper_db=None):
    """Display limits in dB re 1 signal-unit squared/Hz; never alter PSD values."""
    if not np.isfinite(range_db) or not 0 < range_db <= 200:
        raise ValueError("Spectrogram color range must be between 0 and 200 dB.")
    if not np.isfinite(headroom_db) or headroom_db < 0:
        raise ValueError("Spectrogram headroom must be finite and nonnegative.")
    peak = float(np.max(density))
    upper = (
        10 * np.log10(peak if peak > 0 else 1e-20) + headroom_db if upper_db is None else upper_db
    )
    lower = upper - range_db if lower_db is None else lower_db
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Spectrogram limits must be finite, with lower dB < upper dB.")
    return float(lower), float(upper)


def save_plots(
    result,
    time,
    values,
    directory,
    *,
    unit,
    title,
    plots,
    fmin,
    fmax,
    dpi,
    spectrogram_range_db=60,
    spectrogram_colormap="viridis",
    spectrogram_interpolation="bilinear",
    waterfall_max_frames=60,
    spectrogram_headroom_db=6,
    spectrogram_lower_db=None,
    spectrogram_upper_db=None,
    rpm_points=None,
):
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.facecolor": "white",
        }
    )

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    cutoff = result.summary["psd_highpass_hz"]
    filter_label = (
        f" | HPF {cutoff:g} Hz, order {result.summary['psd_filter_order_per_pass']}/pass"
        if cutoff
        else " | HPF off"
    )

    if "time" in plots:
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        ax.plot(time, values, lw=0.6, color="#2563eb")
        ax.set(xlabel="Elapsed time [s]", ylabel=f"Signal [{unit}]", title=title)
        save(fig, "time")
    if "fft" in plots:
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        ax.plot(result.fft_frequency, result.amplitude, lw=0.8, color="#2563eb")
        ax.set(
            xlim=(fmin, fmax),
            xlabel="Frequency [Hz]",
            ylabel=f"Peak amplitude [{unit}]",
            title=f"{title} | FFT amplitude",
        )
        save(fig, "fft")
    if "psd" in plots:
        for logarithmic in (False, True):
            fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
            mask = (
                (result.frequency > 0)
                if logarithmic
                else np.ones(len(result.frequency), dtype=bool)
            )
            ax.plot(result.frequency[mask], result.psd[mask], lw=0.9, color="#0f766e")
            if not np.any(result.psd > 0):
                ax.set_ylim(1e-20, 1e-16)
                ax.text(
                    0.5,
                    0.5,
                    "Zero PSD: no positive values on log axis",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.set_yscale("log")
            if logarithmic:
                ax.set_xscale("log")
            left = max(fmin, result.frequency[1]) if logarithmic else fmin
            ax.set(
                xlim=(left, fmax),
                xlabel="Frequency [Hz]",
                ylabel=f"PSD [({unit})²/Hz]",
                title=f"{title} | Welch PSD{filter_label}\nRMS = {result.summary['welch_rms']:.4g} {unit}",
            )
            save(fig, "psd_log_x" if logarithmic else "psd_linear_x")
    selected = (result.frequency >= fmin) & (result.frequency <= fmax)
    frequency = result.frequency[selected]
    if "spectrogram" in plots:
        spec_selected = (result.spectrogram_frequency >= fmin) & (
            result.spectrogram_frequency <= fmax
        )
        spec_frequency = result.spectrogram_frequency[spec_selected]
        density = result.spectrogram_psd[:, spec_selected].T
        density_db = 10 * np.log10(np.maximum(density, np.finfo(float).tiny))
        lower_db, upper_db = spectrogram_color_limits(
            density,
            spectrogram_range_db,
            spectrogram_headroom_db,
            spectrogram_lower_db,
            spectrogram_upper_db,
        )
        hop = result.summary["spectrogram_hop_seconds"]
        df = result.summary["spectrogram_bin_width_hz"]
        if rpm_points:
            fig = plt.figure(figsize=(11, 6.5), layout="constrained")
            grid = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[1, 0.035])
            ax = fig.add_subplot(grid[0, 0])
            rpm_ax = fig.add_subplot(grid[1, 0], sharex=ax)
            color_ax = fig.add_subplot(grid[0, 1])
            for point in rpm_points:
                start, stop = point["start_seconds"], point["end_seconds"]
                rpm_ax.axvspan(start, stop, ymin=0.6, ymax=0.95, color="#dbeafe")
                rpm_ax.plot([start, stop], [0.8, 0.8], color="#475569", lw=1.5)
                rpm_ax.text(
                    (start + stop) / 2,
                    0.5,
                    f"{point['mean_rpm']:.0f} RPM",
                    ha="right",
                    va="top",
                    rotation=35,
                    fontsize=9,
                )
            rpm_ax.set(
                ylim=(-0.9, 1), yticks=[], ylabel="Mean RPM", xlabel="Source CSV elapsed time [s]"
            )
            rpm_ax.grid(False)
            rpm_ax.spines["left"].set_visible(False)
            ax.tick_params(labelbottom=False)
        else:
            fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
            color_ax = None
        mesh = ax.imshow(
            density_db,
            origin="lower",
            aspect="auto",
            extent=(
                result.spectrogram_time[0] - hop / 2,
                result.spectrogram_time[-1] + hop / 2,
                spec_frequency[0] - df / 2,
                spec_frequency[-1] + df / 2,
            ),
            interpolation=spectrogram_interpolation,
            interpolation_stage="data",
            cmap=spectrogram_colormap,
            vmin=lower_db,
            vmax=upper_db,
        )
        ax.grid(False)
        ax.set(
            xlabel="" if rpm_points else "Source CSV elapsed time [s]",
            ylabel="Frequency [Hz]",
            ylim=(fmin, fmax),
            title=(
                f"{title} | Spectrogram{filter_label}\n"
                f"Window {result.summary['spectrogram_segment_seconds'] * 1000:.3g} ms | "
                f"Hop {hop * 1000:.3g} ms | Bin spacing {df:.3g} Hz"
            ),
        )
        ax.set_xlim(result.spectrogram_time[0] - hop / 2, result.spectrogram_time[-1] + hop / 2)
        below, above = bool(np.any(density_db < lower_db)), bool(np.any(density_db > upper_db))
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        fig.colorbar(mesh, ax=ax, cax=color_ax, extend=extend, label=f"PSD [dB re 1 ({unit})^2/Hz]")
        save(fig, "spectrogram")
    if "waterfall" in plots:
        fig = plt.figure(figsize=(11, 6), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        frame_count = len(result.time)
        displayed_count = (
            frame_count if waterfall_max_frames == 0 else min(waterfall_max_frames, frame_count)
        )
        indices = np.unique(np.linspace(0, frame_count - 1, displayed_count).astype(int))
        for index in indices:
            ax.plot(
                frequency,
                np.full(len(frequency), result.time[index]),
                result.window_amplitude[index, selected],
                lw=0.7,
                color=plt.cm.viridis(index / max(len(result.time) - 1, 1)),
            )
        ax.set(
            xlabel="Frequency [Hz]",
            ylabel="Window center [s]",
            zlabel=f"Peak amplitude [{unit}]",
            title=f"{title} | FFT waterfall | {len(indices)} of {frame_count} frames",
        )
        ax.view_init(elev=28, azim=-65)
        save(fig, "waterfall")
