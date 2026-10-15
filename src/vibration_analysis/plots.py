"""Headless plot exports; all numerical arrays are exported separately by the CLI."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
        peak_density = float(density.max())
        maximum_db = 10 * np.log10(peak_density if peak_density > 0 else 1e-20)
        hop = result.summary["spectrogram_hop_seconds"]
        df = result.summary["spectrogram_bin_width_hz"]
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
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
            vmin=maximum_db - spectrogram_range_db,
            vmax=maximum_db,
        )
        ax.grid(False)
        ax.set(
            xlabel="Window center [s]",
            ylabel="Frequency [Hz]",
            ylim=(fmin, fmax),
            title=(
                f"{title} | Spectrogram{filter_label}\n"
                f"Window {result.summary['spectrogram_segment_seconds'] * 1000:.3g} ms | "
                f"Hop {hop * 1000:.3g} ms | Bin spacing {df:.3g} Hz"
            ),
        )
        fig.colorbar(mesh, ax=ax, label=f"PSD [dB re 1 ({unit})²/Hz]")
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
