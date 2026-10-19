"""RPM-guided order candidates from existing, unfiltered window FFT magnitudes."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def parse_orders(value):
    if not str(value or "").strip():
        return []
    try:
        orders = sorted({float(part.strip()) for part in str(value).split(",")})
    except ValueError as exc:
        raise ValueError("Orders must be comma-separated numbers, e.g. 0.5,1,2,3,7.") from exc
    if len(orders) > 30 or not np.isfinite(orders).all() or min(orders) <= 0:
        raise ValueError("Choose up to 30 positive, finite shaft orders; blank disables tracking.")
    return orders


def track_orders(reports, means, orders, tolerance):
    """Choose a local maximum only inside each predicted order's search band.

    Peaks are candidates, not proof of combustion origin or structural resonance.
    Missing/ambiguous orders remain missing; no extrapolation beyond Nyquist.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Order search tolerance must be finite and nonnegative.")
    rows = []
    spacing = min([1.0, *np.diff(orders)])
    for point, (report, rpm) in enumerate(zip(reports, means), 1):
        f, amplitude = report["frequency"], report["mean_amplitude"]
        peaks, _ = find_peaks(amplitude)
        resolution = float(np.max(np.diff(f)))
        for order in orders:
            expected = order * rpm / 60
            half_width = max(tolerance, resolution)
            row = {
                "point": point,
                "mean_rpm": float(rpm),
                "order": order,
                "expected_hz": float(expected),
                "search_half_width_hz": half_width,
                "peak_hz": np.nan,
                "peak_amplitude": np.nan,
                "detuning_hz": np.nan,
                "status": "no_local_peak",
            }
            if expected > f[-1] or expected < f[0]:
                row["status"] = "outside_bandwidth"
            elif half_width >= rpm / 60 * spacing / 2:
                row["status"] = "ambiguous_search_width"
            else:
                candidates = peaks[abs(f[peaks] - expected) <= half_width]
                if len(candidates):
                    peak = candidates[np.argmax(amplitude[candidates])]
                    row.update(
                        peak_hz=float(f[peak]),
                        peak_amplitude=float(amplitude[peak]),
                        detuning_hz=float(f[peak] - expected),
                        status="candidate",
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def order_plots(directory, table, orders, unit, title, dpi):
    from .plots import plt

    colors = {order: f"C{i % 10}" for i, order in enumerate(orders)}
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    panel, axes = plt.subplots(
        int(np.ceil(len(orders) / 2)),
        2,
        squeeze=False,
        figsize=(12, 2.6 * int(np.ceil(len(orders) / 2)) + 1),
        layout="constrained",
    )
    for i, order in enumerate(orders):
        values = table[table.order == order].sort_values(["mean_rpm", "point"])
        ax.plot(
            values.mean_rpm,
            values.peak_amplitude,
            "o-",
            ms=4,
            lw=1,
            color=colors[order],
            label=f"{order:g}×",
        )
        row = axes.flat[i]
        row.plot(values.mean_rpm, values.peak_amplitude, "o-", ms=4, color=colors[order])
        for point in values.itertuples():
            if np.isfinite(point.peak_amplitude):
                row.annotate(
                    str(point.point),
                    (point.mean_rpm, point.peak_amplitude),
                    xytext=(3, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        row.set(
            title=f"{order:g}× shaft order",
            xlabel="Mean speed [RPM]",
            ylabel=f"FFT peak amplitude [{unit}]",
            ylim=(0, None),
        )
        row.grid(alpha=0.2)
    for unused in list(axes.flat)[len(orders) :]:
        unused.set_visible(False)
    ax.set(
        xlabel="Mean engine speed [RPM]",
        ylabel=f"Mean window FFT peak amplitude [{unit}]",
        ylim=(0, None),
        title=f"{title} | Order amplitude versus RPM\nRPM-guided local peak candidates · unfiltered",
    )
    ax.grid(alpha=0.2)
    ax.legend(title="Shaft order", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2)
    panel.suptitle(
        f"{title} | Order amplitudes (point numbers shown)\nMissing candidates are gaps; lines only guide the eye"
    )
    fig.savefig(directory / "harmonic_amplitude_vs_rpm.png", dpi=dpi, bbox_inches="tight")
    panel.savefig(directory / "harmonic_amplitude_vs_rpm_panels.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    plt.close(panel)


def overlay_orders(axis, table, orders, point, low, high):
    for i, order in enumerate(orders):
        row = table[(table.point == point) & (table.order == order)].iloc[0]
        color = f"C{i % 10}"
        if low <= row.expected_hz <= high:
            axis.axvline(row.expected_hz, color=color, ls="--", lw=0.7, alpha=0.5)
            axis.text(
                row.expected_hz,
                0.98,
                f"{order:g}×",
                color=color,
                fontsize=7,
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
            )
        if row.status == "candidate" and low <= row.peak_hz <= high:
            axis.plot(row.peak_hz, row.peak_amplitude, "o", ms=3, color=color)
