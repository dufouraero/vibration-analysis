"""Interactive views and portable HTML reports; no analysis on reduced display data."""

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .dashboard_data import display_indices

COLORS = ["#2563eb", "#e67722", "#0f8b78", "#9333ea", "#db2777", "#64748b", "#b45309"]


def empty_figure(message="Load a recording to begin"):
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(
        template="plotly_white", height=360, margin={"l": 55, "r": 25, "t": 45, "b": 45}
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def curve(x, y, low=None, high=None, budget=10000):
    first = 0 if low is None else max(0, int(np.searchsorted(x, low, side="left")))
    stop = len(x) if high is None else min(len(x), int(np.searchsorted(x, high, side="right")))
    if stop <= first:
        return np.array([]), np.array([])
    indices = display_indices(y, first, stop, budget)
    return np.asarray(x[indices]), np.asarray(y[indices])


def overview(dataset, channel, regions, bounds=None):
    recording = dataset.recording
    low, high = bounds or (0, dataset.duration)
    low, high = max(0, low), min(dataset.duration, high)
    traces = []
    if dataset.origin is not None:
        if dataset.raw_rpm is not None:
            traces.append(("Raw RPM", dataset.raw_rpm.seconds, dataset.raw_rpm.values, "RPM"))
        elif dataset.mrpm is not None:
            traces.append(("mRPM · detection source", *dataset.mrpm, "RPM"))
        if dataset.electrical is not None:
            traces.append(
                (
                    dataset.electrical.label,
                    dataset.electrical.seconds,
                    dataset.electrical.values,
                    dataset.electrical.label,
                )
            )
    figure = make_subplots(
        rows=1 + len(traces),
        cols=1,
        shared_xaxes=True,
        row_heights=[3] + [1] * len(traces),
        vertical_spacing=0.05,
    )
    x, y = curve(recording.time, recording.channels[channel], low, high, 14000)
    figure.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            name=channel,
            line={"color": COLORS[0], "width": 1},
            hovertemplate="%{x:.6f} s<br>%{y:.5g}<extra>%{fullData.name}</extra>",
        ),
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text=dataset.config["unit"], row=1, col=1)
    for index, (name, times, values, label) in enumerate(traces, 2):
        tx, ty = curve(times, values, low, high, 6000)
        if len(tx):
            positive = np.diff(times)
            cadence = np.median(positive[positive > 0]) if np.any(positive > 0) else np.inf
            gap_times = times[np.flatnonzero(positive > 3 * cadence) + 1]
            gaps = np.unique(np.searchsorted(tx, gap_times))
            gaps = gaps[(gaps > 0) & (gaps < len(tx))]
            tx, ty = np.insert(tx, gaps, np.nan), np.insert(ty.astype(float), gaps, np.nan)
        figure.add_trace(
            go.Scattergl(x=tx, y=ty, name=name, line={"color": COLORS[index - 1], "width": 1}),
            row=index,
            col=1,
        )
        figure.update_yaxes(title_text=label, row=index, col=1)
    for region in regions:
        if region.get("dataset") == dataset.id and region.get("channel") == channel:
            try:
                a, b = float(region["start"]), float(region["end"])
            except (ValueError, TypeError):
                continue
            figure.add_vrect(
                x0=a, x1=b, fillcolor="#38bdf8", opacity=0.12, line_width=0, row=1, col=1
            )
    figure.update_layout(
        template="plotly_white",
        height=400 + len(traces) * 110,
        dragmode="select",
        selectdirection="h",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02},
        margin={"l": 60, "r": 25, "t": 75, "b": 45},
        uirevision=f"{dataset.id}-{channel}",
        title={"text": dataset.name, "font": {"size": 15}},
    )
    figure.update_xaxes(range=[low, high])
    figure.update_xaxes(title_text="Source CSV elapsed time [s]", row=1 + len(traces), col=1)
    return figure


def result_label(result):
    return f"{result.region['recording']} · {result.region['name']}"


def visible_results(service, bundle_id, selected_ids):
    results = service.bundles.get(bundle_id, [])
    selected = set(selected_ids or [])
    return [result for result in results if result.region["id"] in selected]


def comparison_figures(service, results, display, spec_id=None):
    if not results:
        blank = empty_figure("Select regions and click Calculate / compare")
        return (
            {"fft": blank, "psd": blank, "spectrogram": blank, "waterfall": blank, "time": blank},
            [],
            "",
        )
    low = float(display.get("fmin") or 0)
    high = (
        max(r.summary["nyquist"] for r in results)
        if display.get("fmax") in (None, "")
        else float(display["fmax"])
    )
    if not np.isfinite([low, high]).all() or not 0 <= low < high:
        raise ValueError("Display limits must satisfy 0 ≤ frequency minimum < maximum.")
    stacked = display.get("fft_layout", "overlay") == "stacked"
    fft = make_subplots(
        rows=len(results) if stacked else 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[r.region["name"] for r in results] if stacked else None,
        vertical_spacing=min(0.08, 0.5 / len(results)),
    )
    psd, waterfall, times = go.Figure(), go.Figure(), go.Figure()
    rows, spectral_max = [], 0.0
    frequency_log = display.get("psd_axis") == "log"
    unit = results[0].summary["unit"]
    for index, result in enumerate(results):
        color = COLORS[index % len(COLORS)]
        label = result_label(result)
        x, y = curve(result.fft_frequency, result.fft_amplitude, low, high)
        if len(y):
            spectral_max = max(spectral_max, float(y.max()))
        fft.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                name=label,
                legendgroup=result.region["id"],
                line={"color": color, "width": 1},
            ),
            row=index + 1 if stacked else 1,
            col=1,
        )
        px, py = curve(
            result.frequency,
            result.psd,
            max(low, result.frequency[1]) if frequency_log else low,
            high,
        )
        psd.add_trace(
            go.Scattergl(
                x=px,
                y=np.maximum(py, 1e-300),
                name=label,
                legendgroup=result.region["id"],
                line={"color": color, "width": 1},
            )
        )
        waterfall.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full(len(x), index + 1),
                z=y,
                name=label,
                mode="lines",
                line={"color": color, "width": 3},
            )
        )
        dataset = service.datasets[result.region["dataset"]]
        relative = display.get("time_axis", "relative") == "relative"
        tx, ty = curve(
            dataset.recording.time,
            dataset.recording.channels[result.region["channel"]],
            result.summary["first_time"],
            result.summary["last_time"],
            10000,
        )
        if relative:
            tx = tx - result.summary["first_time"]
        times.add_trace(go.Scattergl(x=tx, y=ty, name=label, line={"color": color, "width": 0.8}))
        effective_high = min(high, result.summary["nyquist"])
        selected = (result.frequency >= low) & (result.frequency <= effective_high)
        band = (
            float(np.sqrt(result.psd[selected].sum() * result.summary["psd_bin_hz"]))
            if selected.any()
            else None
        )
        rows.append(
            {
                "id": result.region["id"],
                "region": result.region["name"],
                "recording": result.region["recording"],
                "seconds": round(result.summary["duration"], 3),
                "fs": round(result.summary["sample_rate"], 3),
                "rpm": round(result.summary["mean_rpm"])
                if result.summary.get("mean_rpm") is not None
                else None,
                "raw_rms": round(result.summary["raw_rms"], 5),
                "full_rms": round(result.summary["welch_rms"], 5),
                "band_rms": round(band, 5) if band is not None else None,
                "band": f"{low:g}–{effective_high:.6g} Hz"
                if effective_high > low
                else "Outside recording bandwidth",
                "peak": round(result.summary["peak"], 5),
                "threshold_percent": round(result.summary["threshold_percent"], 5),
                "threshold": result.summary["threshold"],
                "fft_bin": round(result.summary["fft_bin_hz"], 6),
                "fft_method": "Averaged window magnitudes"
                if result.settings["fft_mode"] == "averaged"
                else "Full-record FFT",
            }
        )
    for fig in (fft, psd, times):
        fig.update_layout(
            template="plotly_white",
            height=max(400, 230 * len(results)) if fig is fft and stacked else 450,
            legend={"orientation": "h", "y": -0.18},
            margin={"l": 65, "r": 25, "t": 60, "b": 100},
            hovermode="x unified",
            uirevision=f"comparison-{display.get('fft_layout')}",
        )
    fft.update_xaxes(title_text="Frequency [Hz]", range=[low, high])
    fft.update_yaxes(
        title_text=f"Peak amplitude [{unit}]", range=[0, max(spectral_max * 1.05, 1e-12)]
    )
    fft.update_layout(
        title="FFT · "
        + (
            "averaged window magnitudes"
            if results[0].settings["fft_mode"] == "averaged"
            else "full-record transform"
        )
    )
    psd.update_layout(title="Welch PSD · band RMS and full-band RMS are listed below")
    psd.update_xaxes(
        title_text="Frequency [Hz]",
        type="log" if frequency_log else "linear",
        range=[np.log10(max(low, min(r.frequency[1] for r in results))), np.log10(high)]
        if frequency_log
        else [low, high],
    )
    psd.update_yaxes(title_text=f"PSD [({unit})²/Hz]", type="log")
    times.update_layout(title="Original time histories · min/max display envelopes")
    times.update_xaxes(
        title_text="Time from region start [s]"
        if display.get("time_axis", "relative") == "relative"
        else "Source CSV elapsed time [s]"
    )
    times.update_yaxes(title_text=f"Signal [{unit}]")
    waterfall.update_layout(
        template="plotly_white",
        height=520,
        title="FFT waterfall by selected region",
        scene={
            "xaxis": {"title": "Frequency [Hz]", "range": [low, high]},
            "yaxis": {
                "title": "Region",
                "tickvals": list(range(1, len(results) + 1)),
                "ticktext": [r.region["name"] for r in results],
            },
            "zaxis_title": f"Peak amplitude [{unit}]",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    result = next((r for r in results if r.region["id"] == spec_id), results[0])
    density = result.spectrogram_psd
    db = 10 * np.log10(np.maximum(density, np.finfo(float).tiny))
    upper = (
        float(display["upper_db"])
        if display.get("upper_db") not in (None, "")
        else float(db.max() + 6)
    )
    lower = (
        float(display["lower_db"])
        if display.get("lower_db") not in (None, "")
        else upper - float(display.get("range_db") or 60)
    )
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Spectrogram lower colour limit must be below its upper limit.")
    spec = go.Figure(
        go.Heatmap(
            x=result.spectrogram_time,
            y=result.spectrogram_frequency,
            z=db.T.astype(np.float32),
            zmin=lower,
            zmax=upper,
            colorscale="Viridis",
            colorbar={"title": "PSD [dB]"},
            hovertemplate="%{x:.3f} s<br>%{y:.2f} Hz<br>%{z:.2f} dB<extra></extra>",
        )
    )
    spec.update_layout(
        template="plotly_white",
        title=f"Spectrogram · {result.region['name']}",
        height=470,
        xaxis_title="Source CSV elapsed time [s]",
        yaxis={"title": "Frequency [Hz]", "range": [low, min(high, result.summary["nyquist"])]},
        margin={"l": 65, "r": 25, "t": 65, "b": 55},
        uirevision=result.region["id"],
    )
    nyquists = [r.summary["nyquist"] for r in results]
    note = "Spectra use original samples. Overview lines preserve bucket minima/maxima; spectrogram display cells average linear PSD. "
    note += "Peak threshold percentage is not an automatic diagnosis of clipping. "
    if max(nyquists) > min(nyquists) * 1.01 and high > min(nyquists):
        note += f"Different bandwidths: some records stop at {min(nyquists):.3f} Hz. Use Common bandwidth for equal-band RMS. "
    methods = {r.settings["fft_mode"] for r in results}
    if len(methods) > 1:
        note += (
            "Mixed FFT methods; recalculate with common settings before comparing peak heights. "
        )
    return (
        {"fft": fft, "psd": psd, "spectrogram": spec, "waterfall": waterfall, "time": times},
        rows,
        note,
    )


def export_report(service, results, figures, statistics, session, output_root):
    directory = Path(output_root) / datetime.now().strftime("comparison_%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "session.json").write_text(
        json.dumps(session, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(statistics).to_csv(directory / "statistics.csv", index=False)
    sections = []
    for index, (name, figure) in enumerate(figures.items()):
        sections.append(
            f"<h2>{html.escape(name.capitalize())}</h2>"
            + pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True},
                validate=True,
            )
        )
    for index, result in enumerate(results, 1):
        np.savez_compressed(
            directory / f"region_{index:02d}_spectra.npz",
            fft_frequency_hz=result.fft_frequency,
            fft_peak_amplitude=result.fft_amplitude,
            frequency_hz=result.frequency,
            psd=result.psd,
            spectrogram_frequency_hz=result.spectrogram_frequency,
            spectrogram_time_s=result.spectrogram_time,
            spectrogram_display_psd=result.spectrogram_psd,
        )
        (directory / f"region_{index:02d}_summary.json").write_text(
            json.dumps(
                {"region": result.region, "settings": result.settings, "summary": result.summary},
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    table = pd.DataFrame(statistics).to_html(index=False, escape=True)
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Vibration comparison</title>"
    )
    document += "<style>body{font:15px system-ui;margin:32px;color:#152238}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #ddd}.table{overflow:auto}</style></head><body>"
    document += "<h1>Vibration comparison</h1><p>Interactive snapshot: zoom and hide/show traces. Recalculating different time regions requires the local dashboard and original source files. Numerical exports and session settings accompany this report.</p>"
    document += "<div class='table'>" + table + "</div>" + "".join(sections) + "</body></html>"
    path = directory / "comparison.html"
    path.write_text(document, encoding="utf-8")
    return path
