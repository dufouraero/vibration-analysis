"""Local interactive analysis workspace. Start with ``vibration-dashboard``."""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from werkzeug.serving import make_server

from .dashboard_service import DashboardService, validate_region
from .dashboard_views import (
    comparison_figures,
    empty_figure,
    export_report,
    overview,
    visible_results,
)

DEFAULTS = {
    "input": "",
    "name": "",
    "mrpm_csv": "",
    "raw_rpm_csv": "",
    "electrical_csv": "",
    "header": "auto",
    "time_column": "",
    "time_unit": "s",
    "sample_rate": None,
    "channels": "",
    "delimiter": "auto",
    "decimal": ".",
    "skiprows": 0,
    "scale": 1,
    "unit": "g",
    "active_channels": True,
    "vibration_start": "",
    "timezone": "Europe/Zurich",
    "alignment": "Auto",
    "offset": 0,
    "mrpm_time_column": "",
    "mrpm_column": "",
    "raw_rpm_time_column": "",
    "raw_rpm_column": "",
    "electrical_time_column": "",
    "electrical_column": "",
    "electrical_quantity": "auto",
    "running_rpm": 500,
    "minimum_run": 30,
    "minimum_hold": 30,
    "stability_window": 20,
    "slope_limit": 2,
    "scatter_limit": 40,
    "trim_start": 5,
    "trim_end": 5,
    "segment_seconds": 0.5,
    "overlap": 0.5,
    "window": "hann",
    "detrend": "constant",
    "highpass_hz": 10,
    "filter_order": 4,
    "spectrogram_seconds": 0.1,
    "spectrogram_overlap": 0.8,
    "fft_mode": "averaged",
    "clip_level": 50,
    "region_name": "",
    "region_start": 0,
    "region_end": None,
    "session_path": "",
}
ANALYSIS_KEYS = (
    "segment_seconds",
    "overlap",
    "window",
    "detrend",
    "highpass_hz",
    "filter_order",
    "spectrogram_seconds",
    "spectrogram_overlap",
    "fft_mode",
    "clip_level",
)
DISPLAY = {
    "fmin": 0,
    "fmax": 1000,
    "fft_layout": "overlay",
    "psd_axis": "linear",
    "time_axis": "relative",
    "range_db": 60,
    "lower_db": None,
    "upper_db": None,
}
FILE_KEYS = ("input", "mrpm_csv", "raw_rpm_csv", "electrical_csv", "session_path")
GRAPH_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def axis_bounds(relayout):
    """Plotly reports the axis of whichever shared row the user zoomed."""
    for key, value in (relayout or {}).items():
        if key.startswith("xaxis") and key.endswith(".range[0]"):
            other = key[:-3] + "[1]"
            if other in relayout:
                return float(value), float(relayout[other])
        if key.startswith("xaxis") and key.endswith(".range") and len(value) == 2:
            return float(value[0]), float(value[1])
    return None


def field(key, label, *, options=None, kind="text", placeholder=None):
    identifier = {"type": "field", "name": key}
    if options is not None:
        widget = dcc.Dropdown(
            id=identifier,
            options=[{"label": name, "value": value} for name, value in options],
            value=DEFAULTS[key],
            clearable=False,
        )
    else:
        widget = dcc.Input(
            id=identifier,
            type=kind,
            value=DEFAULTS[key],
            placeholder=placeholder,
            debounce=True,
            className="field-input",
        )
    children = [
        html.Label(label, htmlFor=json.dumps(identifier, sort_keys=True, separators=(",", ":"))),
        widget,
    ]
    if key in FILE_KEYS:
        children = [
            html.Label(
                label, htmlFor=json.dumps(identifier, sort_keys=True, separators=(",", ":"))
            ),
            html.Div(
                [widget, html.Button("Browse", id=f"browse-{key}", className="browse")],
                className="file-field",
            ),
        ]
    return html.Div(children, className="field")


def number(key, label):
    return field(key, label, kind="number")


def _native_file_picker():
    code = "import tkinter as tk; from tkinter import filedialog; r=tk.Tk(); r.withdraw(); p=filedialog.askopenfilename(filetypes=[('CSV or session','*.csv *.json'),('All files','*.*')]); print(p); r.destroy()"
    executable = Path(sys.executable)
    if os.name == "nt":
        executable = executable.with_name("python.exe")
    result = subprocess.run(
        [str(executable), "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        check=True,
    )
    return result.stdout.strip()


def demo(service, progress):
    """A repeatable, local example with two holds and separate telemetry sources."""
    root = service.cache_root / "demo"
    root.mkdir(parents=True, exist_ok=True)
    source = root / "Demo_2026-01-01T12-00-00-000.csv"
    mrpm, raw, volts = root / "mrpm.csv", root / "raw_rpm.csv", root / "voltage.csv"
    if not source.exists():
        progress("Creating synthetic example…")
        fs = 2048
        t = np.arange(fs * 180) / fs
        rpm = np.where(
            t < 20,
            0,
            np.where(
                t < 75, 4500, np.where(t < 90, 4500 + (t - 75) * 60, np.where(t < 155, 5400, 0))
            ),
        )
        phase = np.cumsum(rpm / 60) * 2 * np.pi / fs
        active = rpm > 0
        rng = np.random.default_rng(10)
        values = active * (
            1.2 * np.sin(phase)
            + 0.6 * np.sin(3 * phase)
            + np.where(t < 90, 0.2, 0.7) * np.sin(2 * np.pi * 700 * t)
        ) + 0.025 * rng.normal(size=len(t))
        pd.DataFrame({"time_s": t, "Acceleration": values}).to_csv(source, index=False)
        rt = np.arange(0, 180, 1.0)
        speed = np.interp(rt, t, rpm)
        stamps = pd.Timestamp("2026-01-01 12:00:00") + pd.to_timedelta(rt, unit="s")
        pd.DataFrame({"Time": stamps, "val_mrpm": speed}).to_csv(mrpm, index=False)
        pd.DataFrame(
            {"Time": stamps, "val_rpm": np.maximum(0, speed + (speed > 0) * 15 * np.sin(rt))}
        ).to_csv(raw, index=False)
        pd.DataFrame(
            {"Time": stamps, "voltage": 57.5 - (speed > 0) * 0.5 - (speed > 5000) * 0.4}
        ).to_csv(volts, index=False)
    return service.load(
        {
            **DEFAULTS,
            "input": str(source),
            "name": "Synthetic engine demo",
            "mrpm_csv": str(mrpm),
            "raw_rpm_csv": str(raw),
            "electrical_csv": str(volts),
            "alignment": "Manual",
        },
        progress,
    )


def create_app(cache_root=None, service=None):
    service = service or DashboardService(cache_root or Path.cwd() / ".dashboard-cache")
    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).with_name("dashboard_assets")),
        title="Vibration workspace",
        update_title=None,
    )
    app.vibration_service = service
    app.index_string = app.index_string.replace(
        "{%favicon%}", '<link rel="icon" type="image/svg+xml" href="/assets/wave.svg">'
    )
    app.layout = html.Div(
        [
            dcc.Store(id="job"),
            dcc.Store(id="bundle"),
            dcc.Store(id="restored"),
            dcc.Download(id="download"),
            dcc.Interval(id="poll", interval=900, disabled=True),
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("∿", className="brand-mark"),
                            html.Div(
                                [
                                    html.H1("Vibration workspace"),
                                    html.P("Select the signal. Compare the response."),
                                ]
                            ),
                        ],
                        className="brand",
                    ),
                    html.Div(
                        [
                            html.Span("LOCAL ANALYSIS", className="pill"),
                            html.Button("Load demo", id="demo", className="secondary"),
                            html.Button("Stop dashboard", id="stop", className="secondary"),
                        ],
                        className="header-actions",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Aside(
                        [
                            html.Div(
                                [html.Span("01", className="step"), html.H2("Recordings")],
                                className="section-title",
                            ),
                            field("input", "Vibration CSV", placeholder="Paste a local CSV path"),
                            field("name", "Recording label", placeholder="e.g. Day 2 · Y axis"),
                            field("mrpm_csv", "mRPM CSV · detection / alignment"),
                            field("raw_rpm_csv", "Raw RPM CSV · time plot only"),
                            field("electrical_csv", "Voltage / current CSV · time plot only"),
                            html.Button("Load recording", id="load", className="primary full"),
                            html.Details(
                                [
                                    html.Summary("CSV and calibration"),
                                    field(
                                        "header",
                                        "CSV header",
                                        options=[
                                            ("Detect automatically", "auto"),
                                            ("Header present", "yes"),
                                            ("No header", "no"),
                                        ],
                                    ),
                                    field("time_column", "Time column · blank = auto"),
                                    field(
                                        "time_unit",
                                        "Time unit",
                                        options=[
                                            (v, v) for v in ["s", "ms", "us", "ns", "datetime"]
                                        ],
                                    ),
                                    number("sample_rate", "Sample rate [Hz] · only if no time"),
                                    field("channels", "Sensor columns · comma separated"),
                                    field(
                                        "active_channels",
                                        "Initial sensor selection",
                                        options=[
                                            ("Suggest strongest activity", True),
                                            ("First selected column", False),
                                        ],
                                    ),
                                    field(
                                        "delimiter",
                                        "Delimiter",
                                        options=[
                                            ("Auto", "auto"),
                                            ("Comma", ","),
                                            ("Semicolon", ";"),
                                            ("Tab", "\t"),
                                        ],
                                    ),
                                    field(
                                        "decimal",
                                        "Decimal separator",
                                        options=[("Point", "."), ("Comma", ",")],
                                    ),
                                    number("skiprows", "Metadata rows to skip"),
                                    number("scale", "Calibration multiplier"),
                                    field("unit", "Calibrated signal unit"),
                                ]
                            ),
                            html.Details(
                                [
                                    html.Summary("mRPM alignment and hold suggestions"),
                                    field("vibration_start", "Vibration start · blank = filename"),
                                    field("timezone", "Grafana display timezone"),
                                    field(
                                        "alignment",
                                        "Alignment",
                                        options=[
                                            ("Automatic start/stop matching", "Auto"),
                                            ("Manual offset", "Manual"),
                                        ],
                                    ),
                                    number("offset", "Manual offset added to telemetry [s]"),
                                    field("mrpm_time_column", "mRPM time column · auto if blank"),
                                    field("mrpm_column", "mRPM value column · auto if blank"),
                                    number("running_rpm", "Running threshold [RPM]"),
                                    number("minimum_run", "Minimum test [s]"),
                                    number("minimum_hold", "Minimum hold [s]"),
                                    number("stability_window", "Stability window [s]"),
                                    number("slope_limit", "Maximum drift [RPM/s]"),
                                    number("scatter_limit", "Maximum scatter [RPM RMS]"),
                                    number("trim_start", "Trim suggested hold start [s]"),
                                    number("trim_end", "Trim suggested hold end [s]"),
                                    html.P(
                                        "mRPM proposes regions. Review the boundaries; raw RPM never drives detection.",
                                        className="hint",
                                    ),
                                ]
                            ),
                            html.Details(
                                [
                                    html.Summary("Raw telemetry columns"),
                                    field("raw_rpm_time_column", "Raw RPM time column"),
                                    field("raw_rpm_column", "Raw RPM value column"),
                                    field("electrical_time_column", "Electrical time column"),
                                    field("electrical_column", "Electrical value column"),
                                    field(
                                        "electrical_quantity",
                                        "Electrical quantity",
                                        options=[
                                            ("Auto", "auto"),
                                            ("Voltage [V]", "voltage"),
                                            ("Current [A]", "current"),
                                        ],
                                    ),
                                ]
                            ),
                            html.Details(
                                [
                                    html.Summary("Windowing and filtering"),
                                    field(
                                        "fft_mode",
                                        "FFT method",
                                        options=[
                                            ("Average short-window magnitudes", "averaged"),
                                            ("One full-record FFT", "full"),
                                        ],
                                    ),
                                    number("segment_seconds", "Welch / FFT window [s]"),
                                    number("overlap", "Overlap fraction"),
                                    field(
                                        "window",
                                        "Window",
                                        options=[
                                            (v.title(), v)
                                            for v in ["hann", "hamming", "flattop", "boxcar"]
                                        ],
                                    ),
                                    field(
                                        "detrend",
                                        "Detrend",
                                        options=[
                                            (v.title(), v) for v in ["constant", "linear", "none"]
                                        ],
                                    ),
                                    number("highpass_hz", "PSD high-pass [Hz] · 0 = off"),
                                    number("filter_order", "Butterworth order per pass"),
                                    number("spectrogram_seconds", "Spectrogram window [s]"),
                                    number("spectrogram_overlap", "Spectrogram overlap"),
                                    number("clip_level", "Peak threshold [signal unit]"),
                                    html.P(
                                        "High-pass applies to PSD and spectrogram only. Change settings, then recalculate. Full-record FFTs over very long regions require more memory.",
                                        className="hint",
                                    ),
                                ]
                            ),
                            html.Details(
                                [
                                    html.Summary("Save or restore a workspace"),
                                    html.Button(
                                        "Download session JSON",
                                        id="save",
                                        className="secondary full",
                                    ),
                                    field("session_path", "Session JSON to restore"),
                                    html.Button(
                                        "Restore session", id="restore", className="secondary full"
                                    ),
                                    html.P(
                                        "A session stores paths, regions and settings. Original CSVs are required to restore it.",
                                        className="hint",
                                    ),
                                ]
                            ),
                        ],
                        className="sidebar",
                    ),
                    html.Main(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        id="status",
                                        children="Load your recording, or try the synthetic demo.",
                                        role="status",
                                    ),
                                    html.Button("Cancel", id="cancel", className="quiet"),
                                ],
                                className="status-bar",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("02", className="step"),
                                                    html.H2("Explore and select"),
                                                ],
                                                className="section-title",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Dropdown(
                                                        id="dataset",
                                                        options=[],
                                                        placeholder="Recording",
                                                        className="recording-select",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="channel",
                                                        options=[],
                                                        placeholder="Sensor",
                                                        className="channel-select",
                                                    ),
                                                ],
                                                className="recording-controls",
                                            ),
                                        ],
                                        className="section-heading",
                                    ),
                                    html.Div(id="dataset-info", className="dataset-info"),
                                    dcc.Graph(
                                        id="overview", figure=empty_figure(), config=GRAPH_CONFIG
                                    ),
                                    dcc.RadioItems(
                                        id="overview-tool",
                                        options=[
                                            {"label": "Select interval", "value": "select"},
                                            {"label": "Zoom", "value": "zoom"},
                                            {"label": "Pan", "value": "pan"},
                                        ],
                                        value="select",
                                        inline=True,
                                        className="time-choice",
                                    ),
                                    html.P(
                                        "Drag a selection across the time history to fill Start / End. Use the toolbar to zoom or pan; zooming redraws original sample extrema. Telemetry remains unfiltered.",
                                        className="hint",
                                    ),
                                    html.Div(
                                        [
                                            field(
                                                "region_name",
                                                "Region label",
                                                placeholder="e.g. 4,500 RPM · warm",
                                            ),
                                            number("region_start", "Start [s]"),
                                            number("region_end", "End [s]"),
                                            html.Button(
                                                "Add region", id="add", className="primary"
                                            ),
                                            html.Button(
                                                "Suggest mRPM holds",
                                                id="suggest",
                                                className="secondary",
                                            ),
                                        ],
                                        className="region-controls",
                                    ),
                                ],
                                className="card",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("03", className="step"),
                                                    html.H2("Regions to compare"),
                                                ],
                                                className="section-title",
                                            ),
                                            html.Button(
                                                "Calculate / compare",
                                                id="calculate",
                                                className="primary",
                                            ),
                                        ],
                                        className="section-heading",
                                    ),
                                    html.P(
                                        "Load more recordings to compare days or configurations. Tick regions; edit labels and bounds directly in the table. Analysis always uses original samples.",
                                        className="hint",
                                    ),
                                    dash_table.DataTable(
                                        id="regions",
                                        columns=[
                                            {"name": "Region", "id": "name", "editable": True},
                                            {"name": "Recording", "id": "recording"},
                                            {"name": "Sensor", "id": "channel"},
                                            {
                                                "name": "Start [s]",
                                                "id": "start",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                            {
                                                "name": "End [s]",
                                                "id": "end",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                        ],
                                        data=[],
                                        row_selectable="multi",
                                        selected_row_ids=[],
                                        row_deletable=True,
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "fontFamily": "inherit",
                                            "fontSize": 13,
                                            "padding": "10px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "backgroundColor": "#f1f5f9",
                                            "fontWeight": 600,
                                        },
                                        page_size=12,
                                    ),
                                ],
                                className="card",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("04", className="step"),
                                                    html.H2("Compare the response"),
                                                ],
                                                className="section-title",
                                            ),
                                            html.Button(
                                                "Export interactive HTML",
                                                id="export",
                                                className="secondary",
                                            ),
                                        ],
                                        className="section-heading",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Frequency min [Hz]"),
                                                    dcc.Input(
                                                        id="fmin",
                                                        type="number",
                                                        value=0,
                                                        debounce=True,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("Frequency max [Hz]"),
                                                    dcc.Input(
                                                        id="fmax",
                                                        type="number",
                                                        value=1000,
                                                        debounce=True,
                                                    ),
                                                ]
                                            ),
                                            html.Button(
                                                "Common bandwidth", id="common", className="quiet"
                                            ),
                                            html.Button(
                                                "Broadband", id="broadband", className="quiet"
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("FFT layout"),
                                                    dcc.Dropdown(
                                                        id="fft_layout",
                                                        options=[
                                                            {
                                                                "label": "Overlay",
                                                                "value": "overlay",
                                                            },
                                                            {
                                                                "label": "Stacked rows",
                                                                "value": "stacked",
                                                            },
                                                        ],
                                                        value="overlay",
                                                        clearable=False,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("PSD frequency axis"),
                                                    dcc.Dropdown(
                                                        id="psd_axis",
                                                        options=[
                                                            {"label": "Linear", "value": "linear"},
                                                            {
                                                                "label": "Logarithmic",
                                                                "value": "log",
                                                            },
                                                        ],
                                                        value="linear",
                                                        clearable=False,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="display-controls",
                                    ),
                                    html.Div(id="result-note", className="hint"),
                                    dcc.Tabs(
                                        id="result-tab",
                                        value="psd",
                                        children=[
                                            dcc.Tab(
                                                label="PSD",
                                                value="psd",
                                                children=[
                                                    dcc.Graph(
                                                        id="psd",
                                                        figure=empty_figure(),
                                                        config=GRAPH_CONFIG,
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="FFT",
                                                value="fft",
                                                children=[
                                                    dcc.Graph(
                                                        id="fft",
                                                        figure=empty_figure(),
                                                        config=GRAPH_CONFIG,
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Waterfall",
                                                value="waterfall",
                                                children=[
                                                    dcc.Graph(
                                                        id="waterfall",
                                                        figure=empty_figure(),
                                                        config=GRAPH_CONFIG,
                                                    )
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Spectrogram",
                                                value="spectrogram",
                                                children=[
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id="spec-region",
                                                                options=[],
                                                                placeholder="Choose a region",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label("Colour range [dB]"),
                                                                    dcc.Input(
                                                                        id="range_db",
                                                                        type="number",
                                                                        value=60,
                                                                        debounce=True,
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Lower PSD [dB] · auto if blank"
                                                                    ),
                                                                    dcc.Input(
                                                                        id="lower_db",
                                                                        type="number",
                                                                        debounce=True,
                                                                    ),
                                                                ]
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Upper PSD [dB] · auto if blank"
                                                                    ),
                                                                    dcc.Input(
                                                                        id="upper_db",
                                                                        type="number",
                                                                        debounce=True,
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        className="display-controls",
                                                    ),
                                                    dcc.Graph(
                                                        id="spectrogram",
                                                        figure=empty_figure(),
                                                        config=GRAPH_CONFIG,
                                                    ),
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Time histories",
                                                value="time",
                                                children=[
                                                    dcc.RadioItems(
                                                        id="time_axis",
                                                        options=[
                                                            {
                                                                "label": " Align region starts",
                                                                "value": "relative",
                                                            },
                                                            {
                                                                "label": " Source CSV time",
                                                                "value": "source",
                                                            },
                                                        ],
                                                        value="relative",
                                                        inline=True,
                                                        className="time-choice",
                                                    ),
                                                    dcc.Graph(
                                                        id="time",
                                                        figure=empty_figure(),
                                                        config=GRAPH_CONFIG,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.H3("Levels and analysis bandwidth"),
                                    dash_table.DataTable(
                                        id="statistics",
                                        data=[],
                                        columns=[
                                            {"name": label, "id": key}
                                            for key, label in [
                                                ("region", "Region"),
                                                ("recording", "Recording"),
                                                ("seconds", "Duration [s]"),
                                                ("rpm", "mRPM mean"),
                                                ("fs", "Sample rate [Hz]"),
                                                ("raw_rms", "Raw RMS"),
                                                ("full_rms", "PSD full RMS"),
                                                ("band_rms", "PSD band RMS"),
                                                ("band", "Actual RMS band"),
                                                ("peak", "Raw |peak|"),
                                                ("threshold_percent", "≥ threshold [%]"),
                                                ("threshold", "Threshold"),
                                                ("fft_bin", "FFT bin [Hz]"),
                                                ("fft_method", "FFT method"),
                                            ]
                                        ],
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "fontFamily": "inherit",
                                            "fontSize": 12,
                                            "padding": "9px",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "backgroundColor": "#f1f5f9",
                                            "fontWeight": 600,
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {
                                                    "filter_query": "{threshold_percent} > 0",
                                                    "column_id": "threshold_percent",
                                                },
                                                "backgroundColor": "#fff1df",
                                                "color": "#9a3412",
                                            }
                                        ],
                                    ),
                                ],
                                className="card",
                            ),
                        ],
                        className="workspace",
                    ),
                ],
                className="app-grid",
            ),
        ]
    )

    for key in FILE_KEYS:

        def register_picker(key):
            @app.callback(
                Output({"type": "field", "name": key}, "value", allow_duplicate=True),
                Input(f"browse-{key}", "n_clicks"),
                prevent_initial_call=True,
            )
            def pick(_):
                return _native_file_picker() or no_update

        register_picker(key)

    @app.callback(
        Output("job", "data"),
        Output("status", "children"),
        Output("dataset", "options"),
        Output("dataset", "value"),
        Output("regions", "data"),
        Output("regions", "selected_row_ids"),
        Output("download", "data"),
        Output("bundle", "data"),
        Output("restored", "data"),
        Output("regions", "selected_rows"),
        *[
            Input(key, "n_clicks")
            for key in [
                "load",
                "demo",
                "add",
                "suggest",
                "calculate",
                "save",
                "restore",
                "export",
                "cancel",
            ]
        ],
        Input("poll", "n_intervals"),
        State({"type": "field", "name": ALL}, "value"),
        State({"type": "field", "name": ALL}, "id"),
        State("dataset", "value"),
        State("channel", "value"),
        State("regions", "data"),
        State("regions", "selected_row_ids"),
        State("job", "data"),
        State("bundle", "data"),
        *[State(key, "value") for key in DISPLAY],
        prevent_initial_call=True,
    )
    def manage(*args):
        values, identifiers, active, channel, rows, selected, job_id, bundle_id = args[10:18]
        config = {**DEFAULTS, **{item["name"]: value for item, value in zip(identifiers, values)}}
        display = dict(zip(DISPLAY, args[18:]))
        processing = {key: config[key] for key in ANALYSIS_KEYS}
        action = ctx.triggered_id
        result = [no_update] * 10

        def finish():
            if result[4] is not no_update or result[5] is not no_update:
                table = rows if result[4] is no_update else result[4]
                checked = selected if result[5] is no_update else result[5]
                result[9] = [i for i, row in enumerate(table or []) if row["id"] in (checked or [])]
            return result

        try:
            if action == "poll":
                if not job_id:
                    return finish()
                job = service.job(job_id)
                result[1] = job["message"]
                if job["state"] == "running":
                    return finish()
                result[0] = None
                if job["state"] == "error":
                    return finish()
                if job["kind"] in ("load", "demo"):
                    result[2], result[3] = service.options(), job["result"]
                    dataset = service.datasets[job["result"]]
                    result[1] = (
                        f"Loaded {dataset.name}: {len(dataset.recording.time):,} original samples. "
                        + " ".join(dataset.notes)
                    )
                    if job["kind"] == "demo":
                        suggested = service.suggest(dataset.id, DEFAULTS)
                        result[4] = (rows or []) + suggested
                        result[5] = (selected or []) + [r["id"] for r in suggested]
                elif job["kind"] == "calculate":
                    result[7] = job["result"]
                    result[1] = (
                        "Comparison ready. Hide/show traces in the legend, change layouts, or adjust the frequency range."
                    )
                elif job["kind"] == "restore":
                    restored = job["result"]
                    result[2], result[3] = service.options(), restored["dataset"]
                    result[4], result[5] = (
                        restored["regions"],
                        [r["id"] for r in restored["regions"]],
                    )
                    result[8] = restored
                    result[1] = (
                        "Session restored. Click Calculate / compare to recompute from the original samples."
                    )
                elif job["kind"] == "export":
                    result[6] = dcc.send_file(job["result"])
                    result[1] = f"Export saved: {job['result']}"
                return finish()
            if action == "cancel":
                if job_id:
                    service.cancel(job_id)
                    result[1] = "Cancellation requested; stopping at the next processing boundary…"
                return finish()
            if job_id:
                result[1] = "Processing is running. Wait for completion or cancel it first."
                return finish()
            if action in ("load", "demo", "restore"):
                if action == "load":
                    result[0] = service.submit("load", service.load, config)
                elif action == "demo":
                    result[0] = service.submit("demo", lambda progress: demo(service, progress))
                else:
                    result[0] = service.submit("restore", service.restore, config["session_path"])
                result[1] = "Starting…"
            elif action in ("add", "suggest"):
                if active not in service.datasets:
                    raise ValueError("Load and choose a recording first.")
                dataset = service.datasets[active]
                additions = (
                    service.suggest(active, config)
                    if action == "suggest"
                    else [
                        validate_region(
                            dataset,
                            {
                                "channel": channel,
                                "name": config["region_name"] or f"Region {len(rows or []) + 1}",
                                "start": config["region_start"],
                                "end": config["region_end"],
                            },
                        )
                    ]
                )
                if action == "suggest":
                    for row in additions:
                        row["channel"] = channel or dataset.default_channel
                    additions = [
                        r
                        for r in additions
                        if not any(
                            r["dataset"] == old.get("dataset")
                            and r["channel"] == old.get("channel")
                            and r["start"] == old.get("start")
                            and r["end"] == old.get("end")
                            for old in rows or []
                        )
                    ]
                result[4], result[5] = (
                    (rows or []) + additions,
                    (selected or []) + [r["id"] for r in additions],
                )
                result[1] = (
                    f"Added {len(additions)} region(s). Review/edit the table, then calculate."
                )
            elif action == "calculate":
                chosen = [r for r in rows or [] if r["id"] in (selected or [])]
                result[0] = service.submit("calculate", service.compare, chosen, processing)
                result[1] = "Recalculating selected regions from original samples…"
            elif action == "save":
                payload = service.session(rows or [], processing, display)
                result[6] = dcc.send_string(
                    json.dumps(payload, indent=2, allow_nan=False),
                    f"vibration_session_{datetime.now():%Y%m%d_%H%M%S}.json",
                )
                result[1] = "Session downloaded: paths, regions and settings."
            elif action == "export":
                chosen = visible_results(service, bundle_id, selected)
                if not chosen:
                    raise ValueError("Calculate selected regions before exporting.")

                def export(progress):
                    progress("Building portable interactive report and numerical exports…")
                    figures, statistics, _ = comparison_figures(service, chosen, display)
                    if active in service.datasets and channel:
                        figures = {
                            "overview": overview(service.datasets[active], channel, rows or []),
                            **figures,
                        }
                    payload = service.session(
                        [r.region for r in chosen], chosen[0].settings, display
                    )
                    return str(
                        export_report(
                            service,
                            chosen,
                            figures,
                            statistics,
                            payload,
                            Path.cwd() / "results" / "dashboard",
                        )
                    )

                result[0] = service.submit("export", export)
                result[1] = "Exporting…"
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result[1] = f"Could not complete: {exc}"
        return finish()

    @app.callback(
        Output("channel", "options"),
        Output("channel", "value"),
        Output("dataset-info", "children"),
        Input("dataset", "value"),
    )
    def choose_dataset(identifier):
        if identifier not in service.datasets:
            return [], None, ""
        dataset = service.datasets[identifier]
        text = f"{dataset.recording.sample_rate:,.4f} samples/s · {dataset.duration / 60:.2f} min · {len(dataset.recording.channels)} sensor(s) · {dataset.config['unit']}"
        return (
            [{"label": name, "value": name} for name in dataset.recording.channels],
            dataset.default_channel,
            text,
        )

    @app.callback(
        Output("overview", "figure"),
        Input("dataset", "value"),
        Input("channel", "value"),
        Input("regions", "data"),
        Input("overview", "relayoutData"),
        Input("overview-tool", "value"),
    )
    def draw_overview(identifier, channel, rows, relayout, tool):
        if (
            identifier not in service.datasets
            or channel not in service.datasets[identifier].recording.channels
        ):
            return empty_figure()
        bounds = None
        if ctx.triggered_id in ("overview", "overview-tool"):
            bounds = axis_bounds(relayout)
        figure = overview(service.datasets[identifier], channel, rows or [], bounds)
        return figure.update_layout(dragmode=tool)

    @app.callback(
        Output({"type": "field", "name": "region_start"}, "value", allow_duplicate=True),
        Output({"type": "field", "name": "region_end"}, "value", allow_duplicate=True),
        Input("overview", "selectedData"),
        Input("dataset", "value"),
        prevent_initial_call=True,
    )
    def pick_region(selection, identifier):
        if ctx.triggered_id == "dataset" and identifier in service.datasets:
            return 0, service.datasets[identifier].duration
        if selection and "range" in selection and "x" in selection["range"]:
            a, b = selection["range"]["x"]
            dataset = service.datasets.get(identifier)
            if dataset:
                return max(0, float(a)), min(dataset.duration, float(b))
        return no_update, no_update

    restore_outputs = [
        Output({"type": "field", "name": key}, "value", allow_duplicate=True) for key in DEFAULTS
    ]
    restore_outputs += [Output(key, "value", allow_duplicate=True) for key in DISPLAY]

    @app.callback(*restore_outputs, Input("restored", "data"), prevent_initial_call=True)
    def restore_fields(restored):
        dataset = service.datasets[restored["dataset"]]
        values = {**DEFAULTS, **dataset.config, **restored["settings"]}
        values["region_start"], values["region_end"] = 0, dataset.duration
        if isinstance(values.get("channels"), list):
            values["channels"] = ", ".join(values["channels"])
        return [values.get(key) for key in DEFAULTS] + [
            {**DISPLAY, **restored["display"]}.get(key) for key in DISPLAY
        ]

    @app.callback(
        Output("spec-region", "options"),
        Output("spec-region", "value"),
        Input("bundle", "data"),
        Input("regions", "selected_row_ids"),
        State("spec-region", "value"),
    )
    def spec_options(bundle, selected, current):
        results = visible_results(service, bundle, selected)
        ids = [r.region["id"] for r in results]
        return [
            {"label": r.region["name"], "value": r.region["id"]} for r in results
        ], current if current in ids else next(iter(ids), None)

    @app.callback(
        *[Output(key, "figure") for key in ["fft", "psd", "spectrogram", "waterfall", "time"]],
        Output("statistics", "data"),
        Output("result-note", "children"),
        Input("bundle", "data"),
        Input("regions", "selected_row_ids"),
        Input("spec-region", "value"),
        *[Input(key, "value") for key in DISPLAY],
    )
    def draw_results(bundle, selected, spec, *values):
        results = visible_results(service, bundle, selected)
        try:
            figures, rows, note = comparison_figures(
                service, results, dict(zip(DISPLAY, values)), spec
            )
            return [figures[key] for key in ["fft", "psd", "spectrogram", "waterfall", "time"]] + [
                rows,
                note,
            ]
        except (ValueError, TypeError) as exc:
            return [empty_figure(str(exc))] * 5 + [[], str(exc)]

    @app.callback(
        Output("fmin", "value", allow_duplicate=True),
        Output("fmax", "value", allow_duplicate=True),
        Input("common", "n_clicks"),
        Input("broadband", "n_clicks"),
        Input("fft", "relayoutData"),
        Input("psd", "relayoutData"),
        State("bundle", "data"),
        State("regions", "selected_row_ids"),
        State("psd_axis", "value"),
        prevent_initial_call=True,
    )
    def frequency_range(_, __, fft_zoom, psd_zoom, bundle, selected, psd_axis):
        results = visible_results(service, bundle, selected)
        if not results:
            return no_update, no_update
        if ctx.triggered_id == "common":
            return 0, min(r.summary["nyquist"] for r in results)
        if ctx.triggered_id == "broadband":
            return 0, max(r.summary["nyquist"] for r in results)
        zoom = fft_zoom if ctx.triggered_id == "fft" else psd_zoom
        bounds = axis_bounds(zoom)
        if bounds:
            low, high = bounds
            if ctx.triggered_id == "psd" and psd_axis == "log":
                low, high = 10**low, 10**high
            return max(0, low), high
        return no_update, no_update

    @app.callback(
        *[
            Output(key, "disabled")
            for key in ["load", "demo", "calculate", "restore", "export", "suggest", "add"]
        ],
        Output("poll", "disabled"),
        Input("job", "data"),
    )
    def busy(identifier):
        return [bool(identifier)] * 7 + [not bool(identifier)]

    @app.callback(Output("stop", "children"), Input("stop", "n_clicks"), prevent_initial_call=True)
    def stop_dashboard(_):
        shutdown = getattr(app, "shutdown_server", None)
        if shutdown is not None:
            threading.Timer(0.5, shutdown).start()
            return "Stopped · you can close this tab"
        return "Close the server to stop this preview"

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local interactive vibration workspace.")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path(".dashboard-cache"))
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.cache / "dashboard.log", level=logging.INFO, encoding="utf-8")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app = create_app(args.cache)
    server = make_server("127.0.0.1", args.port, app.server, threaded=True)
    app.shutdown_server = server.shutdown
    url = f"http://127.0.0.1:{server.server_port}"
    logging.getLogger(__name__).info("Dashboard listening at %s", url)
    print(f"Vibration workspace: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.vibration_service.close()


if __name__ == "__main__":
    main()
