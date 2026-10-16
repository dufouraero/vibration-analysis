"""Desktop front end for the same analysis commands as the CLI (standard-library Tk)."""

import csv
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .progress import parse as parse_progress


def sine_wave_icon(root):
    """Draw a small waveform icon without adding image assets or dependencies."""
    size = 64
    image = tk.PhotoImage(master=root, width=size, height=size)
    background, wave = (15, 23, 42), (56, 189, 248)
    for y in range(size):
        colors = []
        for x in range(size):
            phase = 2 * math.pi * (x - 8) / 48
            center = 32 - 17 * math.sin(phase)
            slope = -17 * 2 * math.pi / 48 * math.cos(phase)
            distance = abs(y - center) / math.sqrt(1 + slope * slope)
            coverage = max(0, min(1, 2.2 - distance)) if 8 <= x <= 56 else 0
            rgb = [round(a + coverage * (b - a)) for a, b in zip(background, wave)]
            colors.append("#{:02x}{:02x}{:02x}".format(*rgb))
        image.put("{" + " ".join(colors) + "}", to=(0, y))
    # Transparent rounded corners keep the icon clear in Windows title bars.
    for y in range(size):
        for x in range(size):
            dx, dy = max(11 - x, x - 52, 0), max(11 - y, y - 52, 0)
            if dx * dx + dy * dy > 121:
                image.transparency_set(x, y, True)
    return image


def build_arguments(settings, plots):
    """Translate form values into CLI arguments without invoking a shell."""
    source = Path(settings["input"])
    if not settings["input"] or not source.is_file():
        raise ValueError("Choose an existing CSV file first.")
    if not plots:
        raise ValueError("Select at least one plot type.")
    if not settings["output"].strip():
        raise ValueError("Choose an output folder.")
    if (
        settings.get("grafana_start", "").strip() or settings.get("grafana_end", "").strip()
    ) and not settings.get("rpm_csv", "").strip():
        raise ValueError(
            "Choose an mRPM CSV to use the Grafana clock-time window, or clear the Grafana start/end fields."
        )
    args = ["analyze", str(source), "--output", settings["output"], "--plots", *plots]
    mapping = {
        "time_column": "--time-column",
        "time_unit": "--time-unit",
        "sample_rate": "--sample-rate",
        "unit": "--unit",
        "scale": "--scale",
        "segment_seconds": "--segment-seconds",
        "overlap": "--overlap",
        "window": "--window",
        "detrend": "--detrend",
        "fmin": "--fmin",
        "fmax": "--fmax",
        "waterfall2d_fmax": "--waterfall2d-fmax",
        "raw_rpm_time_column": "--raw-rpm-time-column",
        "raw_rpm_column": "--raw-rpm-column",
        "electrical_time_column": "--electrical-time-column",
        "electrical_column": "--electrical-column",
        "electrical_quantity": "--electrical-quantity",
        "skiprows": "--skiprows",
        "decimal": "--decimal",
        "psd_highpass_hz": "--psd-highpass-hz",
        "spectrogram_seconds": "--spectrogram-seconds",
        "spectrogram_overlap": "--spectrogram-overlap",
        "spectrogram_range_db": "--spectrogram-range-db",
        "spectrogram_headroom_db": "--spectrogram-headroom-db",
        "spectrogram_lower_db": "--spectrogram-lower-db",
        "spectrogram_upper_db": "--spectrogram-upper-db",
        "spectrogram_colormap": "--spectrogram-colormap",
        "spectrogram_interpolation": "--spectrogram-interpolation",
        "waterfall_max_frames": "--waterfall-max-frames",
        "psd_filter_order": "--psd-filter-order",
        "start_seconds": "--start-seconds",
        "end_seconds": "--end-seconds",
        "trim_start_seconds": "--trim-start-seconds",
        "trim_end_seconds": "--trim-end-seconds",
        "vibration_start": "--vibration-start",
        "grafana_timezone": "--grafana-timezone",
    }
    if settings.get("raw_rpm_csv", "").strip():
        if not Path(settings["raw_rpm_csv"]).is_file():
            raise ValueError("Choose an existing raw RPM CSV for the time plot, or leave it blank.")
        args.extend(["--raw-rpm-csv", settings["raw_rpm_csv"]])
    if settings.get("electrical_csv", "").strip():
        if not Path(settings["electrical_csv"]).is_file():
            raise ValueError("Choose an existing voltage/current CSV or leave it blank.")
        args.extend(["--electrical-csv", settings["electrical_csv"]])
    if settings.get("rpm_csv", "").strip():
        if not Path(settings["rpm_csv"]).is_file():
            raise ValueError(
                "Choose an existing mRPM CSV or leave it blank for manual interval analysis."
            )
        args.extend(["--mrpm-csv", settings["rpm_csv"]])
        mapping.update(
            {
                "rpm_time_column": "--rpm-time-column",
                "rpm_column": "--rpm-column",
                "vibration_start": "--vibration-start",
                "rpm_alignment": "--rpm-alignment",
                "rpm_offset_seconds": "--rpm-offset-seconds",
                "running_rpm": "--running-rpm",
                "minimum_run_seconds": "--minimum-run-seconds",
                "minimum_hold_seconds": "--minimum-hold-seconds",
                "rpm_stability_seconds": "--rpm-stability-seconds",
                "rpm_slope_limit": "--rpm-slope-limit",
                "rpm_scatter_limit": "--rpm-scatter-limit",
                "grafana_start": "--grafana-start",
                "grafana_end": "--grafana-end",
                "grafana_timezone": "--grafana-timezone",
            }
        )
    for name, flag in mapping.items():
        value = settings.get(name, "").strip()
        if value:
            args.extend([flag, value])
    delimiter = settings.get("delimiter", "Auto")
    if delimiter != "Auto":
        args.extend(
            ["--delimiter", {"Comma": ",", "Semicolon": ";", "Tab": "\t", "Pipe": "|"}[delimiter]]
        )
    channels = [value.strip() for value in settings.get("channels", "").split(",") if value.strip()]
    if channels:
        args.extend(["--channels", *channels])
    elif settings.get("channel_mode", "Auto activity") == "Auto activity":
        args.append("--active-channels")
    if settings.get("no_header") == "1":
        args.append("--no-header")
    for band in settings.get("bands", "").split(","):
        if band.strip():
            args.extend(["--band", band.strip()])
    if settings.get("spectrogram_rpm", "1") == "0":
        args.append("--no-spectrogram-rpm")
    return args


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Vibration analysis")
        self.window_icon = sine_wave_icon(root)
        self.root.iconphoto(True, self.window_icon)
        self.root.geometry("1060x940")
        self.root.minsize(960, 820)
        self.messages = queue.Queue()
        self.process = None
        self.busy = False
        self.last_output = None
        self.log_output = None
        self.completion_message = None
        self.variables = {}
        self.plot_variables = {}
        self.buttons = []
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Vibration analysis", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Choose your recording, select plots, then run.").pack(
            anchor="w", pady=(2, 12)
        )
        paths = ttk.LabelFrame(outer, text="Files", padding=10)
        paths.pack(fill="x")
        paths.columnconfigure(1, weight=1)
        self.field(paths, "CSV file", "input", "", 0)
        self.button(paths, "Browse…", self.choose_csv, row=0, column=2)
        self.field(paths, "Output folder", "output", str(Path("results").resolve()), 1)
        self.button(paths, "Browse…", self.choose_output, row=1, column=2)
        self.field(paths, "mRPM CSV (analysis/detection)", "rpm_csv", "", 2)
        self.button(paths, "Browse…", self.choose_rpm, row=2, column=2)
        self.field(paths, "Raw RPM CSV (time plot only)", "raw_rpm_csv", "", 3)
        self.button(paths, "Browse…", self.choose_raw_rpm, row=3, column=2)
        self.field(paths, "Voltage/current CSV (optional)", "electrical_csv", "", 4)
        self.button(paths, "Browse…", self.choose_electrical, row=4, column=2)
        self.button(
            paths, "Try synthetic demo", self.demo, row=5, column=1, sticky="w", pady=(6, 0)
        )
        tabs = ttk.Notebook(outer)
        tabs.pack(fill="x", pady=12)
        csv_tab = ttk.Frame(tabs, padding=12)
        analysis_tab = ttk.Frame(tabs, padding=12)
        spectrogram_tab = ttk.Frame(tabs, padding=12)
        interval_tab = ttk.Frame(tabs, padding=12)
        rpm_tab = ttk.Frame(tabs, padding=12)
        clock_tab = ttk.Frame(tabs, padding=12)
        electrical_tab = ttk.Frame(tabs, padding=12)
        tabs.add(csv_tab, text="CSV settings")
        tabs.add(analysis_tab, text="Analysis settings")
        tabs.add(spectrogram_tab, text="Spectrogram")
        tabs.add(interval_tab, text="Interval & trim")
        tabs.add(rpm_tab, text="mRPM detection")
        tabs.add(clock_tab, text="Grafana time window")
        tabs.add(electrical_tab, text="Operating data CSVs")
        for tab in (
            csv_tab,
            analysis_tab,
            spectrogram_tab,
            interval_tab,
            rpm_tab,
            clock_tab,
            electrical_tab,
        ):
            tab.columnconfigure(1, weight=1)
            tab.columnconfigure(3, weight=1)
        self.field(csv_tab, "Time column (blank = auto)", "time_column", "", 0)
        self.field(
            csv_tab, "Time unit", "time_unit", "s", 0, 2, ["s", "ms", "us", "ns", "datetime"]
        )
        self.field(csv_tab, "Sample rate [Hz] (if no time)", "sample_rate", "", 1)
        self.field(csv_tab, "Signal unit", "unit", "g", 1, 2)
        self.field(
            csv_tab,
            "Sensor selection",
            "channel_mode",
            "Auto activity",
            2,
            choices=["Auto activity", "All channels"],
        )
        self.field(csv_tab, "Calibration multiplier", "scale", "1", 2, 2)
        self.field(
            csv_tab,
            "Delimiter",
            "delimiter",
            "Auto",
            3,
            choices=["Auto", "Comma", "Semicolon", "Tab", "Pipe"],
        )
        self.field(csv_tab, "Decimal separator", "decimal", ".", 3, 2, [".", ","])
        self.field(csv_tab, "Metadata rows to skip", "skiprows", "0", 4)
        self.variables["no_header"] = tk.StringVar(value="0")
        ttk.Checkbutton(
            csv_tab,
            text="CSV has no header",
            variable=self.variables["no_header"],
            onvalue="1",
            offvalue="0",
        ).grid(row=4, column=2, columnspan=2, sticky="w")
        ttk.Label(
            csv_tab,
            text="Auto activity excludes inputs with negligible vibration relative to the strongest. Explicit sensor names override it. Unit is a label; multiplier calibrates values.",
            wraplength=900,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.field(csv_tab, "Specific sensors (optional)", "channels", "", 5)
        self.button(
            csv_tab,
            "Choose sensor channels…",
            self.choose_channels,
            row=5,
            column=2,
            columnspan=2,
            sticky="w",
        )
        self.field(analysis_tab, "Welch/waterfall window [s]", "segment_seconds", "0.5", 0)
        self.field(analysis_tab, "Overlap fraction", "overlap", "0.5", 0, 2)
        self.field(
            analysis_tab,
            "Window",
            "window",
            "hann",
            1,
            choices=["hann", "hamming", "flattop", "boxcar"],
        )
        self.field(
            analysis_tab, "Detrend", "detrend", "constant", 1, 2, ["constant", "linear", "none"]
        )
        self.field(analysis_tab, "Trimmed frequency min [Hz]", "fmin", "0", 2)
        self.field(analysis_tab, "Trimmed frequency max [Hz]", "fmax", "", 2, 2)
        self.field(analysis_tab, "RMS bands (e.g. 100:500)", "bands", "", 3)
        self.field(analysis_tab, "PSD high-pass [Hz] (0 = off)", "psd_highpass_hz", "10", 3, 2)
        self.field(analysis_tab, "Waterfall frames (0 = all)", "waterfall_max_frames", "60", 4)
        self.field(analysis_tab, "HPF order per pass", "psd_filter_order", "4", 4, 2)
        self.field(analysis_tab, "2D waterfall max [Hz] (optional)", "waterfall2d_fmax", "", 5)
        self.field(electrical_tab, "Electrical time column (auto)", "electrical_time_column", "", 0)
        self.field(electrical_tab, "Electrical value column (auto)", "electrical_column", "", 0, 2)
        self.field(
            electrical_tab,
            "Quantity",
            "electrical_quantity",
            "auto",
            1,
            choices=["auto", "voltage", "current"],
        )
        self.field(electrical_tab, "Raw RPM time column (auto)", "raw_rpm_time_column", "", 2)
        self.field(electrical_tab, "Raw RPM value column (auto)", "raw_rpm_column", "", 2, 2)
        ttk.Label(
            electrical_tab,
            text="These optional CSVs supply the simple time plot only. The separate mRPM CSV drives alignment and hold detection. Auto prefers voltage if present, otherwise current. Raw logged samples are plotted against Grafana local time. Missing sources are omitted.",
            wraplength=850,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(
            analysis_tab,
            text="Frequency plots save broadband and trimmed views. Frequency limits crop the display. HPF applies to PSD, PSD-derived RMS and spectrogram. Order is per pass; forward–backward filtering doubles the roll-off order. FFT, waterfall and time trace use the original signal.",
            wraplength=750,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.field(interval_tab, "Interval start [s]", "start_seconds", "0", 0)
        self.field(interval_tab, "Interval end [s] (blank = file end)", "end_seconds", "", 0, 2)
        self.field(interval_tab, "Trim from start [s]", "trim_start_seconds", "5", 1)
        self.field(interval_tab, "Trim from end [s]", "trim_end_seconds", "5", 1, 2)
        self.field(clock_tab, "Grafana start (local time)", "grafana_start", "", 0)
        self.field(clock_tab, "Grafana end (local time)", "grafana_end", "", 0, 2)
        self.field(clock_tab, "Grafana display timezone", "grafana_timezone", "Europe/Zurich", 1)
        ttk.Label(
            clock_tab,
            text=(
                "Optional: enter YYYY-MM-DD HH:MM:SS (e.g. 2026-10-14 16:01:00), or HH:MM:SS "
                "for the first date in the RPM export. Use the same local clock shown in Grafana. "
                "Requires an mRPM CSV; the estimated/manual alignment offset is applied automatically. "
                "Blank bounds leave that side unrestricted. This window intersects any elapsed-time limits; "
                "all exported test/point spectra and statistics use only samples inside it."
            ),
            wraplength=900,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Label(
            interval_tab,
            text=(
                "Times are elapsed seconds in the source CSV. Trims apply inside the chosen interval, "
                "before every analysis. Use 0 to disable. Example: a hold from 100 to 160 s with "
                "5 s at each end retains 105–155 s. With an mRPM CSV, limits restrict the tests and "
                "trims apply to each detected hold; whole-test plots retain transitions."
            ),
            wraplength=750,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.field(rpm_tab, "Vibration start timestamp", "vibration_start", "", 0)
        self.field(rpm_tab, "Alignment mode", "rpm_alignment", "Auto", 0, 2, ["Auto", "Manual"])
        self.field(rpm_tab, "Manual offset [s]", "rpm_offset_seconds", "0", 1)
        self.field(rpm_tab, "Running threshold [RPM]", "running_rpm", "500", 1, 2)
        self.field(rpm_tab, "Minimum test duration [s]", "minimum_run_seconds", "30", 2)
        self.field(rpm_tab, "Minimum hold duration [s]", "minimum_hold_seconds", "30", 2, 2)
        self.field(rpm_tab, "Stability window [s]", "rpm_stability_seconds", "20", 3)
        self.field(rpm_tab, "Max RPM drift [RPM/s]", "rpm_slope_limit", "2", 3, 2)
        self.field(rpm_tab, "Max scatter [RPM RMS]", "rpm_scatter_limit", "40", 4)
        self.field(rpm_tab, "mRPM value column (blank = auto)", "rpm_column", "", 4, 2)
        self.field(rpm_tab, "mRPM time column (blank = auto)", "rpm_time_column", "", 5)
        ttk.Label(
            rpm_tab,
            text=(
                "Blank start timestamp uses the vibration filename. Auto alignment matches starts/stops "
                "within ±300 s using the first selected vibration channel. Inspect the overview before "
                "accepting results. Positive manual offset moves RPM events later on the CSV timeline. "
                "Holds are suggestions; gentle ramps and warmup need review."
            ),
            wraplength=900,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.field(spectrogram_tab, "Window duration [s]", "spectrogram_seconds", "0.1", 0)
        self.field(spectrogram_tab, "Overlap fraction", "spectrogram_overlap", "0.8", 0, 2)
        self.field(spectrogram_tab, "Color range [dB]", "spectrogram_range_db", "60", 1)
        self.field(
            spectrogram_tab,
            "Colormap",
            "spectrogram_colormap",
            "viridis",
            1,
            2,
            ["viridis", "magma", "turbo"],
        )
        self.field(
            spectrogram_tab,
            "Display blending",
            "spectrogram_interpolation",
            "bilinear",
            2,
            choices=["bilinear", "nearest"],
        )
        self.field(
            spectrogram_tab, "Auto upper headroom [dB]", "spectrogram_headroom_db", "6", 2, 2
        )
        self.field(spectrogram_tab, "Lower PSD [dB] (blank = auto)", "spectrogram_lower_db", "", 3)
        self.field(
            spectrogram_tab, "Upper PSD [dB] (blank = auto)", "spectrogram_upper_db", "", 3, 2
        )
        self.variables["spectrogram_rpm"] = tk.StringVar(value="1")
        ttk.Checkbutton(
            spectrogram_tab,
            text="Show mean RPM operating points underneath (when available)",
            variable=self.variables["spectrogram_rpm"],
            onvalue="1",
            offvalue="0",
        ).grid(row=4, column=0, columnspan=4, sticky="w")
        ttk.Label(
            spectrogram_tab,
            text=(
                "Shorter windows reveal sweeps and impacts; longer windows separate close tones. "
                "Bilinear blends displayed pixels only; exported PSD values are unchanged. "
                "Limits are PSD dB re 1 signal-unit^2/Hz. Auto upper = peak + headroom; "
                "auto lower = upper minus color range. Increase upper to reduce bright saturation. "
                "Use fixed limits to compare operating points."
            ),
            wraplength=750,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
        plot_frame = ttk.LabelFrame(outer, text="Plots", padding=10)
        plot_frame.pack(fill="x")
        for col, (name, label) in enumerate(
            [
                ("fft", "FFT"),
                ("waterfall", "Waterfall"),
                ("psd", "PSD (linear + log x)"),
                ("spectrogram", "Spectrogram"),
                ("time", "Time trace"),
            ]
        ):
            self.plot_variables[name] = tk.BooleanVar(value=True)
            ttk.Checkbutton(plot_frame, text=label, variable=self.plot_variables[name]).grid(
                row=0, column=col, padx=5
            )
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=12)
        self.run_button = self.button(actions, "Run selected analyses", self.run, row=0, column=0)
        self.open_button = self.button(
            actions, "Open results folder", self.open_results, row=0, column=1, padx=10
        )
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=2)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(0, 6))
        self.output_tabs = ttk.Notebook(outer)
        self.output_tabs.pack(fill="both", expand=True)
        self.log = ScrolledText(
            self.output_tabs, height=8, wrap="word", state="disabled", font=("Segoe UI", 10)
        )
        self.details = ScrolledText(
            self.output_tabs, height=8, wrap="word", state="disabled", font=("Consolas", 9)
        )
        self.output_tabs.add(self.log, text="Progress")
        self.output_tabs.add(self.details, text="Technical details")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.poll)

    def field(self, parent, label, key, default, row, col=0, choices=None):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        var = self.variables[key] = tk.StringVar(value=default)
        widget = (
            ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=17)
            if choices
            else ttk.Entry(parent, textvariable=var, width=24)
        )
        widget.grid(row=row, column=col + 1, sticky="ew", padx=(0, 10), pady=4)

    def button(self, parent, text, command, **grid):
        widget = ttk.Button(parent, text=text, command=command)
        widget.grid(**grid)
        self.buttons.append(widget)
        return widget

    def choose_csv(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV / text files", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if filename:
            self.variables["input"].set(filename)
            self.variables["output"].set(str(Path("results").resolve()))

    def choose_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.variables["output"].set(directory)

    def choose_rpm(self):
        filename = filedialog.askopenfilename(
            filetypes=[("mRPM analysis CSV", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.variables["rpm_csv"].set(filename)

    def choose_raw_rpm(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Raw RPM CSV (time plot only)", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.variables["raw_rpm_csv"].set(filename)

    def choose_electrical(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Voltage/current CSV", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.variables["electrical_csv"].set(filename)

    def choose_channels(self):
        try:
            settings = {key: var.get() for key, var in self.variables.items()}
            with Path(settings["input"]).open(encoding="utf-8-sig", newline="") as stream:
                skip = int(settings["skiprows"])
                if skip < 0:
                    raise ValueError("Metadata rows to skip cannot be negative.")
                for _ in range(skip):
                    stream.readline()
                sample = stream.read(65536)
            separators = {"Comma": ",", "Semicolon": ";", "Tab": "\t", "Pipe": "|"}
            delimiter = separators.get(settings["delimiter"])
            if delimiter is None:
                try:
                    delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error:
                    delimiter = ","
            row = next(csv.reader(sample.splitlines(), delimiter=delimiter))
            names = (
                [str(i) for i in range(len(row))]
                if settings["no_header"] == "1"
                else [c.strip() for c in row]
            )
            ignored = {
                "sample",
                "sample index",
                "time",
                "time_s",
                "time (s)",
                "time [s]",
                "timestamp",
                "t",
                settings["time_column"].lower(),
            }
            names = [name for name in names if name and name.lower() not in ignored]
            if not names:
                raise ValueError("No signal column names found; check CSV settings.")
        except (OSError, ValueError, StopIteration, csv.Error) as exc:
            messagebox.showerror(
                "Choose sensor channels",
                f"Choose a CSV and check its delimiter/header settings first.\n{exc}",
            )
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Choose sensor channels")
        dialog.geometry("560x330")
        dialog.transient(self.root)
        ttk.Label(
            dialog,
            text="Select the connected sensor(s). This overrides automatic selection.",
            wraplength=520,
        ).pack(padx=15, pady=12)
        options = tk.Listbox(dialog, selectmode="multiple", exportselection=False)
        options.pack(fill="both", expand=True, padx=15)
        current = {name.strip() for name in settings["channels"].split(",")}
        for i, name in enumerate(names):
            options.insert("end", name)
            if name in current:
                options.selection_set(i)

        def apply():
            selected = [names[i] for i in options.curselection()]
            if not selected:
                messagebox.showinfo(
                    "Select a sensor", "Select at least one connected sensor.", parent=dialog
                )
                return
            self.variables["channels"].set(", ".join(selected))
            dialog.destroy()

        ttk.Button(dialog, text="Use selected sensors", command=apply).pack(pady=12)
        dialog.grab_set()

    def append(self, text, *, details=False):
        widget = self.details if details else self.log
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def launch(self, args, success):
        for widget in (self.log, self.details):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")
        self.output_tabs.select(0)
        self.completion_message = None
        self.log_output = None
        if args[0] == "analyze":
            args = [*args, "--gui-progress"]
            self.log_output = Path(args[args.index("--output") + 1])
            self.append(f"Starting analysis: {Path(args[1]).name}\n")
            self.append(f"Source: {args[1]}\nResults: {self.log_output}\n", details=True)
        else:
            self.append("Creating synthetic demo data…\n")
        self.busy = True
        for button in self.buttons:
            button.configure(state="disabled")
        self.progress.start()
        self.status.set("Starting…")
        python = (
            Path(sys.executable).with_name("python.exe")
            if os.name == "nt"
            else Path(sys.executable)
        )

        def worker():
            try:
                self.process = subprocess.Popen(
                    [str(python), "-u", "-m", "vibration_analysis", *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                for line in self.process.stdout:
                    self.messages.put(("log", line))
                code = self.process.wait()
                self.messages.put(("done", (code, success)))
            except (OSError, UnicodeError) as exc:
                self.messages.put(("log", f"Error: {exc}\n"))
                self.messages.put(("done", (1, success)))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    event = parse_progress(value)
                    if event is not None:
                        self.status.set(event["message"])
                        self.append(event["message"] + "\n", details=True)
                        if event.get("history", True):
                            self.append(event["message"] + "\n")
                        if event.get("complete"):
                            self.completion_message = event["message"]
                    else:
                        self.append(value, details=True)
                        if value.lstrip().startswith(("Warning:", "Error:")):
                            self.append(value)
                else:
                    self.busy = False
                    self.process = None
                    self.progress.stop()
                    for button in self.buttons:
                        button.configure(state="normal")
                    code, success = value
                    self.status.set(
                        (self.completion_message or success)
                        if code == 0
                        else "Analysis failed — open Technical details."
                    )
                    if code == 0 and not self.completion_message:
                        self.append(success + "\n")
                    if self.log_output is not None and self.log_output.is_dir():
                        try:
                            (self.log_output / "analysis.log").write_text(
                                self.details.get("1.0", "end-1c"), encoding="utf-8"
                            )
                        except OSError as exc:
                            self.append(f"Warning: could not save the technical log: {exc}\n")
                    if code:
                        self.output_tabs.select(1)
                        messagebox.showerror(
                            "Could not complete analysis",
                            "See Technical details for the error. The full log is saved with the results when possible.",
                        )
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def run(self):
        try:
            settings = {key: var.get() for key, var in self.variables.items()}
            # Each click gets a separate folder to preserve earlier analyses.
            destination = Path(settings["output"]) / (
                Path(settings["input"]).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            args = build_arguments(
                settings, [name for name, var in self.plot_variables.items() if var.get()]
            )
            args[args.index("--output") + 1] = str(destination)
            # Reserve a new root: never merge this run with existing results.
            destination.mkdir(parents=True, exist_ok=False)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check settings", str(exc))
            return
        self.last_output = destination
        self.launch(args, "Done — click Open results folder to view plots.")

    def demo(self):
        destination = Path("data/demo.csv").resolve()
        self.variables["input"].set(str(destination))
        self.variables["output"].set(str(Path("results/demo").resolve()))
        for key, value in {
            "time_column": "time_s",
            "time_unit": "s",
            "sample_rate": "",
            "channels": "accel_steady_g, accel_sweep_g",
            "skiprows": "0",
            "delimiter": "Auto",
            "decimal": ".",
            "no_header": "0",
            "scale": "1",
            "unit": "g",
            "fmin": "0",
            "fmax": "1000",
            "bands": "",
            "segment_seconds": "0.5",
            "overlap": "0.5",
            "window": "hann",
            "detrend": "constant",
            "psd_highpass_hz": "10",
            "spectrogram_seconds": "0.1",
            "spectrogram_overlap": "0.8",
            "spectrogram_range_db": "60",
            "spectrogram_headroom_db": "6",
            "spectrogram_lower_db": "",
            "spectrogram_upper_db": "",
            "spectrogram_rpm": "1",
            "spectrogram_colormap": "viridis",
            "spectrogram_interpolation": "bilinear",
            "waterfall_max_frames": "60",
            "psd_filter_order": "4",
            "start_seconds": "0",
            "end_seconds": "",
            "trim_start_seconds": "0",
            "trim_end_seconds": "0",
            "rpm_csv": "",
            "raw_rpm_csv": "",
            "raw_rpm_time_column": "",
            "raw_rpm_column": "",
            "electrical_csv": "",
            "electrical_time_column": "",
            "electrical_column": "",
            "electrical_quantity": "auto",
            "waterfall2d_fmax": "",
            "channel_mode": "Auto activity",
            "grafana_start": "",
            "grafana_end": "",
        }.items():
            self.variables[key].set(value)
        self.launch(
            ["demo", "--output", str(destination)], "Demo ready — click Run selected analyses."
        )

    def open_results(self):
        destination = self.last_output or Path(self.variables["output"].get())
        if not destination.is_dir():
            messagebox.showinfo("Results", "Run an analysis first to create the results folder.")
            return
        if os.name == "nt":
            os.startfile(destination)
        else:
            import webbrowser

            webbrowser.open(destination.resolve().as_uri())

    def close(self):
        if self.busy:
            messagebox.showinfo(
                "Analysis running", "Please wait for the current analysis to finish before closing."
            )
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
