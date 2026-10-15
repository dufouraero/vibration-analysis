"""Desktop front end for the same analysis commands as the CLI (standard-library Tk)."""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


def build_arguments(settings, plots):
    """Translate form values into CLI arguments without invoking a shell."""
    source = Path(settings["input"])
    if not settings["input"] or not source.is_file():
        raise ValueError("Choose an existing CSV file first.")
    if not plots:
        raise ValueError("Select at least one plot type.")
    if not settings["output"].strip():
        raise ValueError("Choose an output folder.")
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
        "skiprows": "--skiprows",
        "decimal": "--decimal",
        "psd_highpass_hz": "--psd-highpass-hz",
        "spectrogram_seconds": "--spectrogram-seconds",
        "spectrogram_overlap": "--spectrogram-overlap",
        "spectrogram_range_db": "--spectrogram-range-db",
        "spectrogram_colormap": "--spectrogram-colormap",
        "spectrogram_interpolation": "--spectrogram-interpolation",
        "waterfall_max_frames": "--waterfall-max-frames",
        "psd_filter_order": "--psd-filter-order",
    }
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
    if settings.get("no_header") == "1":
        args.append("--no-header")
    for band in settings.get("bands", "").split(","):
        if band.strip():
            args.extend(["--band", band.strip()])
    return args


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Vibration analysis")
        self.root.geometry("880x810")
        self.root.minsize(780, 740)
        self.messages = queue.Queue()
        self.process = None
        self.busy = False
        self.last_output = None
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
        self.button(
            paths, "Try synthetic demo", self.demo, row=2, column=1, sticky="w", pady=(6, 0)
        )
        tabs = ttk.Notebook(outer)
        tabs.pack(fill="x", pady=12)
        csv_tab = ttk.Frame(tabs, padding=12)
        analysis_tab = ttk.Frame(tabs, padding=12)
        spectrogram_tab = ttk.Frame(tabs, padding=12)
        tabs.add(csv_tab, text="CSV settings")
        tabs.add(analysis_tab, text="Analysis settings")
        tabs.add(spectrogram_tab, text="Spectrogram")
        for tab in (csv_tab, analysis_tab, spectrogram_tab):
            tab.columnconfigure(1, weight=1)
            tab.columnconfigure(3, weight=1)
        self.field(csv_tab, "Time column (blank = auto)", "time_column", "", 0)
        self.field(
            csv_tab, "Time unit", "time_unit", "s", 0, 2, ["s", "ms", "us", "ns", "datetime"]
        )
        self.field(csv_tab, "Sample rate [Hz] (if no time)", "sample_rate", "", 1)
        self.field(csv_tab, "Signal unit", "unit", "g", 1, 2)
        self.field(csv_tab, "Channels (comma separated)", "channels", "", 2)
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
            text="Blank channels = all numeric columns except time. Unit changes labels; multiplier calibrates values.",
            wraplength=750,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
        self.field(analysis_tab, "PSD/waterfall window [s]", "segment_seconds", "1", 0)
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
        self.field(analysis_tab, "Plot minimum [Hz]", "fmin", "0", 2)
        self.field(analysis_tab, "Plot maximum [Hz] (optional)", "fmax", "", 2, 2)
        self.field(analysis_tab, "RMS bands (e.g. 100:500)", "bands", "", 3)
        self.field(analysis_tab, "PSD high-pass [Hz] (0 = off)", "psd_highpass_hz", "10", 3, 2)
        self.field(analysis_tab, "Waterfall frames (0 = all)", "waterfall_max_frames", "60", 4)
        self.field(analysis_tab, "HPF order per pass", "psd_filter_order", "4", 4, 2)
        ttk.Label(
            analysis_tab,
            text="HPF applies to PSD, PSD-derived RMS and spectrogram. Order is per pass; forward–backward filtering doubles the roll-off order. FFT, waterfall and time trace use the original signal.",
            wraplength=750,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(10, 0))
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
        ttk.Label(
            spectrogram_tab,
            text=(
                "Shorter windows reveal sweeps and impacts; longer windows separate close tones. "
                "Bilinear blends displayed pixels only; exported PSD values are unchanged. "
                "Try 0.05–0.2 s with 80–90% overlap for changing engine speed."
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
        self.log = tk.Text(outer, height=9, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
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
            self.variables["output"].set(str(Path("results").resolve() / Path(filename).stem))

    def choose_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.variables["output"].set(directory)

    def append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def launch(self, args, success):
        self.busy = True
        for button in self.buttons:
            button.configure(state="disabled")
        self.progress.start()
        self.status.set("Working…")
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
                self.messages.put(("log", f"{exc}\n"))
                self.messages.put(("done", (1, success)))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    self.append(value)
                else:
                    self.busy = False
                    self.process = None
                    self.progress.stop()
                    for button in self.buttons:
                        button.configure(state="normal")
                    code, success = value
                    self.status.set(
                        success if code == 0 else "Analysis failed — see details below."
                    )
                    if code:
                        messagebox.showerror(
                            "Could not complete analysis",
                            "Check the details below for the cause and adjust the CSV/settings.",
                        )
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def run(self):
        try:
            settings = {key: var.get() for key, var in self.variables.items()}
            # Each click gets a separate folder to preserve earlier analyses.
            destination = Path(settings["output"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            args = build_arguments(
                settings, [name for name, var in self.plot_variables.items() if var.get()]
            )
            args[args.index("--output") + 1] = str(destination)
        except ValueError as exc:
            messagebox.showerror("Check settings", str(exc))
            return
        self.last_output = destination
        self.append(f"\nAnalyzing {settings['input']}\nResults: {destination}\n")
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
            "segment_seconds": "1",
            "overlap": "0.5",
            "window": "hann",
            "detrend": "constant",
            "psd_highpass_hz": "10",
            "spectrogram_seconds": "0.1",
            "spectrogram_overlap": "0.8",
            "spectrogram_range_db": "60",
            "spectrogram_colormap": "viridis",
            "spectrogram_interpolation": "bilinear",
            "waterfall_max_frames": "60",
            "psd_filter_order": "4",
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
