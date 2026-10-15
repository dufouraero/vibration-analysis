# Vibration analysis

A small Python toolkit for accelerometer/test-bench CSV recordings. Produces a
full-record FFT, FFT waterfall, Welch PSD with **linear and log frequency axes**,
PSD spectrogram, and a time trace. Includes numerical exports, RMS statistics,
synthetic example data, tests and GitHub Actions checks. No GUI is required.

## Quick start

### From VS Code (recommended)

Open this repo folder in VS Code and install the recommended **Python** and
**Python Debugger** extensions if prompted. Press **F5** or choose **Run → Start
Without Debugging** (**Ctrl+F5**) to open **Vibration Analysis — Desktop GUI**.
The launch configuration runs `uv sync --locked` automatically first, then uses
the repo's `.venv` interpreter. No terminal commands are needed.

Alternatively, press **Ctrl+Shift+B** to run the **Open Vibration Analysis** task.
This launches the same interface without needing the Python debugger extension.
Task errors can be inspected in VS Code's terminal panel; normal launches keep it hidden.
On Windows the tasks use uv's default installation at `%USERPROFILE%\.local\bin\uv.exe`;
adjust `.vscode/tasks.json` if you installed uv elsewhere.

### Desktop interface (Windows — no terminal needed)

Install uv once using the installation link below. Then double-click
**Open Vibration Analysis.vbs** in the repo folder. The launcher runs
`uv sync --locked --no-dev` automatically and opens a desktop interface.
On first launch, downloading Python/dependencies can take a minute; later launches
usually open quickly. Setup errors appear in a dialog with details in `setup.log`.
Windows must allow Windows Script Host and local PowerShell scripts for this launcher.

1. Click **Browse…** to choose your CSV, or **Try synthetic demo** to test it now.
2. Set CSV options as needed (time column or sampling rate, channels, units).
3. Tick FFT, Waterfall, PSD, Spectrogram and/or Time trace individually.
4. Click **Run selected analyses**, then **Open results folder** to view the PNGs.

Each desktop run gets a new output subfolder named from the vibration CSV plus
a timestamp (including microseconds), preserving previous runs. The popup reserves
this folder without overwriting an existing folder.
The **Progress** tab shows only the main stages, test/operating-point progress,
warnings and completion. The current sensor appears in the status line, without
adding a message for every channel to the progress history. Detailed RMS values,
sample counts and file paths remain available under **Technical details** and are
saved as `analysis.log` in that run's parent folder when the analysis finishes.
On failure, Technical details opens automatically.

**Sensor selection:** the popup defaults to **Auto activity** under CSV settings.
It compares sustained one-second AC-RMS activity and excludes inputs below 5% of
the strongest channel's score. The score is the median of the strongest 1% of
one-second blocks (at least three blocks when available), reducing sensitivity to
single handling/startup bursts. Included channels and the excluded count are shown
in Progress; names and scores are recorded in Technical details and summaries.
This suggests active inputs; it does not detect physical sensor connections.
For quiet sensors that you need to retain, choose **All channels** or click
**Choose sensor channels…** and select the connected inputs explicitly. Explicit
names override Auto activity. Each included sensor gets its own subfolder inside
each test/point. With only one connected sensor, auto selection should normally
leave one sensor subfolder. CLI defaults to all numeric channels; use
`--active-channels` for the same automatic suggestion.
The PSD checkbox generates both linear and log frequency plots. CSV and analysis
settings are available in separate tabs, including optional RMS bands.

To update the repo without a terminal, use **Fetch origin / Pull origin** in
GitHub Desktop, then reopen the launcher. `git pull` updates source files;
`uv sync` installs the matching dependencies. The launcher handles the latter.
There is no automatic Git pull, so updates stay under your control.

For other operating systems, launch the same interface with
`uv run python -m vibration_analysis.gui` (requires Tk support in Python).

### Command line

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then run from this repo:

```powershell
uv sync --locked --dev
uv run vibration demo
uv run vibration analyze data/demo.csv --channels accel_steady_g accel_sweep_g --band 90:110 --band 450:750 --fmax 1000
```

Open `results/01_accel_steady_g/` and `results/02_accel_sweep_g/` to see the PNGs.
The demo has a 0.75 g peak 100 Hz tone, a 2 g peak 500 Hz tone, a 1.5 g peak
700 Hz tone and noise on the steady channel; the other channel sweeps from
80 to 800 Hz and contains a short impact. These are synthetic signals, not engine measurements.

Python 3.11+ is supported; `.python-version` selects 3.12 for development.
`uv.lock` pins the environment. No separate requirements file is needed with uv.
For pip users, `python -m pip install -e .` also works, but does not use the lockfile.

## Your CSV

Preferred format (one row per sample; any signal column names):

```csv
time_s,accel_x,accel_y,accel_z
0.0000,0.12,0.05,-0.02
0.0001,0.15,0.04,-0.01
0.0002,0.11,0.08,-0.03
0.0003,0.13,0.06,-0.02
```

```powershell
# Explicit time column and channels; timestamps determine sample rate.
uv run vibration analyze data/run.csv --time-column time_s --channels accel_x accel_y --unit g

# No time column: provide the acquisition sample rate in Hz.
uv run vibration analyze data/run.csv --sample-rate 10000 --channels accel_x --unit 'm/s^2'

# European CSV: metadata, semicolon delimiter, decimal commas, time in milliseconds.
uv run vibration analyze data/run.csv --skiprows 6 --delimiter ';' --decimal ',' --time-column Time --time-unit ms --channels Accel

# Headerless CSV: columns use zero-based string names.
uv run vibration analyze data/run.csv --no-header --time-column 0 --channels 1 2 --time-unit s

# ISO datetime timestamps (must have enough precision for the sample rate).
uv run vibration analyze data/run.csv --time-column timestamp --time-unit datetime --channels accel_x

# Example calibration: convert a signal already in g to m/s².
uv run vibration analyze data/run.csv --time-column time_s --channels accel_x --scale 9.80665 --unit 'm/s^2'
```

Delimiter detection is automatic by default. Column names are stripped of leading/trailing
spaces and can be quoted if they contain spaces. UTF-8 CSVs (including BOM) are supported.
`--skiprows` skips initial metadata lines before the header. Without `--channels`,
all numeric columns except the selected time column are analyzed. Select channels
explicitly if the CSV also has RPM, temperature, counters or other metadata columns.

When neither time column nor sample rate is supplied, one column named `time`,
`time_s`, `timestamp`, or `t` is auto-detected, case-insensitively. If you supply
`--sample-rate` alone, all rows are assumed uniformly spaced at that rate;
any time-like columns must be excluded using `--channels`. Supply both time column
and sample rate to check them against each other.

Signals and timestamps must be finite, with at least four samples. Timestamps must
increase strictly and have uniform intervals within 1% by default. Missing rows,
duplicate times and large interval changes are rejected rather than silently removed.
`--timing-tolerance` controls this relative interval tolerance. This is a validation
tolerance, not a correction: small accepted timing errors still use a uniform-sampling
approximation. For meaningful jitter or dropped samples, resample deliberately upstream.
The tool cannot recover aliasing already introduced during acquisition; choose a suitable
sample rate and acquisition anti-alias filter.

`--unit` changes labels only. `--scale` multiplies every selected channel by the same
calibration factor. For different channel sensitivities, run channels separately or
calibrate the CSV first. Data is never automatically interpreted as volts, counts or g.

## Analysis controls

```powershell
uv run vibration analyze data/run.csv --time-column time_s --channels accel_x --segment-seconds 2 --overlap 0.5 --window hann --detrend constant --band 100:500 --band 500:1000 --fmin 10 --fmax 1500 --output results/run_01
uv run vibration analyze --help
```

| Option | Meaning / default |
| --- | --- |
| `--segment-seconds` | Welch PSD and waterfall window duration; 0.5 s (500 ms) |
| `--overlap` | Segment overlap fraction; 0.5 |
| `--window` | `hann` (default), `hamming`, `flattop`, `boxcar` |
| `--detrend` | `constant` removes mean; `linear` removes a fitted trend; `none` retains DC |
| `--band LOW:HIGH` | Repeatable band RMS calculation in Hz |
| `--fmin`, `--fmax` | Display limits only; numerical exports/RMS still use the full spectrum |
| `--plots` | `all` (default), or a selection of `time fft psd waterfall spectrogram` |
| `--dpi` | PNG export resolution; 160 |
| `--waterfall-max-frames` | Maximum displayed waterfall spectra; 60 by default, 0 shows all |
| `--psd-highpass-hz` | PSD/spectrogram high-pass cutoff; CLI default 0 (off), GUI default 10 Hz |
| `--psd-filter-order` | Butterworth order per pass; default 4, positive integers |
| `--spectrogram-seconds` | Independent spectrogram window; 0.1 s |
| `--spectrogram-overlap` | Independent spectrogram overlap fraction; 0.8 |
| `--spectrogram-range-db` | Color range below maximum displayed PSD; 60 dB |
| `--spectrogram-colormap` | `viridis` (default), `magma`, `turbo` |
| `--spectrogram-interpolation` | `bilinear` display blending (default), or `nearest` to see individual cells |
| `--start-seconds`, `--end-seconds` | Select a half-open source interval; default entire recording |
| `--trim-start-seconds`, `--trim-end-seconds` | Trim inside selected interval; CLI default 0 each, desktop default 5 s each |

### Interval selection and configurable trim

The desktop **Interval & trim** tab lets you enter elapsed CSV start/end times
and separate trim margins. A blank interval end means the end of the recording.
The popup defaults to **5 s at the start and 5 s at the end**. Use **0** to disable
either margin. Selecting a hold from 100 to 160 s with these margins analyses
**105 <= t < 155 s**, retaining approximately 50 s. The synthetic demo resets both
trims to 0 because its recording is only 8 seconds long.

```powershell
uv run vibration analyze data/run.csv --time-column time_s --channels accel_x --start-seconds 100 --end-seconds 160 --trim-start-seconds 5 --trim-end-seconds 5
```

Trimming applies before all analyses: FFT, PSD, waterfall, spectrogram, time trace
and statistics use the same retained samples. `summary.json` records requested
interval boundaries, trim settings, retained sample indices and retained duration.
Plots and exported segment centers retain the **original CSV elapsed time**, so
the analysed portion can be matched back to the source. FFT phase remains relative
to the first retained sample. Trims that remove the interval or leave fewer than
four samples are rejected. Welch still warns when there are too few samples for
the requested window or fewer than two averaging segments.

Without an RPM CSV, the interface analyses **one manually selected interval**.
With an RPM CSV, it detects candidate runs and settled holds as described below.
The interval limits restrict which tests are analysed; start/end trims apply to
each detected hold. Whole-test plots retain startup, ramp, and shutdown transitions.
Trimming is not a replacement for clock alignment or checking that a point has settled.

### RPM CSV, detected tests, operating points and comparisons

Choose the vibration CSV and the optional **RPM CSV** in the popup. The **RPM detection**
tab provides automatic/manual alignment and configurable detection thresholds.
A Grafana RPM export with a `Time` column and one numeric RPM column is supported
directly. Other column names can be selected explicitly. Only RPM telemetry is
currently supported in this workflow; current/voltage import is not implemented.

The supplied datalogger format (`Sample`, `Time (s)`, sensor columns and an empty
trailing column) is also supported directly: the sample counter and empty unnamed
column are excluded from automatic signal selection. Select the useful sensor
channels explicitly, particularly for alignment.

The workflow produces one parent folder per desktop analysis:

```text
results/
  Job1_2026-10-14T14-58-37-482_20261015_120000_123456/
    alignment.json
    detection_settings.json
    detection_overview.png
    analysis.log                   # desktop run's full technical output
    tests.csv
    test_01/
      whole_test/
        01_sensor_name/            # selected plots, spectra and summary
      operating_points.csv        # suggested boundaries, trims, RPM spread, skips
      operating_points/
        point_01_4500rpm/
          01_sensor_name/
        point_02_5650rpm/
          01_sensor_name/
      comparisons/
        01_sensor_name/
          fft_rpm_waterfall.png
          psd_comparison.png
          operating_point_spectra.npz
          operating_points.csv
    test_02/
      ...
```

The comparison waterfall belongs alongside whole-test and per-point results, inside
each test's **comparisons** folder. It plots one **mean windowed FFT magnitude spectrum**
per retained hold, using the same FFT window length at every point. It is not the
full-record FFT from unequal-duration holds. Its numbered axis retains chronological
point order and labels each point's mean RPM, keeping repeated speeds separate.
Each point also exports its own full-record FFT. Comparison arrays and PSD overlays
use the same window settings. FFT comparison magnitudes remain unfiltered; PSDs use
the configured high-pass. Image creation follows the selected FFT/waterfall and PSD
checkboxes; numerical comparison arrays are always exported when holds are available.

**Clock alignment:** provide the timestamp corresponding to the first vibration
sample, or leave it blank to read a timestamp such as `2026-10-14T14-58-37-482`
from the vibration filename. Automatic mode searches ±300 seconds for agreement
between engine running/downtime in RPM and the first selected sensor's vibration
levels. It requires observable starts/stops in both sources, rejects weak or ambiguous
matches, and is a **heuristic suggestion, not clock synchronization**. Its threshold,
score and candidate offset range are recorded. Always inspect `detection_overview.png`.

In Manual mode, the mapping is:
`CSV elapsed seconds = RPM timestamp - vibration start timestamp + offset`.
A positive offset moves RPM events **later** on the source CSV elapsed timeline.
For the supplied exports an estimated offset around **+30 seconds** is consistent
with the starts/stops. Do not assume it is correct for a new recording.
Timezone-aware timestamps are normalized; naive timestamps are treated as a common
nominal clock. If the sources use different zones, specify an explicit offset in
the vibration start timestamp. Five-second telemetry cannot establish precise
sub-second synchronization. The current model handles a constant offset, not clock
drift or a logger that removes pauses from its exported timeline.

**Run and hold detection:** default running threshold is 500 RPM, minimum test length
30 s, minimum candidate hold length 30 s. A local 20 s regression identifies regions
whose RPM trend is at most 2 RPM/s and scatter about that trend at most 40 RPM RMS.
Continuous ramps above that slope are excluded; settled regions after ramps can
qualify as new holds. Large telemetry gaps break tests. Very slow ramps can resemble
drifting holds: tune the slope/scatter settings and review the suggested boundaries.
The first hold is not automatically labelled warmup: RPM alone cannot determine intent
or load, and a load change at constant RPM may not be detected. Short candidates are
listed as skipped when trims leave too little data for two Welch windows or fewer
than two RPM observations. Warmup/load labels and manual editing of individual
suggested boundaries are not yet part of the popup; use manual interval mode to
override a particular hold.

```powershell
uv run vibration analyze data/Job1_2026-10-14T14-58-37-482.csv --rpm-csv data/rpm.csv --channels "CMCP776A - 1060 (g)" --trim-start-seconds 5 --trim-end-seconds 5 --psd-highpass-hz 10 --output results/new_rpm_analysis
```

CLI RPM batch mode requires a **new or empty output folder**, refusing to overwrite
previous batch results. Desktop mode creates that fresh parent folder automatically.

### Welch window controls

Under **Analysis settings**, the **Welch/waterfall window [s]** defaults to **0.5**,
**Overlap fraction** defaults to **0.5**, and **Window** defaults to **Hann**.
All three are configurable. At approximately 1706.67 samples/s, a 500 ms segment
rounds to about 853 samples, with roughly 2 Hz frequency-bin spacing. The chosen
window also broadens spectral lines, so bin spacing alone is not effective resolution.
The summary records actual segment samples, bin spacing, overlap and average count.
Spectrogram window duration and overlap remain independently configurable in their tab.

### PSD high-pass filter

The desktop **Analysis settings** tab defaults to a **10 Hz high-pass**; enter **0**
to disable it. For CLI use, enable explicitly with `--psd-highpass-hz 10`.
Filtering applies to Welch PSD, its total/band RMS, and the PSD spectrogram.
The time trace, full-record FFT and amplitude waterfall use the original calibrated
signal. Raw time-domain RMS and peaks also refer to the original signal. Titles
and `summary.json` record whether the PSD high-pass is active.

The **HPF order per pass** field sets the Butterworth order (default **4**).
Higher orders produce a steeper roll-off. Enter a positive integer; the CLI equivalent
is `--psd-filter-order 6`, for example. The filter uses second-order sections, applied forward
and backward to the full record before segmenting, using
[SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
It has zero phase delay, with squared single-pass magnitude response: the specified
nominal cutoff has approximately **6 dB attenuation** after both passes, rather than
the single-pass 3 dB. Forward–backward application doubles the effective roll-off
order (for example, order 4 per pass gives order 8 effective roll-off).
It rolls off gradually, rather than deleting bins below 10 Hz.
Finite-record edge transients can remain; inspect record edges for short runs or
large offsets. Very short records that cannot support filter padding are rejected.
Filtering changes physical RMS and should only be used when low-frequency content
is outside the analysis of interest. `--fmin` only crops a plot and does not filter.

### Spectrogram appearance and resolution

The desktop **Spectrogram** tab controls its window and overlap separately from
the Welch PSD. The defaults are **100 ms / 80% overlap** (approximately 20 ms steps),
so speed sweeps appear with much less temporal blocking than the old 1 s / 50% settings.
Increasing overlap samples the evolution more densely, but does not improve the
underlying window's time/frequency resolution. Shorter windows follow rapid changes;
longer windows resolve closer frequencies. The title reports actual window, step
and bin spacing after rounding to samples.

Colors now show `10 log10(PSD)` in dB relative to 1 unit²/Hz, with a configurable
range below the maximum in the displayed frequency band. Bilinear interpolation
blends displayed pixels in dB space for smoother color transitions; choose **nearest**
to inspect the actual cells. This changes display only. Interpolation does not add
measurement resolution, and numerical exports always retain the original linear PSD.

Full-record FFT bin spacing is fs/N = 1/T. Segment spectra use fs/segment_samples.
The window also broadens spectral lines: bin spacing alone is not the ability to
resolve nearby tones. Longer segments improve frequency resolution and reduce time
resolution and the number of averages. Segment lengths are rounded to whole samples,
and shortened with a warning if the record is shorter than requested. Partial final
segments are discarded; the summary reports how many tail samples were unused.
One segment gives a periodogram without Welch averaging or useful time evolution.

## What the outputs mean

Each channel gets a numbered, sanitized subfolder under `--output`:

| File | Contents |
| --- | --- |
| `time.png` | Calibrated retained time signal, on the original CSV elapsed timeline |
| `fft.png`, `fft.csv` | Full-record one-sided **peak** amplitude spectrum plus phase in radians in CSV |
| `psd_linear_x.png` | Welch PSD, linear frequency axis, log density axis |
| `psd_log_x.png` | Same Welch PSD, log frequency axis, log density axis; DC omitted |
| `psd.csv` | All one-sided frequency bins and PSD in unit²/Hz, including DC |
| `waterfall.png` | 3D sequence of windowed FFT peak amplitude spectra versus time |
| `spectrogram.png` | Time–frequency PSD heatmap, logarithmic density color scale |
| `time_frequency.npz` | Windowed FFT amplitudes/PSD plus separate spectrogram arrays and coordinates |
| `summary.json` | Acquisition/settings, time-domain peak/RMS, Welch RMS, band RMS |

Spectrogram colors default to a 60 dB range below that channel's maximum displayed PSD.
This is only display clipping; exports keep all values. Waterfall plots display at most
60 evenly spaced frames by default. Change **Waterfall frames (0 = all)** under
**Analysis settings** in the desktop interface, or use `--waterfall-max-frames`.
For example, 200 displays up to 200 frames and 0 displays every frame. More frames
can make a long recording slower to render and harder to read. The plot title and
summary record the displayed and available counts; exports retain every complete frame
regardless of this display setting. This is a time-based
waterfall. The separate RPM workflow adds a comparison waterfall across detected holds;
neither performs angle-resampled order tracking.

FFT amplitude uses the coherent window gain: interior bins are
`2 * abs(rfft(window * detrended_signal)) / sum(window)`. DC and even-length
Nyquist bins are not doubled. A bin-centered, well-separated sinusoid of 2 g peak
therefore reads approximately 2 g. Off-bin tones can leak and suffer scalloping;
flattop helps amplitude accuracy at the cost of broader peaks. Noise-bin amplitudes
depend on record length. Phase is relative to the first sample and a cosine reference;
phase at weak bins or for off-bin tones is not a reliable physical measurement.

Welch PSD averages complete, overlapping, windowed periodograms with normalization
`abs(rfft(window * segment))² / (fs * sum(window²))`, doubled at interior one-sided
bins. This is checked against [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
With acceleration in g, density units are g²/Hz; this is mean-square acceleration,
not mechanical energy in joules. FFT amplitude does not give the instantaneous peak.

Welch RMS is `sqrt(sum(PSD) * df)`, using rectangular frequency-bin quadrature so
DC/Nyquist contributions remain consistent with the discrete one-sided normalization.
Band RMS sums bins whose centers satisfy `LOW <= f < HIGH` (Nyquist is included
if HIGH equals Nyquist). Limits are thus quantized to bins and adjacent bands do not
double-count shared boundaries. Increase segment length for finer bands. Total and band
RMS are computed from linear density even when plotted on logarithmic axes.

Raw RMS includes DC. Default detrending removes the mean from the full record for
FFT, and **separately from each segment** for PSD, waterfall and spectrogram.
Welch RMS and time-domain detrended RMS can differ because of window weighting,
per-segment detrending, finite averages and discarded tail samples; they are not
forced to match. The summary records both, along with raw absolute peak and peak-to-peak.

Load segment spectra in Python (arrays have shape **time × frequency**):

```python
import numpy as np

data = np.load("results/01_accel_x/time_frequency.npz")
frequency = data["frequency_hz"]
time = data["time_s"]  # complete-window centers on the original CSV elapsed timeline
amplitude = data["peak_amplitude"]
psd = data["psd"]
# Spectrogram has its own, shorter window and denser time grid:
spec_frequency = data["spectrogram_frequency_hz"]
spec_time = data["spectrogram_time_s"]
spec_psd = data["spectrogram_psd"]
```

For scripting, import `read_csv` from `vibration_analysis.io` and `analyze` from
`vibration_analysis.analysis`. Plotting is separate in `vibration_analysis.plots`.

## Development

```powershell
uv sync --locked --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover tone amplitude and phase, DC/Nyquist scaling, PSD normalization against
SciPy, white-noise RMS, band RMS, CSV variants, invalid sampling and end-to-end exports.
GitHub Actions runs checks on Windows and Linux with Python 3.11–3.13.

Put real recordings in `data/` and exports in `results/` (both ignored by Git).
Re-running a channel to the same output path overwrites matching exports. Use a
different `--output` per recording/settings combination to keep comparisons.
The implementation is in-memory and intended for moderate recordings; segment arrays
can be large for very long acquisitions. Choose lower overlap or split long runs if needed.

testcommit

