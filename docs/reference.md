# Advanced reference

For normal use, follow the [desktop guide](../README.md). This document covers
command-line options, CSV formats, analysis methods, numerical exports and development.
Run commands from the repository folder.

**Input terminology:** `--mrpm-csv` (legacy alias `--rpm-csv`) is the **mRPM
analysis source**. It drives alignment, tests, holds and operating-point means.
`--raw-rpm-csv` is separate and supplies only the simple RPM/voltage time plot.

## Command-line setup

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
| `--spectrogram-range-db` | Color range below upper display limit when lower is auto; 60 dB |
| `--spectrogram-headroom-db` | Automatic upper limit above the peak PSD; 6 dB |
| `--spectrogram-lower-db`, `--spectrogram-upper-db` | Optional fixed PSD colour limits, dB re 1 signal-unit²/Hz |
| `--no-spectrogram-rpm` | Hide mean RPM operating-point labels (shown by default with an RPM log) |
| `--spectrogram-colormap` | `viridis` (default), `magma`, `turbo` |
| `--spectrogram-interpolation` | `bilinear` display blending (default), or `nearest` to see individual cells |
| `--start-seconds`, `--end-seconds` | Select a half-open source interval; default entire recording |
| `--trim-start-seconds`, `--trim-end-seconds` | Trim inside selected interval; CLI default 0 each, desktop default 5 s each |

### Grafana local clock-time window

In the desktop **Grafana time window** tab, enter optional local start/end values
using the same clock as the Grafana dashboard. For example:

```text
Grafana start:    2026-10-14 16:01:00
Grafana end:      2026-10-14 16:09:30
Display timezone: Europe/Zurich
```

You can also enter `16:01:00` and `16:09:30`; time-only entries use the first date
of the RPM export in that local clock. For overnight or multi-day recordings, use
full dates. A blank bound leaves that side unrestricted. The timezone is an IANA
name such as `Europe/Zurich` or `UTC`; match the Grafana display setting.

An RPM CSV is required for this clock-time selection. The app first aligns the
timelines, then converts your requested bounds to elapsed vibration time using
the estimated/manual offset. **Do not manually add the alignment offset to the
times you enter.** The Grafana window intersects any elapsed-time interval limits.
Whole-test and per-point plots, FFT/PSD exports and RMS statistics are restricted
to that intersection; the end is exclusive. Hold trims are applied inside it.
Alignment can inspect the full recording for shared starts/stops, but samples
outside your selected window do not enter the exported spectra or statistics.
`alignment.json` records both requested local bounds and the final source-time bounds.

CLI equivalents are `--grafana-start`, `--grafana-end`, and `--grafana-timezone`.

### Elapsed interval and trim

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

### mRPM CSV, detected tests, operating points and comparisons

Choose the vibration CSV and the optional **mRPM CSV (analysis/detection)** in the popup. The **mRPM detection**
tab provides automatic/manual alignment and configurable detection thresholds.
A Grafana RPM export with a `Time` column and one numeric RPM column is supported
directly. Other column names can be selected explicitly. mRPM drives detection. Separate raw RPM and voltage/current inputs supply the optional
operating-condition time plot; see the input-role section below.
Identical repeated timestamp/RPM rows (for example a duplicated final Grafana row)
are removed automatically and counted in the import metadata. Conflicting RPM values
at the same timestamp and backward time steps are rejected rather than silently merged.

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
          fft_rpm_waterfall_trimmed.png
          fft_rpm_waterfall_2d_trimmed.png
          psd_comparison_trimmed.png
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

The **2D waterfall** is a single figure with one stacked FFT line plot per detected
RPM hold, sharing the frequency axis and physical amplitude scale. Rows are numbered
chronologically and labelled with their mean RPM. No per-row amplitude normalization
is applied, so vibration levels remain comparable. It uses the same averaged window
magnitudes as the 3D comparison and is saved automatically when FFT or waterfall plots
are selected in RPM mode. These are detected operating points, not arbitrary bins
through a ramp; repeat visits to the same RPM remain separate rows.

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
Timezone-aware timestamps retain their actual instants. Naive RPM timestamps and the
vibration filename/start timestamp use the chosen Grafana display timezone (default
Europe/Zurich). If the logger uses a different clock zone, specify an explicit offset
in its start timestamp. Five-second telemetry cannot establish precise
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
| `fft_broadband.png`, `fft_trimmed.png`, `fft.csv` | Full-record one-sided **peak** amplitude spectrum plus phase in radians in CSV |
| `psd_linear_x_broadband.png`, `psd_linear_x_trimmed.png` | Welch PSD, linear frequency axis, log density axis |
| `psd_log_x_broadband.png`, `psd_log_x_trimmed.png` | Same Welch PSD, log frequency axis, log density axis; DC omitted |
| `psd.csv` | All one-sided frequency bins and PSD in unit²/Hz, including DC |
| `waterfall_broadband.png`, `waterfall_trimmed.png` | 3D sequence of windowed FFT peak amplitude spectra versus time |
| `spectrogram_broadband.png`, `spectrogram_trimmed.png` | Time–frequency PSD heatmap, logarithmic density color scale |
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

### Spectrogram contrast and RPM

The Spectrogram tab controls both colour limits. Blank upper uses the strongest
displayed PSD plus 6 dB of headroom; blank lower uses upper minus Color range.
Increase the upper limit if strong features look too bright; adjust the lower
limit to reveal or suppress weak background features. Fixed limits allow fair
visual comparisons between tests and operating points. Colourbar triangles
indicate values outside the chosen display range. These controls only change
the display; exported PSD values are unchanged.

With an RPM CSV, spectrograms show each recognised operating point's mean RPM
beneath its corresponding time interval. The labels use the same trimmed holds
as the FFT operating-point comparison; ramps and unclassified periods stay blank.
Clear **Show mean RPM operating points underneath** to hide the labels.
Summary JSON records the displayed intervals, means and actual colour limits.

Decimal timestamp rounding is accepted when the entire timestamp sequence fits
a uniform sampling grid within one timestamp rounding unit. This handles, for
example, 17,066.7 Hz data exported with six decimal places in seconds (58/59 µs
steps). Missing samples and departures from that grid still trigger timing
validation. Sample rate uses the full timestamp span; signal samples are not
resampled or changed.


### Broadband and limited 2D FFT waterfalls

Each operating-point comparison saves two 2D waterfall plots:

- `fft_rpm_waterfall_2d_broadband.png`: 0 Hz to Nyquist, regardless of other plot limits.
- `fft_rpm_waterfall_2d_trimmed.png`: the selected frequency view. Set **Analysis settings → 2D waterfall max [Hz]** (`--waterfall2d-fmax`). Blank uses the general plot maximum, then Nyquist if both are blank.

All operating-point rows in each image share the same amplitude scale. Each
image scales amplitudes for its own visible frequency range, allowing the limited
view to reveal weaker features. Numerical exports retain the full spectrum.

### Separate mRPM analysis and raw RPM time-plot inputs

The desktop file fields have distinct roles:

| Input | Used for |
| --- | --- |
| **mRPM CSV (analysis/detection)** | Timeline alignment, test/hold detection, operating-point means, FFT/PSD comparisons and spectrogram mean RPM labels |
| **Raw RPM CSV (time plot only)** | RPM row in the simple operating-condition time plot only |
| **Voltage/current CSV (optional)** | Electrical row in the simple time plot only |

Keep using the earlier **mRPM** export for analysis. Add the new **raw RPM**
export in the additional field. The raw log never replaces the mRPM detection
source, changes alignment or influences operating-point selection. Leaving raw
RPM blank omits that row; it does not fall back to the mRPM log.

`operating_conditions.png` shows raw RPM above voltage/current on a shared
Grafana local-time axis. One supplied plot source produces one row; neither
produces no operating-condition plot. Plots are saved at the result root and in
each recognised test folder. Logged samples retain their native timestamps and
values without smoothing or resampling. Large telemetry gaps break the line.

The **Operating data CSVs** tab provides optional column overrides. Auto prefers
a voltage column; a current-only electrical export is labelled **Current [A]**.
The selected vibration interval is mapped back to the Grafana clock using the
mRPM alignment offset before clipping these plots. Without an mRPM CSV, a known
vibration start clips the optional logs to the retained vibration interval;
without a start, the full logs are plotted with an explanatory progress note.
CSV data and JSON source/column provenance accompany the image.

CLI: `--mrpm-csv` is the analysis input (`--rpm-csv` remains a compatibility
alias). The separate `--raw-rpm-csv` is only for the simple plot, with optional
`--raw-rpm-time-column` and `--raw-rpm-column`. Electrical options are
`--electrical-csv`, `--electrical-time-column`, `--electrical-column`, and
`--electrical-quantity auto|voltage|current`.

The display-only electrical importer preserves multiple readings at the same
exported timestamp, including differing values. Such readings appear at the same
x coordinate; no averaging, row selection or invented timestamp is applied.
A progress note and plot JSON report repeated timestamps. This does not relax
the independent mRPM importer used for alignment and hold detection.

### Paired broadband and trimmed frequency plots

All selected frequency plots save two images, in whole tests and individual
operating points: `*_broadband.png` covers 0 Hz to Nyquist, and `*_trimmed.png`
uses `--fmin`/`--fmax` (GUI **Trimmed frequency min/max [Hz]**). Log-frequency PSD
views start at the first positive bin when the lower limit is zero. A blank
maximum uses Nyquist, so the two views may be identical.

This applies to FFT, linear/log-frequency PSD, spectrogram, time waterfall,
operating-point 3D/2D waterfalls and operating-point PSD comparisons. The 2D
waterfall retains its dedicated `--waterfall2d-fmax` override for the trimmed
view. Spectrograms crop frequency on the vertical axis; their time range is
unchanged, and the pair shares one colour scale calculated from the full band.
Time-history and raw RPM/electrical plots retain their chosen time windows.

The PSD titles distinguish RMS integrated over the displayed band from full-band
RMS. `summary.json` records `frequency_plot_views_hz`, `psd_display_rms` and
shared spectrogram colour limits. Frequency limits change the view, not the input
signal. Numerical exports remain full bandwidth. Spectra are calculated once
and rendered twice; additional PNG exports take more rendering time.

The full-record FFT title reports its record duration and bin spacing. It uses
one transform over the entire interval. RPM comparison waterfalls average
short-window FFT magnitudes (default 0.5 s windows), so their peak heights can
differ for drifting tones and intermittent signals. Longer duration alone does
not reduce the amplitude of a steady bin-centred sinusoid.
