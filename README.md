# Vibration analysis

A desktop app for vibration CSVs: FFT, Welch PSD (linear and log frequency axes),
waterfalls, spectrograms and time traces. Optional mRPM data identifies tests and
operating points; separate raw RPM and voltage logs provide a simple time plot.

## Open the app

**First use on Windows**

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once.
2. Download or clone this repository to your computer. Extract the ZIP first if you downloaded one.
3. Double-click **Open Vibration Analysis.vbs** in the repository folder.

The launcher prepares Python and the required packages automatically. The first
launch needs an internet connection and may take a few minutes. No manual
`uv sync` or terminal commands are needed.

**Everyday use:** double-click the same launcher.

**From VS Code:** open the repository folder and press **Ctrl+Shift+B** to run
**Open Vibration Analysis**. This opens the same desktop window.

## Choose your files

Only the vibration CSV is required. The three optional logs have different jobs:

| File field | What it does |
| --- | --- |
| **CSV file** | Vibration sensor recording to analyse. |
| **mRPM CSV (analysis/detection)** | Aligns timelines and detects tests and steady operating points. Supplies the mean RPM labels used in comparisons. Keep using your mRPM export here. |
| **Raw RPM CSV (time plot only)** | Supplies the RPM row of the simple operating-condition plot. Never used for alignment or detection. |
| **Voltage/current CSV (optional)** | Supplies the electrical row of that simple plot. Voltage is preferred automatically; current-only exports are labelled in amperes. |

Leave unused inputs blank. Raw RPM does not replace the mRPM input. The simple
plot includes only the supplied raw RPM/electrical sources, without filtering or
averaging. Try **Try synthetic demo** if you just want to check the app opens and plots.

## Run an analysis

1. Choose the vibration CSV and any optional logs using **Browse…** or paste their paths.
2. Check **CSV settings**. For a headerless export, tick **CSV has no header** and set **Time column** to `0`. Sensor columns then start at `1`. Otherwise leave the time column blank for automatic detection. Leave sample rate blank when the CSV contains usable time values.
3. Select sensors. **Auto activity** suggests active inputs; use **Choose sensor channels…** to select them yourself.
4. If using mRPM, check **mRPM detection**. A timestamp in the vibration filename supplies the recording start. Otherwise enter **Vibration start timestamp** in the same local clock as Grafana. Auto alignment estimates the remaining clock offset.
5. Optionally select a **Grafana time window**, using Grafana's local start/end times and display timezone. This requires the mRPM input. Alternatively use elapsed seconds under **Interval & trim**.
6. Tick the plots you want and click **Run selected analyses**.
7. When complete, click **Open results folder**. With mRPM, check `detection_overview.png` before trusting the suggested operating points.

The **Progress** tab shows the main stages. Open **Technical details** if there is
an error or you need more information. Large recordings can take time to read and plot.

## Useful settings

Defaults are a starting point; these controls are in the window:

| Tab | Main controls |
| --- | --- |
| **Analysis settings** | Welch window length, overlap and window type; PSD high-pass cutoff/order; **Trimmed frequency min/max [Hz]**; **2D waterfall max [Hz]**. |
| **Spectrogram** | Independent window/overlap, colour range and lower/upper PSD limits. Increase the upper limit or automatic headroom to reduce bright saturation. |
| **Interval & trim** | Elapsed-time bounds and trimming. With mRPM, the default 5 s trim at each end applies to each detected hold; whole-test plots retain transitions. |
| **mRPM detection** | Alignment and minimum test/hold duration, RPM drift and scatter limits. |
| **Operating data CSVs** | Optional raw RPM and electrical column overrides. Usually leave these on automatic detection. |

All frequency plots save **both broadband and trimmed-frequency views**, named
`*_broadband.png` and `*_trimmed.png`. This includes FFTs, PSDs (linear and log
frequency axes), spectrograms and waterfalls, in whole tests and operating points.
Set **Trimmed frequency min/max [Hz]** for the trimmed view; leave maximum blank
for Nyquist. Spectrograms crop the frequency axis and share colour limits between
views. PSD titles state both displayed-band and full-band RMS. Time plots keep
the selected time interval.

For the 2D operating-point waterfall, set **2D waterfall max [Hz]** to override the trimmed maximum; blank uses the general
trimmed maximum. Broadband always extends from 0 Hz to Nyquist.

## Find your results

Each desktop run creates a new folder under your chosen output folder, named
from the vibration CSV plus a timestamp. Previous runs are preserved.

With mRPM, each detected test contains:

- **whole_test/** — plots for the full test. Frequency plots include both broadband and trimmed views.
- **operating_points/** — plots for each detected, trimmed hold.
- **comparisons/** — operating-point FFT waterfalls and PSD comparisons, including both 2D waterfall views.

Sensor subfolders contain PNG plots and numerical exports. The optional simple
time plot is `operating_conditions.png`, saved at the run root and in each test
folder. Results are ignored by Git.

## Updates and troubleshooting

- **Update the app:** use **Fetch origin / Pull origin** in GitHub Desktop, then reopen the launcher. It updates dependencies automatically; it does not pull Git changes itself.
- **The launcher fails:** read the popup and `setup.log` in the repository folder. Confirm uv is installed; Windows must permit the launcher scripts.
- **Analysis fails:** open **Technical details**. A copy is saved as `analysis.log` in the result folder.
- **Missing vibration start timestamp:** enter the recording's actual start under **mRPM detection**, or retain the original timestamped export filename. Relative sample times alone do not provide the recording date.

For command-line use, detailed CSV options, analysis definitions and development,
see the [advanced reference](docs/reference.md).
