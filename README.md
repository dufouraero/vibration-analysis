# Vibration analysis

An interactive workspace for vibration CSVs: select time regions, recalculate
FFT/PSD/spectrograms, and compare operating points or recordings from different days.
Your data stays on your computer.

## Open the dashboard

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once.
2. Clone or download this repository. Extract the ZIP if you downloaded one.
3. Double-click **Open Interactive Dashboard.vbs**.

The launcher installs Python and dependencies automatically, then opens the
dashboard in your browser. First launch needs internet; later analysis works
offline. No terminal commands are needed.

**In VS Code:** open the repository folder and press **Ctrl+Shift+B**, or choose
**Interactive vibration dashboard** under Run and Debug.

## Use it

1. Paste or browse to your **Vibration CSV**, then click **Load recording**.
   Headerless exports are detected automatically; expand **CSV and calibration**
   for overrides. The strongest sensor is suggested and all imported sensors remain available.
2. Drag across the time history to select a region, give it a name and click
   **Add region**. You can also enter start/end seconds directly. With an mRPM
   log, **Suggest mRPM holds** adds steady operating points with configurable end trims.
3. Tick the regions to compare and click **Calculate / compare**. Edit bounds
   or window/filter settings and calculate again to recompute from original samples.
4. Explore **PSD, FFT, Waterfall, Spectrogram and Time histories**. Click legend
   entries to hide/show curves. FFTs support overlays or a separate row per region.
   Set frequency limits, choose **Common bandwidth**, or view **Broadband**.
5. Load another recording to compare it with regions already in the table.
   **Download session JSON** stores the sources, regions and settings for later restoration.
   **Export interactive HTML** saves a shareable report and numerical spectra.

Try **Load demo** first to explore without your own CSVs. Use **Stop dashboard**
when finished; closing only the browser tab leaves the local server running.

## Optional logs

| Input | Purpose |
| --- | --- |
| **mRPM CSV** | Alignment, steady-point detection and mean RPM labels. Keep using the mean-RPM export here. |
| **Raw RPM CSV** | Unfiltered RPM time plot only; does not replace mRPM. |
| **Voltage / current CSV** | Unfiltered electrical time plot. Voltage is preferred automatically. |

Telemetry needs the vibration recording's start timestamp, inferred from the
original filename or entered under **mRPM alignment and hold suggestions**.
Check the timeline after automatic alignment. Raw RPM and voltage can be used
without mRPM when the start timestamp and manual offset are supplied.

## Comparing results

The default FFT averages **0.5 s Hann-window magnitudes**, giving roughly **2 Hz
bin spacing** at either sampling rate. Welch PSD uses the same configurable
duration and overlap. A full-record FFT is also available; its peaks can differ
when speed drifts. PSD/spectrogram high-pass filtering is configurable and does
not alter the raw time history or FFT.

The statistics table lists raw RMS, PSD full-band RMS, displayed-band RMS,
sample rate, peaks and the percentage above a chosen peak threshold. Use a
**common frequency band** when comparing recordings with different bandwidths.
Exceeding a threshold alone does not prove clipping.

## Saved work and large files

- CSVs are cached locally as disk-backed arrays under **.dashboard-cache/**.
  First import scans the CSV twice; repeat loads reuse the cache. Time plots
  preserve bucket minima/maxima for display; spectra use original samples.
- Spectrogram display cells average linear PSD to keep the browser responsive.
  Select a shorter region and recalculate to inspect shorter events.
- HTML exports go into a new timestamped folder under **results/dashboard/**.
  Reports work offline with zoom and curve toggles. Recalculating new time
  regions requires the running dashboard and original CSVs.
- Session files reference local source paths. Keep the original CSVs; changed
  source files are rejected during restore so results are not silently changed.
- Results and cache files are ignored by Git. With the dashboard stopped, you
  can delete **.dashboard-cache/** to reclaim disk space; CSVs are unaffected.

## Existing desktop app

**Open Vibration Analysis.vbs** still opens the original desktop window and PNG
batch workflow. See the [desktop guide](docs/desktop-app.md) for its controls.
The [advanced reference](docs/reference.md) covers command-line options.

Use GitHub Desktop **Pull origin** to update, then reopen the launcher. It syncs
dependencies automatically; it does not pull Git changes itself. Setup errors
appear in **setup.log**; dashboard details are in **.dashboard-cache/dashboard.log**.
