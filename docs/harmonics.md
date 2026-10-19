# Harmonic amplitude versus RPM (desktop app)

Load vibration and **mRPM CSV (analysis/detection)** in the original desktop app.
Select FFT or Waterfall, then run the analysis as usual. Each test's
`comparisons/<sensor>/` folder now also contains:

- `harmonic_amplitude_vs_rpm.png`: all selected orders overlaid.
- `harmonic_amplitude_vs_rpm_panels.png`: one panel per order, with point numbers.
- `harmonic_peaks.csv`: expected/observed frequencies, peak amplitudes, detuning,
  actual search half-width and match status for every operating point/order.
- `harmonic_tracking.json`: the extraction settings and interpretation.

The existing broadband/trimmed 2D FFT waterfalls show dashed expected order
frequencies with order labels and markers on the selected local peaks. The 3D
waterfalls connect the selected candidates for each order.

Under **Analysis settings**, adjust **Shaft orders (comma-separated)** and
**Order search ± [Hz]**. Defaults are orders 1–10 and ±5 Hz. Clear the order list
to disable tracking. Fractional orders such as `0.5,1,1.5,2,7` are allowed.

An order is relative to shaft rotation: `frequency = order × mean RPM / 60`.
For example, 7× is 700 Hz at 6000 RPM. The combustion fundamental depends on the
engine configuration; this tool does not infer cylinder count or firing pattern.

Within the search band, the strongest **local maximum** in the existing
unfiltered, window-averaged FFT magnitude is selected. The half-width is at least
one FFT bin, so rounding to frequency bins does not reject a valid peak. Search
widths spanning half the neighbouring order spacing are rejected as ambiguous.
No peak or a predicted frequency above the sampled bandwidth produces a missing
value, not zero. No peak interpolation, smoothing, RMS conversion or signal
filtering is added. Each point retains its original measured amplitude; repeated
RPMs are not averaged. Lines are guides between measured points, not fitted curves.

Candidates may include unrelated tones or noise within the search band. Review
the waterfall overlays and frequency detuning before interpreting them. An
amplitude rise can flag a possible resonance but also depends on excitation/load,
mounting and clipping. These plots alone do not establish a natural frequency.
