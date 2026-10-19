"""Region spectra: original samples, bounded FFT batches, explicit display pooling."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from .analysis import _detrend, _one_sided_factor


@dataclass(frozen=True)
class Settings:
    segment_seconds: float = 0.5
    overlap: float = 0.5
    window: str = "hann"
    detrend: str = "constant"
    highpass_hz: float = 10
    filter_order: int = 4
    spectrogram_seconds: float = 0.1
    spectrogram_overlap: float = 0.8
    fft_mode: str = "averaged"
    clip_level: float = 50

    @classmethod
    def from_dict(cls, values):
        settings = cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})
        numbers = (
            settings.segment_seconds,
            settings.spectrogram_seconds,
            settings.highpass_hz,
            settings.overlap,
            settings.spectrogram_overlap,
            settings.clip_level,
        )
        if not all(np.isfinite(v) for v in numbers):
            raise ValueError("Analysis settings must be finite.")
        if settings.segment_seconds <= 0 or settings.spectrogram_seconds <= 0:
            raise ValueError("Window durations must be positive.")
        if not 0 <= settings.overlap < 1 or not 0 <= settings.spectrogram_overlap < 1:
            raise ValueError("Overlap must be between zero (inclusive) and one (exclusive).")
        if settings.highpass_hz < 0 or settings.clip_level <= 0:
            raise ValueError("High-pass must be nonnegative; peak threshold must be positive.")
        if not isinstance(settings.filter_order, int) or not 1 <= settings.filter_order <= 20:
            raise ValueError("Butterworth order must be an integer from 1 to 20.")
        if settings.window not in {"hann", "hamming", "flattop", "boxcar"}:
            raise ValueError("Unsupported FFT window.")
        if settings.detrend not in {"constant", "linear", "none"}:
            raise ValueError("Unsupported detrend setting.")
        if settings.fft_mode not in {"averaged", "full"}:
            raise ValueError("FFT method must be averaged windows or full record.")
        return settings


@dataclass
class RegionResult:
    region: dict
    settings: dict
    fft_frequency: np.ndarray
    fft_amplitude: np.ndarray
    frequency: np.ndarray
    psd: np.ndarray
    spectrogram_frequency: np.ndarray
    spectrogram_time: np.ndarray
    spectrogram_psd: np.ndarray
    summary: dict


def _raw_statistics(x, threshold, progress):
    total = square = 0.0
    peak, minimum, maximum, above = 0.0, np.inf, -np.inf, 0
    for first in range(0, len(x), 250000):
        block = np.asarray(x[first : first + 250000])
        total += float(block.sum())
        square += float(np.dot(block, block))
        minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
        peak = max(peak, abs(minimum), abs(maximum))
        above += int(np.sum(np.abs(block) >= threshold))
        progress("Measuring original time-domain samples…")
    mean = total / len(x)
    return {
        "mean": mean,
        "raw_rms": float(np.sqrt(square / len(x))),
        "ac_rms": float(np.sqrt(max(0, square / len(x) - mean**2))),
        "peak": peak,
        "minimum": minimum,
        "maximum": maximum,
        "threshold": threshold,
        "threshold_percent": 100 * above / len(x),
    }


def _window_average(x, density_signal, fs, n, overlap, window, detrend, progress):
    hop = n - int(n * overlap)
    win = signal.get_window(window, n, fftbins=True)
    factor = _one_sided_factor(n)
    amplitudes = np.zeros(len(factor))
    densities = np.zeros(len(factor))
    starts = np.arange(0, len(x) - n + 1, hop)
    raw_view = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    filtered_view = np.lib.stride_tricks.sliding_window_view(density_signal, n)[::hop]
    for first in range(0, len(starts), 128):
        raw_fft = np.abs(
            np.fft.rfft(_detrend(raw_view[first : first + 128], detrend) * win, axis=-1)
        )
        amplitudes += raw_fft.sum(axis=0) * factor / win.sum()
        if density_signal is x:
            psd_fft = raw_fft
        else:
            psd_fft = np.abs(
                np.fft.rfft(_detrend(filtered_view[first : first + 128], detrend) * win, axis=-1)
            )
        densities += (psd_fft**2).sum(axis=0) * factor / (fs * np.sum(win**2))
        progress(f"Window spectra: {min(first + 128, len(starts)):,} / {len(starts):,}…")
    return amplitudes / len(starts), densities / len(starts), len(starts), hop


def _spectrogram(x, fs, n, overlap, window, detrend, progress, max_times=700, max_frequencies=900):
    """Average linear PSD into display cells, never into analysis statistics."""
    hop = n - int(n * overlap)
    starts = np.arange(0, len(x) - n + 1, hop)
    view = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    win = signal.get_window(window, n, fftbins=True)
    factor = _one_sided_factor(n)
    frequency = np.fft.rfftfreq(n, 1 / fs)
    rows, columns = min(max_times, len(starts)), min(max_frequencies, len(frequency))
    frequency_groups = np.minimum(
        np.arange(len(frequency)) * columns // len(frequency), columns - 1
    )
    frequency_counts = np.bincount(frequency_groups, minlength=columns)
    display_frequency = np.bincount(frequency_groups, weights=frequency) / frequency_counts
    display = np.zeros((rows, columns))
    counts = np.zeros(rows)
    times = np.zeros(rows)
    for first in range(0, len(starts), 128):
        frames = _detrend(view[first : first + 128], detrend)
        magnitude = np.abs(np.fft.rfft(frames * win, axis=-1))
        density = magnitude**2 * factor / (fs * np.sum(win**2))
        pooled = (
            np.add.reduceat(
                density, np.r_[0, np.flatnonzero(np.diff(frequency_groups)) + 1], axis=1
            )
            / frequency_counts
        )
        groups = np.arange(first, first + len(frames)) * rows // len(starts)
        np.add.at(display, groups, pooled)
        np.add.at(counts, groups, 1)
        np.add.at(times, groups, (starts[first : first + len(frames)] + n / 2) / fs)
        progress(f"Spectrogram: {min(first + 128, len(starts)):,} / {len(starts):,} frames…")
    return display_frequency, times / counts, display / counts[:, None], len(starts), hop


def compute_region(dataset, region, settings, progress=lambda _: None):
    recording = dataset.recording
    fs = recording.sample_rate
    if settings.highpass_hz >= fs / 2:
        raise ValueError("PSD high-pass cutoff must be below Nyquist.")
    first = int(np.searchsorted(recording.time, region["start"], side="left"))
    stop = int(np.searchsorted(recording.time, region["end"], side="left"))
    if stop - first < 4:
        raise ValueError("A region must contain at least four original samples.")
    x = recording.channels[region["channel"]][first:stop]
    n = min(round(settings.segment_seconds * fs), len(x))
    spec_n = min(round(settings.spectrogram_seconds * fs), len(x))
    if min(n, spec_n) < 4:
        raise ValueError("Window durations must contain at least four samples.")
    statistics = _raw_statistics(x, settings.clip_level, progress)
    density_signal = x
    if settings.highpass_hz:
        progress("Applying the PSD/spectrogram high-pass filter…")
        sos = signal.butter(
            settings.filter_order, settings.highpass_hz, btype="highpass", fs=fs, output="sos"
        )
        try:
            density_signal = signal.sosfiltfilt(sos, x)
        except ValueError as exc:
            raise ValueError(
                "Region is too short for the high-pass filter; disable it or expand the region."
            ) from exc
    amplitude, psd, count, hop = _window_average(
        x, density_signal, fs, n, settings.overlap, settings.window, settings.detrend, progress
    )
    frequency = np.fft.rfftfreq(n, 1 / fs)
    fft_frequency = frequency
    if settings.fft_mode == "full":
        progress("Calculating the full-record FFT…")
        win = signal.get_window(settings.window, len(x), fftbins=True)
        amplitude = (
            np.abs(np.fft.rfft(_detrend(x, settings.detrend) * win))
            * _one_sided_factor(len(x))
            / win.sum()
        )
        fft_frequency = np.fft.rfftfreq(len(x), 1 / fs)
    sf, st, sp, spec_count, spec_hop = _spectrogram(
        density_signal,
        fs,
        spec_n,
        settings.spectrogram_overlap,
        settings.window,
        settings.detrend,
        progress,
    )
    statistics.update(
        samples=len(x),
        sample_rate=fs,
        nyquist=fs / 2,
        first_time=float(recording.time[first]),
        last_time=float(recording.time[stop - 1]),
        duration=len(x) / fs,
        welch_rms=float(np.sqrt(psd.sum() * fs / n)),
        fft_bin_hz=float(fft_frequency[1] - fft_frequency[0]),
        psd_bin_hz=fs / n,
        window_samples=n,
        window_seconds=n / fs,
        segment_count=count,
        hop_seconds=hop / fs,
        spec_bin_hz=fs / spec_n,
        spec_hop_seconds=spec_hop / fs,
        spec_frames=spec_count,
        spectrogram_display_shape=list(sp.shape),
        display_pooling="Linear PSD averaged into display cells; analysis spectra use all complete windows",
        unit=dataset.config["unit"],
        source=dataset.config["input"],
    )
    if dataset.mrpm is not None:
        t, rpm = dataset.mrpm
        inside = (t >= region["start"]) & (t < region["end"])
        statistics["mean_rpm"] = float(rpm[inside].mean()) if inside.any() else None
    return RegionResult(
        dict(region),
        asdict(settings),
        fft_frequency,
        amplitude,
        frequency,
        psd,
        sf,
        st + recording.time[first],
        sp,
        statistics,
    )
