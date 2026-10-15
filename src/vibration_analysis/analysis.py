"""Spectral normalization lives here, separate from plotting and CSV conventions."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class Analysis:
    fft_frequency: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    psd: np.ndarray
    time: np.ndarray
    window_amplitude: np.ndarray
    window_psd: np.ndarray
    segment_samples: int
    hop_samples: int
    summary: dict
    spectrogram_frequency: np.ndarray
    spectrogram_time: np.ndarray
    spectrogram_psd: np.ndarray


def _detrend(values, mode):
    return values if mode == "none" else signal.detrend(values, type=mode, axis=-1)


def _one_sided_factor(n):
    factor = np.full(n // 2 + 1, 2.0)
    factor[0] = 1.0
    if n % 2 == 0:
        factor[-1] = 1.0
    return factor


def _segment_spectra(x, fs, n, overlap, window, detrend):
    hop = n - int(n * overlap)
    win = signal.get_window(window, n, fftbins=True)
    starts = np.arange(0, len(x) - n + 1, hop)
    frame_view = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    shape = (len(starts), n // 2 + 1)
    amplitude = np.empty(shape)
    density = np.empty(shape)
    factor = _one_sided_factor(n)
    for first in range(0, len(starts), 128):
        frames = _detrend(frame_view[first : first + 128], detrend)
        magnitude = np.abs(np.fft.rfft(frames * win, axis=-1))
        amplitude[first : first + len(frames)] = magnitude * factor / win.sum()
        density[first : first + len(frames)] = magnitude**2 * factor / (fs * np.sum(win**2))
    return amplitude, density, starts, hop


def analyze(
    values,
    sample_rate,
    *,
    window="hann",
    detrend="constant",
    segment_seconds=1.0,
    overlap=0.5,
    bands=(),
    psd_highpass_hz=0.0,
    psd_filter_order=4,
    spectrogram_seconds=0.1,
    spectrogram_overlap=0.8,
):
    """FFT peak amplitude, Welch PSD and complete-window time-frequency spectra.

    Only complete segments are used; the final partial segment is discarded.
    Frames are processed in bounded batches to avoid allocating a huge window matrix.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Signal must be a finite 1-D array with at least four samples.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive.")
    if detrend not in {"none", "constant", "linear"}:
        raise ValueError("Detrend must be none, constant or linear.")
    if not np.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("Segment duration must be finite and positive.")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")
    if not np.isfinite(psd_highpass_hz) or not 0 <= psd_highpass_hz < sample_rate / 2:
        raise ValueError("PSD high-pass must be 0 (off) or below Nyquist.")
    if not isinstance(psd_filter_order, (int, np.integer)) or psd_filter_order < 1:
        raise ValueError("PSD filter order must be a positive integer.")
    if not np.isfinite(spectrogram_seconds) or spectrogram_seconds <= 0:
        raise ValueError("Spectrogram window duration must be finite and positive.")
    if not np.isfinite(spectrogram_overlap) or not 0 <= spectrogram_overlap < 1:
        raise ValueError("Spectrogram overlap must be between 0 and 1 (exclusive).")
    requested_spec = round(spectrogram_seconds * sample_rate)
    if requested_spec < 4:
        raise ValueError("Spectrogram duration must contain at least four samples.")
    density_signal = x
    if psd_highpass_hz:
        sos = signal.butter(
            psd_filter_order, psd_highpass_hz, btype="highpass", fs=sample_rate, output="sos"
        )
        try:
            density_signal = signal.sosfiltfilt(sos, x)
        except ValueError as exc:
            raise ValueError(
                "Record too short for the PSD high-pass filter; disable it or use more samples."
            ) from exc
    requested = round(segment_seconds * sample_rate)
    if requested < 4:
        raise ValueError("Segment duration must contain at least four samples.")
    n = min(requested, len(x))
    win_full = signal.get_window(window, len(x), fftbins=True)
    transform = np.fft.rfft(_detrend(x, detrend) * win_full)
    amplitude = np.abs(transform) * _one_sided_factor(len(x)) / win_full.sum()
    frame_amplitude, frame_psd, starts, hop = _segment_spectra(
        x, sample_rate, n, overlap, window, detrend
    )
    if psd_highpass_hz:
        _, frame_psd, _, _ = _segment_spectra(
            density_signal, sample_rate, n, overlap, window, detrend
        )
    spec_n = min(requested_spec, len(x))
    _, spec_psd, spec_starts, spec_hop = _segment_spectra(
        density_signal, sample_rate, spec_n, spectrogram_overlap, window, detrend
    )
    frequency = np.fft.rfftfreq(n, 1 / sample_rate)
    psd = frame_psd.mean(axis=0)  # Welch: average complete-window periodograms.
    df = sample_rate / n
    # Sum density times bin width: handles DC/Nyquist bins consistently with Parseval.
    psd_rms = float(np.sqrt(psd.sum() * df))
    band_results = []
    for low, high in bands:
        if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high <= sample_rate / 2):
            raise ValueError(f"Band {low}:{high} must lie between 0 and Nyquist.")
        selected = (frequency >= low) & (frequency < high)
        if high == sample_rate / 2:
            selected[-1] = True
        if not selected.any():
            raise ValueError(f"No PSD bins in band {low}:{high}; use longer segments.")
        band_results.append(
            {
                "low_hz": low,
                "high_hz": high,
                "rms": float(np.sqrt(psd[selected].sum() * df)),
                "bin_count": int(selected.sum()),
            }
        )
    processed = _detrend(x, detrend)
    summary = {
        "samples": len(x),
        "sample_rate_hz": sample_rate,
        "record_duration_s": len(x) / sample_rate,
        "timestamp_span_s": (len(x) - 1) / sample_rate,
        "nyquist_hz": sample_rate / 2,
        "fft_bin_width_hz": sample_rate / len(x),
        "psd_bin_width_hz": df,
        "window": window,
        "detrend": detrend,
        "segment_samples": n,
        "requested_segment_seconds": segment_seconds,
        "segment_was_shortened": requested > len(x),
        "hop_samples": hop,
        "actual_overlap": 1 - hop / n,
        "segment_count": len(starts),
        "unused_tail_samples": int(len(x) - (starts[-1] + n)),
        "mean": float(x.mean()),
        "raw_rms": float(np.sqrt(np.mean(x**2))),
        "detrended_rms": float(np.sqrt(np.mean(processed**2))),
        "peak_absolute": float(np.max(np.abs(x))),
        "peak_to_peak": float(np.ptp(x)),
        "welch_rms": psd_rms,
        "bands": band_results,
        "psd_highpass_hz": psd_highpass_hz,
        "psd_filter_order_per_pass": psd_filter_order,
        "psd_filter_effective_order": 2 * psd_filter_order if psd_highpass_hz else None,
        "psd_filter": f"Butterworth order {psd_filter_order}, forward-backward SOS (-6 dB at nominal cutoff)"
        if psd_highpass_hz
        else "off",
        "psd_filter_scope": "Welch PSD, PSD-derived RMS and spectrogram only",
        "spectrogram_segment_samples": spec_n,
        "spectrogram_segment_seconds": spec_n / sample_rate,
        "spectrogram_bin_width_hz": sample_rate / spec_n,
        "spectrogram_hop_seconds": spec_hop / sample_rate,
        "spectrogram_overlap": 1 - spec_hop / spec_n,
        "spectrogram_segment_count": len(spec_starts),
        "spectrogram_segment_was_shortened": requested_spec > len(x),
    }
    return Analysis(
        np.fft.rfftfreq(len(x), 1 / sample_rate),
        amplitude,
        np.angle(transform),
        frequency,
        psd,
        (starts + n / 2) / sample_rate,
        frame_amplitude,
        frame_psd,
        n,
        hop,
        summary,
        np.fft.rfftfreq(spec_n, 1 / sample_rate),
        (spec_starts + spec_n / 2) / sample_rate,
        spec_psd,
    )
