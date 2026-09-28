"""Resolved sinusoidal trajectories, estimated with a zero-padded Hann QIFFT.

See Julius O. Smith, Spectral Audio Signal Processing, 'Spectrum Analysis of
Sinusoids'. Configuration defaults are conservative product choices; they are
not universal definitions of musical content or hearing thresholds.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.ndimage import find_objects, label


@dataclass(frozen=True)
class TonalConfig:
    n_fft: int = 4096
    hop: int = 1024
    zero_padding: int = 4
    min_hz: float = 40.0
    max_hz: float = 16000.0
    prominence_db: float = 12.0
    dynamic_range_db: float = 60.0
    max_lobe_error: float = 0.15
    max_step_cents: float = 100.0
    min_track_frames: int = 4
    family_tolerance_cents: float = 50.0
    consistency_cents: float = 2.0
    stable_fraction: float = 0.75
    min_family_partials: int = 4
    fit_iterations: int = 8

    def __post_init__(self):
        if self.n_fft < 16 or self.n_fft % 2 or not 0 < self.hop <= self.n_fft // 2:
            raise ValueError("tonal analysis requires an even FFT and hop <= half the window")

        if self.zero_padding < 1 or self.min_track_frames < 3 or self.min_family_partials < 3:
            raise ValueError("tonal tracking requires padding >= 1 and at least three observations/partials")

        limits = (self.min_hz, self.max_hz, self.prominence_db, self.dynamic_range_db,
                  self.max_lobe_error, self.max_step_cents, self.family_tolerance_cents,
                  self.consistency_cents, self.stable_fraction, self.fit_iterations)

        if not np.all(np.isfinite(limits)) or min(limits) <= 0:
            raise ValueError("tonal configuration limits must be finite and positive")

        if self.max_hz <= self.min_hz or self.stable_fraction > 1 or self.max_lobe_error >= 1:
            raise ValueError("invalid tonal frequency range or confidence limits")


@dataclass
class PartialTrack:
    frames: list[int] = field(default_factory=list)
    frequencies: list[float] = field(default_factory=list)
    coefficients: list[np.ndarray] = field(default_factory=list)

    def append(self, frame, frequency, coefficient):
        self.frames.append(frame)
        self.frequencies.append(float(frequency))
        self.coefficients.append(coefficient)

    def uncertainty_cents(self, sample_rate: int, hop: int) -> float:
        """Observed disagreement between phase advance and spectral frequency."""
        coefficients = np.asarray(self.coefficients)
        channel = int(np.argmax(np.sum(np.abs(coefficients) ** 2, axis=0)))
        frequency = np.asarray(self.frequencies)
        midpoint = (frequency[1:] + frequency[:-1]) / 2
        advance = np.angle(coefficients[1:, channel] * coefficients[:-1, channel].conj())
        expected = 2 * np.pi * midpoint * hop / sample_rate
        error = np.angle(np.exp(1j * (advance - expected)))
        error_hz = np.abs(error) * sample_rate / (2 * np.pi * hop)
        return float(np.quantile(1200 * np.log2(1 + error_hz / midpoint), 0.95))

    def retain_channels(self, minimum_frames: int):
        """A tonal peak in one channel must not authorize editing noise in another."""
        coefficients = np.asarray(self.coefficients)

        for channel in range(coefficients.shape[1]):
            labels, _ = label(np.abs(coefficients[:, channel]) > 0)
            lengths = np.bincount(labels)
            valid = (labels > 0) & (lengths[labels] >= minimum_frames)
            coefficients[~valid, channel] = 0

        self.coefficients = list(coefficients)

    def supported_segments(self, minimum_frames: int):
        """Keep continuous observations after channel confidence removes short runs."""
        supported = np.any(np.abs(self.coefficients) > 0, axis=1)

        for (selection,) in find_objects(label(supported)[0]):
            if selection.stop - selection.start < minimum_frames:
                continue

            yield PartialTrack(self.frames[selection], self.frequencies[selection],
                               self.coefficients[selection])


class PartialTracker:
    """Track isolated Hann-shaped peaks with one-to-one frequency assignments."""

    def __init__(self, config: TonalConfig):
        self.config = config
        self.window = np.hanning(config.n_fft + 1)[:-1]
        self.fft_size = config.n_fft * config.zero_padding

    def analyze(self, audio: np.ndarray, sample_rate: int) -> tuple[list[PartialTrack], np.ndarray]:
        audio = np.asarray(audio, dtype=np.float64)

        if audio.ndim == 1:
            audio = audio[:, None]

        if audio.ndim != 2 or sample_rate <= 0 or not np.all(np.isfinite(audio)):
            raise ValueError("tonal analysis requires finite mono/stereo audio and a positive sample rate")

        half = self.config.n_fft // 2
        centers = np.arange(half, len(audio) - half + 1, self.config.hop)
        tracks: list[PartialTrack] = []
        active: list[int] = []

        for frame, center in enumerate(centers):
            frequency, coefficient = self.peaks(audio[center-half:center+half], sample_rate)
            active = self.continue_tracks(tracks, active, frame, frequency, coefficient)

        tracks = [track for track in tracks if len(track.frames) >= self.config.min_track_frames]

        for track in tracks:
            track.retain_channels(self.config.min_track_frames)

        return [segment for track in tracks
                for segment in track.supported_segments(self.config.min_track_frames)], centers

    def peaks(self, segment: np.ndarray, sample_rate: int):
        spectrum = np.fft.rfft(segment * self.window[:, None], n=self.fft_size, axis=0)
        bins = np.arange(len(spectrum))
        spectrum *= np.exp(1j * 2 * np.pi * bins * (self.config.n_fft // 2) / self.fft_size)[:, None]
        magnitude = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=1))
        floor = max(float(np.max(magnitude)) * 10 ** (-self.config.dynamic_range_db / 20), np.finfo(float).tiny)
        decibels = 20 * np.log10(np.maximum(magnitude, floor))
        peaks, _ = find_peaks(decibels, prominence=self.config.prominence_db)
        lobe_radius = 2 * self.config.zero_padding  # Hann's first zeros are +/- 2 bins.
        peaks = peaks[(peaks > lobe_radius) & (peaks < len(spectrum) - lobe_radius)]
        curvature = decibels[peaks-1] - 2 * decibels[peaks] + decibels[peaks+1]
        offset = (decibels[peaks-1] - decibels[peaks+1]) / (2 * curvature)
        frequency = (peaks + offset) * sample_rate / self.fft_size
        valid = (frequency >= self.config.min_hz) & (frequency <= self.config.max_hz)
        peaks, offset, frequency = peaks[valid], offset[valid], frequency[valid]
        lobe_bins = np.arange(-lobe_radius, lobe_radius + 1)
        delta = (lobe_bins[None, :] - offset[:, None]) / self.config.zero_padding
        # Exact finite, centered, periodic Hann transform normalized to its DC value.
        dirichlet = lambda value: np.exp(1j * np.pi * value / self.config.n_fft) * np.sinc(value) / np.sinc(value / self.config.n_fft)
        template = np.abs(dirichlet(delta) + (dirichlet(delta-1) + dirichlet(delta+1)) / 2)
        observed = magnitude[peaks[:, None] + lobe_bins]
        amplitude = np.sum(observed * template, axis=1) / np.sum(template ** 2, axis=1)
        error = np.linalg.norm(observed - amplitude[:, None] * template, axis=1) / np.maximum(np.linalg.norm(observed, axis=1), floor)
        valid = (error <= self.config.max_lobe_error) & (amplitude > floor)
        peaks, offset, frequency, error = peaks[valid], offset[valid], frequency[valid], error[valid]
        template = template[valid]
        channel_lobe = np.abs(spectrum[peaks[:, None] + lobe_bins])
        channel_amplitude = np.sum(channel_lobe * template[:, :, None], axis=1) / np.sum(template ** 2, axis=1)[:, None]
        channel_error = np.linalg.norm(channel_lobe - template[:, :, None] * channel_amplitude[:, None, :], axis=1)
        channel_error /= np.maximum(np.linalg.norm(channel_lobe, axis=1), floor)
        neighbor = peaks + np.sign(offset).astype(int)
        coefficient = (1 - np.abs(offset[:, None])) * spectrum[peaks] + np.abs(offset[:, None]) * spectrum[neighbor]
        coefficient = np.exp(1j * np.angle(coefficient)) * channel_amplitude * (2 / np.sum(self.window))
        coefficient[channel_error > self.config.max_lobe_error] = 0
        supported = np.any(np.abs(coefficient) > 0, axis=1)
        return frequency[supported], coefficient[supported]

    def continue_tracks(self, tracks, active, frame, frequency, coefficient):
        assigned: dict[int, int] = {}

        if active and len(frequency):
            predicted = np.array([tracks[index].frequencies[-1] for index in active])
            cost = np.abs(1200 * np.log2(frequency[None, :] / predicted[:, None]))
            rows, columns = linear_sum_assignment(np.minimum(cost, self.config.max_step_cents))
            assigned = {int(column): active[row] for row, column in zip(rows, columns)
                        if cost[row, column] < self.config.max_step_cents}

        next_active = []

        for peak in range(len(frequency)):
            index = assigned.get(peak)

            if index is None:
                index = len(tracks)
                tracks.append(PartialTrack())

            tracks[index].append(frame, frequency[peak], coefficient[peak])
            next_active.append(index)

        return next_active
