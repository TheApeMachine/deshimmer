"""Self-referenced partial correction by continuous-phase sinusoidal replacement.

The unmodeled residual is retained: output = input - modeled + corrected.
Short events, unresolved peaks and unsupported families never enter the model.
This is not source separation, note quantization, or a claim to identify intent.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import find_objects, label

from harmonic_field import HarmonicFamily, HarmonicField
from tonal_tracking import PartialTracker, TonalConfig


class TonalRepair:
    """Analyze harmonic evidence and repair only deviations supported by peers."""

    def __init__(self, config: TonalConfig | None = None):
        self.config = config if config is not None else TonalConfig()
        self.tracker = PartialTracker(self.config)

    def analyze(self, audio: np.ndarray, sample_rate: int) -> HarmonicField:
        tracks, _ = self.tracker.analyze(audio, sample_rate)
        return HarmonicField(tracks, self.config, sample_rate).fit()

    def process(self, audio: np.ndarray, sample_rate: int, *, amount: float = 0.5,
                max_shift_cents: float = 25.0) -> tuple[np.ndarray, dict]:
        if not np.isfinite(amount) or not 0 <= amount <= 1:
            raise ValueError("tonal repair amount must be finite and between 0 and 1")

        if not np.isfinite(max_shift_cents) or max_shift_cents <= 0:
            raise ValueError("tonal maximum shift must be finite and positive")

        original = np.asarray(audio)

        if original.dtype.kind != "f":
            raise ValueError("tonal repair requires real floating-point audio")

        field = self.analyze(original, sample_rate)
        output = original.astype(np.float64, copy=True)

        if output.ndim == 1:
            output = output[:, None]

        changed = 0

        for family in field.families:
            changed += self.reconstruct(output, field, family, amount, max_shift_cents)

        output = output.reshape(original.shape).astype(original.dtype)
        after = self.analyze(output, sample_rate) if changed else field
        return output, {
            "enabled": True,
            "stage": "before_wet_dry_mix_and_delivery",
            "before": field.metrics(),
            "after": after.metrics(),
            "changed_partials": changed,
            "coherence_gain": field.compare(after),
            "amount": float(amount),
            "max_shift_cents": float(max_shift_cents),
        }

    def reconstruct(self, output: np.ndarray, field: HarmonicField, family: HarmonicFamily,
                    amount: float, max_shift_cents: float) -> int:
        changed = 0
        centers = (family.first_frame + np.arange(family.observed.shape[1])) * self.config.hop + self.config.n_fft // 2
        seconds = centers / field.sample_rate

        for row, member in enumerate(family.members):
            if member in field.ambiguous_members:
                continue

            error = family.disagreement[row]
            excess = np.sign(error) * np.maximum(np.abs(error) - family.tolerance[row], 0)
            correction = amount * np.clip(excess, -max_shift_cents, max_shift_cents)

            if not np.any(correction):
                continue

            track = field.tracks[member]
            start = family.first_frame - track.frames[0]
            coefficients = np.asarray(track.coefficients)[start:start + len(centers)]
            frequency = family.observed[row]
            delta_frequency = frequency * np.expm1(-correction * np.log(2) / 1200)
            phase_delta = cumulative_trapezoid(2 * np.pi * delta_frequency, seconds, initial=0)
            delta = CubicHermiteSpline(seconds, phase_delta, 2 * np.pi * delta_frequency)
            applied = False

            for channel in range(output.shape[1]):
                supported = np.abs(coefficients[:, channel]) > 0

                for (selection,) in find_objects(label(supported)[0]):
                    if selection.stop - selection.start < self.config.min_track_frames:
                        continue

                    self.replace_partial(output[:, channel], field.sample_rate, centers[selection],
                                         coefficients[selection, channel], frequency[selection], delta)
                    applied = True

            changed += int(applied)

        return changed

    def replace_partial(self, output, sample_rate, centers, coefficients, frequency, delta):
        """Replace a sinusoid only inside one channel's continuously supported span."""
        seconds = centers / sample_rate
        sample_indices = np.arange(centers[0], centers[-1] + 1)
        sample_times = sample_indices / sample_rate
        edge = np.minimum(sample_indices - sample_indices[0], sample_indices[-1] - sample_indices)
        fade = np.sin(np.minimum(edge / (self.config.n_fft / 2), 1) * np.pi / 2) ** 2
        expected_advance = 2 * np.pi * (frequency[1:] + frequency[:-1]) / 2 * np.diff(seconds)
        observed_phase = np.angle(coefficients)
        residual = np.angle(np.exp(1j * (np.diff(observed_phase) - expected_advance)))
        unwrapped = np.r_[observed_phase[0], observed_phase[0] + np.cumsum(expected_advance + residual)]
        phase = CubicHermiteSpline(seconds, unwrapped, 2 * np.pi * frequency)(sample_times)
        amplitude = np.interp(sample_times, seconds, np.abs(coefficients))
        output[sample_indices] += fade * amplitude * (np.cos(phase + delta(sample_times)) - np.cos(phase))
