"""Known sinusoidal scenes for regression tests and full-reference benchmarks."""

import numpy as np


class TonalFixture:
    """Five-partial A=442 family with independently selectable perturbations."""

    def __init__(self, sample_rate=44100, duration=2.0):
        self.sample_rate = sample_rate
        self.time = np.arange(round(sample_rate * duration)) / sample_rate

    def render(self, mode="coherent", *, fundamental=221.0, stereo=False, vibrato_rate=2.0):
        audio = np.zeros(len(self.time))

        for harmonic in range(1, 6):
            frequency = np.full(len(self.time), fundamental * harmonic)

            if mode == "wandering" and harmonic == 4:
                frequency *= np.exp2(20 * np.sin(2 * np.pi * 2 * self.time) / 1200)

            if mode == "vibrato":
                frequency *= np.exp2(20 * np.sin(2 * np.pi * vibrato_rate * self.time) / 1200)

            if mode == "stretched":
                frequency *= np.sqrt(1 + 0.0002 * harmonic ** 2)

            audio += 0.15 / harmonic * np.cos(2 * np.pi * np.cumsum(frequency) / self.sample_rate)

        if mode == "chorus":
            audio = self.render(fundamental=fundamental * np.exp2(8 / 1200))
            audio += self.render(fundamental=fundamental * np.exp2(-8 / 1200))
            audio /= 2

        if mode == "polyphonic":
            audio = self.render("vibrato") + self.render("vibrato", fundamental=277.2, vibrato_rate=3.3)

        if mode == "noise":
            audio = np.random.default_rng(64).normal(0, 0.1, len(self.time))

        if mode == "transient":
            audio[:] = 0
            audio[len(audio) // 2] = 0.5

        if stereo:
            audio = np.column_stack((audio, -0.7 * audio))

        return audio.astype(np.float32)
