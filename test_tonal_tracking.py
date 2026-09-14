import unittest
from dataclasses import replace

import numpy as np

from tonal_fixtures import TonalFixture
from tonal_tracking import PartialTracker, TonalConfig


class TestPartialTracker(unittest.TestCase):
    def test_analyze(self):
        fixture = TonalFixture()
        tracker = PartialTracker(TonalConfig())
        tracks, centers = tracker.analyze(fixture.render(stereo=True), fixture.sample_rate)
        self.assertEqual(len(tracks), 5)

        for order, track in enumerate(tracks, 1):
            with self.subTest(harmonic=order):
                np.testing.assert_allclose(track.frequencies, order * 221.0, atol=0.02)
                self.assertEqual(track.frames, list(range(len(centers))))
                self.assertLess(track.uncertainty_cents(fixture.sample_rate, tracker.config.hop), 0.1)
                coefficients = np.asarray(track.coefficients)
                np.testing.assert_allclose(coefficients[:, 1], -0.7 * coefficients[:, 0], atol=1e-6)

        for duration in (0.0, 0.02):
            with self.subTest(duration=duration):
                tracks, _ = tracker.analyze(TonalFixture(duration=duration).render(), fixture.sample_rate)
                self.assertEqual(tracks, [])

        with self.assertRaisesRegex(ValueError, "finite"):
            tracker.analyze(np.full(8192, np.nan), fixture.sample_rate)

        with self.assertRaisesRegex(ValueError, "FFT"):
            replace(tracker.config, n_fft=4095)
