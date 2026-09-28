"""Full-reference fixtures for self-consistency and preservation."""

import unittest

import numpy as np

from tonal_fixtures import TonalFixture
from tonal_repair import TonalRepair


class TestTonalRepair(unittest.TestCase):
    def test_reconstruct(self):
        fixture = TonalFixture()
        processor = TonalRepair()
        original = fixture.render("wandering", stereo=True)
        field = processor.analyze(original, fixture.sample_rate)
        # Family overlap may truncate otherwise long per-channel support. No
        # channel with fewer than the required observations can authorize repair.
        for track in field.tracks:
            coefficients = np.zeros_like(track.coefficients)
            coefficients[:processor.config.min_track_frames - 1] = np.asarray(track.coefficients)[:processor.config.min_track_frames - 1]
            track.coefficients = list(coefficients)

        output = original.copy()
        changed = processor.reconstruct(output, field, field.families[0], 0.5, 25.0)
        self.assertEqual(changed, 0)
        np.testing.assert_array_equal(output, original)

    def test_analyze(self):
        fixture = TonalFixture()
        analyzer = TonalRepair()
        field = analyzer.analyze(fixture.render(), fixture.sample_rate)
        self.assertEqual(len(field.families), 1)
        self.assertEqual(len(field.tracks), 5)
        self.assertLess(field.metrics()["p95_disagreement_cents"], 0.1)
        self.assertAlmostEqual(np.median(field.families[0].expected[0]), 221.0, places=2)

    def test_process(self):
        fixture = TonalFixture()
        processor = TonalRepair()

        for mode in ("coherent", "vibrato", "stretched", "chorus", "polyphonic", "noise", "transient"):
            with self.subTest(mode=mode):
                original = fixture.render(mode)
                output, report = processor.process(original, fixture.sample_rate)
                np.testing.assert_array_equal(output, original)
                self.assertEqual(report["changed_partials"], 0)

        for stereo in (False, True):
            with self.subTest(stereo=stereo, mode="wandering"):
                original = fixture.render("wandering", stereo=stereo)
                output, report = processor.process(original, fixture.sample_rate)
                self.assertEqual(report["changed_partials"], 1)
                self.assertLess(report["after"]["p95_disagreement_cents"], report["before"]["p95_disagreement_cents"])
                self.assertGreater(report["coherence_gain"], 0)
                self.assertTrue(np.all(np.isfinite(output)))
                np.testing.assert_array_equal(output[:processor.config.n_fft // 2], original[:processor.config.n_fft // 2])
                self.assertLess(abs(np.mean(output ** 2) / np.mean(original ** 2) - 1), 0.01)

                if stereo:
                    np.testing.assert_allclose(output[:, 1], -0.7 * output[:, 0], atol=1e-7)

        original = fixture.render("wandering")
        noisy_channel = np.random.default_rng(99).normal(0, 0.01, len(original)).astype(np.float32)
        stereo_input = np.column_stack((original, noisy_channel))
        stereo_output, report = processor.process(stereo_input, fixture.sample_rate)
        np.testing.assert_array_equal(stereo_output[:, 1], noisy_channel)
        self.assertGreater(report["changed_partials"], 0)

        with self.subTest("channel confidence gaps are not interpolated into noise"):
            stereo_input = np.column_stack((original, original.copy()))
            gap = slice(len(original) // 3, 2 * len(original) // 3)
            stereo_input[gap, 1] = noisy_channel[gap]
            stereo_output, report = processor.process(stereo_input, fixture.sample_rate)
            np.testing.assert_array_equal(stereo_output[gap, 1], stereo_input[gap, 1])
            self.assertGreater(report["changed_partials"], 0)

        output, report = processor.process(original, fixture.sample_rate, amount=0.0)
        np.testing.assert_array_equal(output, original)
        self.assertEqual(report["changed_partials"], 0)

        with self.assertRaisesRegex(ValueError, "amount"):
            processor.process(original, fixture.sample_rate, amount=np.nan)
