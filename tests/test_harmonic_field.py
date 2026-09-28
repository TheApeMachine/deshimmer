import unittest

import numpy as np

from tonal_fixtures import TonalFixture
from tonal_repair import TonalRepair


class TestHarmonicField(unittest.TestCase):
    def test_fit(self):
        fixture = TonalFixture()
        analyzer = TonalRepair()

        for mode in ("coherent", "stretched", "vibrato"):
            with self.subTest(mode=mode):
                field = analyzer.analyze(fixture.render(mode), fixture.sample_rate)
                self.assertEqual(len(field.families), 1)
                self.assertLess(field.metrics()["p95_disagreement_cents"], 0.1)

        # Four harmonics remain when the fundamental is missing.
        fundamental = 0.15 * np.cos(2 * np.pi * 221 * (fixture.time + 1 / fixture.sample_rate))
        missing = fixture.render() - fundamental
        field = analyzer.analyze(missing, fixture.sample_rate)
        self.assertEqual(len(field.families), 1)
        self.assertEqual(len(field.families[0].members), 4)

    def test_compare(self):
        fixture = TonalFixture()
        processor = TonalRepair()
        original = fixture.render("wandering")
        field = processor.analyze(original, fixture.sample_rate)
        repaired, _ = processor.process(original, fixture.sample_rate)
        self.assertEqual(field.compare(field), 0.0)
        self.assertGreater(field.compare(processor.analyze(repaired, fixture.sample_rate)), 0.0)

        for output in (np.zeros_like(original), original * 0.5):
            with self.subTest("attenuation cannot masquerade as coherence"):
                self.assertLess(field.compare(processor.analyze(output, fixture.sample_rate)), 0.0)

        # Identical notes appearing at different times must not cross-match.
        repeated = np.concatenate((original, np.zeros(4410), original))
        repeated_field = processor.analyze(repeated, fixture.sample_rate)
        self.assertAlmostEqual(repeated_field.compare(repeated_field), 0.0)
