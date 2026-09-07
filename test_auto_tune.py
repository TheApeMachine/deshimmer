"""Regression tests for the optimizer's measurement and selection contracts."""

import tempfile
import unittest
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import optuna

import auto_tune
import master


class TestWeights(unittest.TestCase):
    def test_objectives(self):
        contributions = {
            "band_reduction": ("band_reduction_db", 0, 1),
            "swish_reduction": ("swish_reduction", 0, 1),
            "out_of_band_change": ("out_of_band_change_db", 1, -1),
            "musical_noise": ("musical_noise", 1, -1),
            "band_energy_loss": ("band_energy_loss_db", 1, -1),
            "centroid_shift": ("centroid_shift", 1, -1),
            "transient_leak": ("transient_leak", 2, -1),
            "harmonic_leak": ("harmonic_leak", 2, -1),
            "diff_onset_corr": ("diff_onset_corr", 2, -1),
            "stereo_width": ("stereo_width_delta", 3, -1),
            "loudness_loss": ("loudness_loss_db", 4, -1),
            "spectral_tilt": ("spectral_tilt_delta", 4, -1),
        }

        for weight, (metric, group, sign) in contributions.items():
            with self.subTest(weight=weight):
                metrics = {entry[0]: 0.0 for entry in contributions.values()}
                metrics[metric] = 3.0
                weights = replace(auto_tune.flat_weights(), **{weight: 2.0})
                expected = np.zeros(5)
                expected[group] = sign * 6.0
                np.testing.assert_array_equal(weights.objectives(metrics), expected)


class TestAutoTune(unittest.TestCase):
    def test_score_objectives(self):
        audio = np.random.default_rng(12).normal(0, 0.1, 8192).astype(np.float32)
        processed = audio * 0.7

        for aggressiveness in (0.0, 0.5, 1.0):
            with self.subTest(aggressiveness=aggressiveness):
                weights = auto_tune.weights_from_aggressiveness(aggressiveness)
                arguments = dict(band=(5100.0, 7200.0), swish_band=(3500.0, 14000.0), weights=weights)
                metrics = auto_tune.score_processed(audio, processed, 44100, **arguments)
                objectives = auto_tune.score_objectives(audio, processed, 44100, **arguments)
                self.assertAlmostEqual(sum(objectives), metrics["total"])

    def test_score_processed(self):
        audio = np.random.default_rng(3).normal(0, 0.1, 8192).astype(np.float32)
        weights = auto_tune.flat_weights()
        arguments = dict(band=(5100.0, 7200.0), swish_band=(3500.0, 14000.0), weights=weights)

        with self.subTest("unchanged input has no reward or damage"):
            metrics = auto_tune.score_processed(audio, audio, 44100, **arguments)
            self.assertAlmostEqual(metrics["total"], 0.0)

        with self.subTest("undersized regions are explicit errors"):
            with self.assertRaisesRegex(ValueError, "2048"):
                auto_tune.score_processed(audio[:512], audio[:512], 44100, **arguments)

        with self.subTest("phase damage above the shimmer band is measured"):
            samples = np.arange(44100) / 44100
            phase_jumps = np.repeat(np.random.default_rng(8).uniform(-np.pi, np.pi, 345), 128)[:44100]
            shimmer = 0.1 * np.sin(2 * np.pi * 6000 * samples)
            unstable = shimmer + 0.1 * np.sin(2 * np.pi * 10000 * samples + phase_jumps)
            stable = shimmer + 0.1 * np.sin(2 * np.pi * 10000 * samples)
            wide = auto_tune.score_processed(unstable, stable, 44100, **arguments)
            narrow = auto_tune.score_processed(
                unstable, stable, 44100, **{**arguments, "swish_band": (5100.0, 7200.0)},
            )
            self.assertEqual(wide["band_reduction_db"], narrow["band_reduction_db"])
            self.assertNotAlmostEqual(wide["swish_reduction"], narrow["swish_reduction"])
            self.assertAlmostEqual(
                wide["phase_instability_in"],
                master.measure_phase_instability(unstable, 44100, 3500.0, 14000.0),
            )

        for invalid in (audio[:-1], np.full_like(audio, np.nan)):
            with self.subTest("invalid scoring input"), self.assertRaises(ValueError):
                auto_tune.score_processed(audio, invalid, 44100, **arguments)

        with self.subTest("unmeasurable phase bands cannot silently score zero"):
            with self.assertRaisesRegex(ValueError, "swish scoring band"):
                auto_tune.score_processed(
                    audio, audio, 44100, **{**arguments, "swish_band": (30000.0, 31000.0)},
                )

        with self.subTest("a full-spectrum shimmer band has no out-of-band penalty"):
            metrics = auto_tune.score_processed(
                audio, audio, 44100, **{**arguments, "band": (0.0, 22050.0)},
            )
            self.assertEqual(metrics["out_of_band_change_db"], 0.0)

    def test_pick_pareto_trial(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(directions=["maximize"] * 5)
        # Both candidates are Pareto optimal. The second has greater total utility.
        for values in ((12.0, -10.0, -10.0, -10.0, -10.0),
                       (3.0, -1.0, 0.0, 0.0, 0.0)):
            study.add_trial(optuna.trial.create_trial(values=values))

        self.assertEqual(auto_tune._pick_pareto_trial(study).number, 1)
        self.assertEqual(auto_tune._pick_pareto_trial(study, mode="safe").number, 1)
        self.assertEqual(auto_tune._pick_pareto_trial(study, mode="aggressive").number, 0)

        for aggressiveness, expected_trial in ((0.0, 1), (1.0, 0)):
            with self.subTest(aggressiveness=aggressiveness):
                weighted_study = optuna.create_study(directions=["maximize"] * 5)
                weights = auto_tune.weights_from_aggressiveness(aggressiveness)

                for reduction, damage in ((8.0, 6.0), (2.0, 0.0)):
                    values = (weights.band_reduction * reduction,
                              -weights.out_of_band_change * damage, 0.0, 0.0, 0.0)
                    weighted_study.add_trial(optuna.trial.create_trial(values=values))

                self.assertEqual(auto_tune._pick_pareto_trial(weighted_study).number, expected_trial)

    def test_evaluate_params_multi(self):
        audio = np.random.default_rng(19).normal(0, 0.1, 44100).astype(np.float32)
        params = master.Params(n_fft=512, hop=128, swish_repair=0.3)
        regions = [auto_tune.Region(0.0, 0.5, "start"), auto_tune.Region(0.5, 0.5, "end")]
        weights = auto_tune.weights_from_aggressiveness(0.5)
        band = (params.start_hz, params.end_hz)
        result = auto_tune._evaluate_params_multi(audio, 44100, regions, params, weights, band, 0.5)
        processed, _ = auto_tune.process_audio(audio, 44100, params=params)
        expected = []

        for region in regions:
            begin = int(region.t0 * 44100)
            end = begin + int(region.dur * 44100)
            expected.append(auto_tune.score_objectives(
                audio[begin:end], processed[begin:end], 44100, band=band,
                swish_band=(params.swish_start_hz, params.swish_end_hz), weights=weights,
            ))

        np.testing.assert_allclose(result, np.mean(expected, axis=0))
        self.assertAlmostEqual(sum(result), auto_tune._evaluate_params_single(
            audio, 44100, regions, params, weights, band, 0.5,
        ))

        with tempfile.TemporaryDirectory() as directory:
            failures = auto_tune.EvaluationFailureLog(directory, planned_trials_per_stage=12)
            failures.begin_trial("swish")

            with self.assertRaises(optuna.TrialPruned):
                auto_tune._evaluate_params_multi(
                    audio * np.nan, 44100, regions, params, weights, band, 0.5,
                    failure_log=failures, stage="swish", trial_number=0,
                )

            events = json.loads(Path(failures.path).read_text())
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["stage"], "swish")

        with self.assertRaisesRegex(ValueError, "scorable region"):
            auto_tune._evaluate_params_multi(audio, 44100, [], params, weights, band, 0.5)

    def test_refine(self):
        audio = np.random.default_rng(21).normal(0, 0.1, 22050).astype(np.float32)
        params = master.Params(n_fft=512, hop=128, mix=0.0, swish_repair=0.2, deres=0.1)

        with tempfile.TemporaryDirectory() as directory:
            selected, summary = auto_tune.refine(
                audio, 44100, base_params=params,
                regions=[auto_tune.Region(0.0, 0.5, "whole")],
                aggressiveness=0.5, n_trials_per_stage=2, refine_dur=0.5,
                debug_dir=directory,
            )

        self.assertEqual(selected.mix, 1.0)
        self.assertEqual(len(summary["stages"]), 4)

        for stage in summary["stages"]:
            with self.subTest(stage=stage["stage"]):
                self.assertGreaterEqual(stage["selected_score"], stage["baseline_score"])
                self.assertAlmostEqual(stage["selected_score"], sum(stage["pareto"]["balanced"]["values"]))

        self.assertAlmostEqual(summary["final_score"], summary["stages"][-1]["selected_score"])
