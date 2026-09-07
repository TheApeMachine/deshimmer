"""Full-pipeline bypass, mastering, and forensic residual regression tests."""

import unittest
from dataclasses import replace

import numpy as np

import master
from deshimmer_api import process_audio


class TestDeshimmerAPI(unittest.TestCase):
    def test_process_audio(self):
        source = np.random.default_rng(31).normal(0, 0.1, 4096).astype(np.float32)
        audio = np.column_stack((source, np.pad(source[:-7], (7, 0))))
        params = master.Params(n_fft=512, hop=128, hf_resynth=True, swish_repair=0.2)

        with self.subTest("HF resynthesis obeys bypass and fractional wet/dry"):
            wet, _ = process_audio(audio, 44100, params=params)

            for mix in (0.0, 0.25, 1.0):
                mixed, _ = process_audio(audio, 44100, params=replace(params, mix=mix))
                np.testing.assert_allclose(mixed, mix * wet + (1 - mix) * audio, atol=1e-7)

                if mix == 0.0:
                    np.testing.assert_array_equal(mixed, audio)

        for mastering in (False, True):
            with self.subTest(mastering=mastering):
                delivery = master.MasterParams(enabled=mastering)
                output, info = process_audio(
                    audio, 44100, params=params, master_params=delivery, include_residuals=True,
                )
                delta, delta_info = process_audio(
                    audio, 44100, params=replace(params, delta_listen=True), master_params=delivery,
                )
                np.testing.assert_array_equal(delta, audio - output)
                self.assertEqual(info["measure_out"], delta_info["measure_out"])
                self.assertNotIn("residuals", delta_info)
                repair, _ = process_audio(audio, 44100, params=replace(params, enhance=False))
                np.testing.assert_array_equal(info["residuals"]["artifact_repair_diff"], audio - repair)
                np.testing.assert_array_equal(info["residuals"]["total_enhancement_diff"], audio - output)
                self.assertFalse(np.array_equal(repair, output))

        with self.subTest("bypass clears artifact confidence and both residuals"):
            master._LAST_ARTIFACT_CONF_FRAMES = np.ones((1, 4), dtype=np.float32)
            bypass, info = process_audio(audio, 44100, params=replace(params, mix=0.0), include_residuals=True)
            self.assertIsNone(master._LAST_ARTIFACT_CONF_FRAMES)
            np.testing.assert_array_equal(bypass, audio)

            for residual in info["residuals"].values():
                np.testing.assert_array_equal(residual, np.zeros_like(audio))

        with self.subTest("debug collection does not change output"):
            output, _ = process_audio(audio, 44100, params=params)
            debug, info = process_audio(audio, 44100, params=params, debug_params=master.DebugParams(enabled=True))
            np.testing.assert_array_equal(debug, output)
            self.assertIn("flux_db", info["dbg_data_keys"])
