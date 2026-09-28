"""Exercise the real preview, residual selection, and WAV export paths."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import ui_gradio
from deshimmer_api import process_audio
from tonal_fixtures import TonalFixture


class TestUIGradio(unittest.TestCase):
    def test_run_once(self):
        source = np.random.default_rng(42).normal(0, 0.1, 22050).astype(np.float32)
        audio = np.column_stack((source, np.pad(source[:-11], (11, 0))))
        state = ui_gradio._default_param_state()
        state.update(n_fft=512, hop=128, spec_n_fft=512, spec_hop=128)
        params, delivery, _ = ui_gradio._build_params(state)
        expected, info = process_audio(audio, 44100, params=params, master_params=delivery, include_residuals=True)

        for full_song in (False, True):
            for repair_only in (False, True):
                with self.subTest(full_song=full_song, repair_only=repair_only):
                    result = ui_gradio.run_once((44100, audio), 0.0, 0.5, 0.0, full_song, state, repair_only)
                    np.testing.assert_array_equal(result[0][1], audio)
                    np.testing.assert_array_equal(result[1][1], expected)
                    key = "artifact_repair_diff" if repair_only else "total_enhancement_diff"
                    np.testing.assert_array_equal(result[2][1], info["residuals"][key])
                    self.assertEqual(result[3].shape[2], 3)

        with self.subTest("bypass produces an exact null even after HF resynthesis was enabled"):
            state.update(mix=0.0, hf_resynth=True)
            result = ui_gradio.run_once((44100, audio), 0.0, 0.5, 0.0, True, state)
            np.testing.assert_array_equal(result[1][1], audio)
            np.testing.assert_array_equal(result[2][1], np.zeros_like(audio))

    def test_render_full_to_files(self):
        audio = np.random.default_rng(4).normal(0, 0.1, (4096, 2)).astype(np.float32)
        state = ui_gradio._default_param_state()
        state.update(n_fft=512, hop=128, master_enabled=True, delta_listen=True)
        params, delivery, _ = ui_gradio._build_params(state)
        expected, info = process_audio(
            audio, 44100, params=replace(params, delta_listen=False),
            master_params=delivery, include_residuals=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ui_gradio, "__file__", str(Path(directory) / "ui_gradio.py")):
                output_path, diff_path, params_path = ui_gradio.render_full_to_files((44100, audio), state, True)

            output, rate = sf.read(output_path, dtype="float32", always_2d=True)
            diff, _ = sf.read(diff_path, dtype="float32", always_2d=True)
            # PCM_24 export has one quantization step of 2**-23.
            np.testing.assert_allclose(output, info["residuals"]["artifact_repair_diff"], atol=2**-23)
            np.testing.assert_allclose(diff, expected, atol=2**-23)
            self.assertEqual(rate, 44100)
            self.assertTrue(json.loads(Path(params_path).read_text())["repair_only_diff"])

    def test_build_ui(self):
        app = ui_gradio.build_ui()
        controls = app.config["components"]
        residual = [item for item in controls if item["props"].get("label") == "Listen to repair-only diff"]
        self.assertEqual(len(residual), 1)
        self.assertFalse(residual[0]["props"]["value"])

        with self.subTest("normal analyze, optimize, render and download workflow"):
            callbacks = {entry.fn.__name__: entry.fn for entry in app.fns.values() if entry.fn is not None}
            fixture = TonalFixture()
            mono = fixture.render("wandering")
            audio = np.column_stack((mono, 0.7 * mono))
            state = ui_gradio._default_param_state()
            self.assertGreater(state["tonal_repair"], 0)
            # Even a previously saved off setting is activated by Set from audio.
            state["tonal_repair"] = 0.0
            analyzed = callbacks["do_analyze"]((fixture.sample_rate, audio), 0.0, 2.0, True, 1, state)
            self.assertGreater(analyzed[0]["tonal_repair"], 0)
            optimized = callbacks["do_refine"](
                (fixture.sample_rate, audio), 0.0, 2.0, True, 1, 0.5, 4, 2.0, analyzed[0],
            )
            state = optimized[0]
            self.assertIn("tonal", optimized[-1])
            self.assertIn("six NSGA-II objectives", optimized[-1])
            self.assertGreater(state["tonal_repair"], 0)
            params, delivery, _ = ui_gradio._build_params(state)
            expected, info = process_audio(audio, fixture.sample_rate, params=params, master_params=delivery)
            self.assertTrue(info["tonal"]["enabled"])
            self.assertGreater(info["tonal"]["changed_partials"], 0)
            self.assertGreater(info["tonal"]["coherence_gain"], 0)
            preview = ui_gradio.run_once((fixture.sample_rate, audio), 0.0, 2.0, 0.0, True, state)
            np.testing.assert_array_equal(preview[1][1], expected)
            np.testing.assert_array_equal(preview[2][1], audio - expected)

            with tempfile.TemporaryDirectory() as directory:
                with patch.object(ui_gradio, "__file__", str(Path(directory) / "ui_gradio.py")):
                    output_path, diff_path, params_path = ui_gradio.render_full_to_files(
                        (fixture.sample_rate, audio), state,
                    )

                output, rate = sf.read(output_path, dtype="float32", always_2d=True)
                diff, diff_rate = sf.read(diff_path, dtype="float32", always_2d=True)
                saved = json.loads(Path(params_path).read_text())
                self.assertEqual((rate, diff_rate), (fixture.sample_rate, fixture.sample_rate))
                self.assertEqual(saved["params"]["tonal_repair"], state["tonal_repair"])
                self.assertFalse(saved["repair_only_diff"])
                np.testing.assert_allclose(output, expected, atol=2**-23)
                np.testing.assert_allclose(output + diff, audio, atol=2 * 2**-23)
