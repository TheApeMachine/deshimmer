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
