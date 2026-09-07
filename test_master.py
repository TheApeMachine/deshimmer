"""Regression tests for channel timing and dry/removed signal contracts."""

import unittest
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

import master


class TestMaster(unittest.TestCase):
    def test_align_lr_delay(self):
        source = np.random.default_rng(7).normal(0, 0.1, 4096).astype(np.float32)
        delay = 11  # Known injected inter-channel delay, within the 3 ms limit.
        delayed = np.pad(source[:-delay], (delay, 0))

        for leading_channel in (0, 1):
            with self.subTest(leading_channel=leading_channel):
                audio = np.column_stack((source, delayed))

                if leading_channel == 1:
                    audio = audio[:, ::-1].copy()

                original = audio.copy()
                aligned = master.align_lr_delay(audio, 44100)
                np.testing.assert_array_equal(aligned[:, 0], aligned[:, 1])
                np.testing.assert_array_equal(audio, original)

        for audio in (source, np.column_stack((source, source)),
                      np.column_stack((source, np.pad(source[:-500], (500, 0))))):
            with self.subTest(shape=audio.shape):
                np.testing.assert_array_equal(master.align_lr_delay(audio, 44100), audio)

    def test_process_stft(self):
        source = np.random.default_rng(9).normal(0, 0.1, 4096).astype(np.float32)
        stereo = np.column_stack((source, np.pad(source[:-9], (9, 0))))

        for audio in (source, stereo):
            original = audio.copy()
            params = master.Params(n_fft=512, hop=128)
            wet = master.process_stft(audio, 44100, params)

            for mix in (0.0, 0.25, 1.0):
                with self.subTest(channels=audio.ndim, mix=mix):
                    output = master.process_stft(audio, 44100, replace(params, mix=mix))
                    expected = mix * wet + (1.0 - mix) * original
                    np.testing.assert_allclose(output, expected, atol=1e-7)
                    removed = master.process_stft(
                        audio, 44100, replace(params, mix=mix, delta_listen=True),
                    )
                    np.testing.assert_allclose(removed, original - output, atol=1e-7)

                    if mix == 0.0:
                        np.testing.assert_array_equal(output, original)
                        np.testing.assert_array_equal(removed, np.zeros_like(original))

            np.testing.assert_array_equal(audio, original)

    def test_main(self):
        audio = np.random.default_rng(54).normal(0, 0.1, (4096, 2)).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            destination = Path(directory) / "output.wav"
            sf.write(source, audio, 44100, subtype="FLOAT")
            command = [sys.executable, str(Path(master.__file__)), str(source), str(destination),
                       "--n-fft", "512", "--hop", "128", "--subtype", "FLOAT"]
            outputs = {}
            scenarios = {
                "bypass": ["--mix", "0", "--hf-resynth"],
                "repair_only": ["--repair-only"],
                "mastered": ["--master", "--hf-resynth"],
                "delta": ["--master", "--hf-resynth", "--delta-listen"],
            }

            for name, flags in scenarios.items():
                with self.subTest(mode=name):
                    result = subprocess.run(command + flags, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    outputs[name], _ = sf.read(destination, dtype="float32", always_2d=True)

            np.testing.assert_array_equal(outputs["bypass"], audio)
            np.testing.assert_array_equal(outputs["delta"], audio - outputs["mastered"])
            repair_only = master.process_stft(audio, 44100, master.Params(n_fft=512, hop=128, enhance=False))
            np.testing.assert_allclose(outputs["repair_only"], repair_only, atol=1e-7)
