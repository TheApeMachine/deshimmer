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

    def test_dsp_pipeline_modular_reordering(self):
        """Test that stages can be isolated and reordered in a custom DSPPipeline."""
        sr = 44100
        audio = np.random.default_rng(42).normal(0, 0.1, (2048, 2)).astype(np.float32)
        p = master.Params(n_fft=512, hop=128, expander=True, swish_repair=0.3)

        default_pipe = master.build_default_pipeline(p)
        self.assertEqual(len(default_pipe.stages), 10)
        self.assertEqual(default_pipe.stages[0].name, "expander")

        # Custom reordered pipeline: swish before expander
        custom_pipe = master.DSPPipeline([
            master.SwishRepairStage.from_params(p),
            master.ExpanderStage.from_params(p),
            master.GainApplicationStage.from_params(p),
        ])
        self.assertEqual(len(custom_pipe.stages), 3)
        self.assertEqual(custom_pipe.stages[0].name, "swish_repair")
        self.assertEqual(custom_pipe.stages[1].name, "expander")

        out = master.process_stft(audio, sr, p, pipeline=custom_pipe)
        self.assertEqual(out.shape, audio.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_deresonator_isolated_unit_test(self):
        """Test that DeResonatorStage can be spun up and unit-tested in isolation without the full STFT loop."""
        sr = 44100
        n_fft = 512
        hop = 128
        n_bins = n_fft // 2 + 1
        n_frames = 20

        # Create a synthetic spectrum with a strong narrowband resonance around bin 50
        freqs = np.linspace(0.0, sr * 0.5, n_bins).astype(np.float32)
        Z = np.ones((n_bins, n_frames, 1), dtype=np.complex64) * 0.01
        res_bin = 50
        Z[res_bin, :, :] = 1.0  # Strong persistent resonance peak

        p = master.Params(
            n_fft=n_fft, hop=hop, deres=1.0, deq_start_hz=100.0, deq_end_hz=10000.0,
            deq_thr_db=3.0, deq_slope=1.0, deq_persist_ms=1.0, deq_persist_thr_db=0.1,
            deq_freq_smooth_bins=1, magnitude_inpaint=False,
        )

        ctx = master.DSPContext.create(Z, sr=sr, hop=hop, n_fft=n_fft, freqs=freqs, p=p)
        stage = master.DeResonatorStage.from_params(p)
        self.assertTrue(stage.is_enabled())

        stage.process(ctx)

        # Gain at the resonance bin should be significantly attenuated (< 0.5)
        # while surrounding non-resonant bins should remain close to 1.0
        self.assertLess(float(ctx.g_total[res_bin, 10]), 0.6)
        self.assertGreater(float(ctx.g_total[res_bin - 15, 10]), 0.9)

    def test_temporal_psychoacoustic_masking(self):
        """Test that forward temporal masking holds mask high for broadband decays and collapses for tonal ringing."""
        sr = 44100
        hop = 128
        F = 128
        T = 20

        # Attack burst at frame 5, then drops to low energy
        mask_psd = np.ones((F, T), dtype=np.float32) * 1e-4
        mask_psd[:, 5] = 1.0

        # 1. Broadband condition (flatness = 0.8)
        flatness_broadband = np.full(T, 0.8, dtype=np.float32)
        decayed_broadband = master.apply_temporal_masking_decay(
            mask_psd, sr=sr, hop=hop, decay_ms=15.0,
            flatness=flatness_broadband, flat_start=0.25, flat_end=0.70, gated=True,
        )
        # Mask at frame 6 and 7 should be sustained well above baseline 1e-4
        self.assertGreater(float(decayed_broadband[50, 6]), 0.5)
        self.assertGreater(float(decayed_broadband[50, 7]), 0.2)

        # 2. Narrow tonal ringing condition (flatness = 0.1, below flat_start)
        flatness_tonal = np.full(T, 0.1, dtype=np.float32)
        decayed_tonal = master.apply_temporal_masking_decay(
            mask_psd, sr=sr, hop=hop, decay_ms=15.0,
            flatness=flatness_tonal, flat_start=0.25, flat_end=0.70, gated=True,
        )
        # Because it is tonal, gate is 0; forward decay is gated off, so frame 6 collapses back to 1e-4
        self.assertAlmostEqual(float(decayed_tonal[50, 6]), 1e-4, delta=1e-5)

