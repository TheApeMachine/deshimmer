"""Regression tests for Sound Goodizer: Tonal Balance, Punch, Low-End Coherence, and Stereo Translation."""

import unittest
from dataclasses import replace

import numpy as np

from deshimmer.api import process_audio
from deshimmer.autotune import Region, analyze, refine
from deshimmer.core import Params, process_stft
from deshimmer.core.stages import (
    HarmonicEnhanceStage,
    LowEndStage,
    SmartDenoiseStage,
    StereoTranslationStage,
    TonalBalanceStage,
    TransientPunchStage,
)
from deshimmer.tonal import TonalRepair


class TestSoundGoodizer(unittest.TestCase):
    def test_denoise_attack_release_consumed(self):
        """Verify that dn_attack_ms and dn_release_ms have a measurable effect on denoise gain."""
        sr = 44100
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, (4096, 1)).astype(np.float32)

        # Compare fast attack vs slow attack
        p_fast = Params(denoise=0.7, dn_attack_ms=2.0, dn_release_ms=60.0)
        p_slow = Params(denoise=0.7, dn_attack_ms=40.0, dn_release_ms=300.0)

        out_fast = process_stft(noise, sr, p_fast)
        out_slow = process_stft(noise, sr, p_slow)

        self.assertFalse(np.array_equal(out_fast, out_slow))
        self.assertTrue(np.all(np.isfinite(out_fast)))
        self.assertTrue(np.all(np.isfinite(out_slow)))

    def test_tonal_repair_amount_zero_fast_bypass(self):
        """Verify that TonalRepair with amount=0 returns immediately without harmonic fitting."""
        sr = 44100
        audio = np.random.default_rng(123).normal(0, 0.1, 4096).astype(np.float32)
        processor = TonalRepair()

        out, report = processor.process(audio, sr, amount=0.0)
        np.testing.assert_array_equal(out, audio)
        self.assertEqual(report["changed_partials"], 0)
        self.assertFalse(report["enabled"])

    def test_tonal_balance_stage(self):
        """Verify that TonalBalanceStage adjusts spectral bands and respects bypass."""
        sr = 44100
        rng = np.random.default_rng(77)
        audio = rng.normal(0, 0.1, (4096, 2)).astype(np.float32)

        # Neutral / disabled
        p_neutral = Params(tonal_balance=0.0)
        stage_neutral = TonalBalanceStage.from_params(p_neutral)
        self.assertFalse(stage_neutral.is_enabled())
        out_neutral = process_stft(audio, sr, p_neutral)

        # Active de-mudding and air boost
        p_active = Params(tonal_balance=0.8, tb_low_mid_db=-2.0, tb_air_db=2.5)
        out_active = process_stft(audio, sr, p_active)
        self.assertFalse(np.array_equal(out_active, out_neutral))
        self.assertTrue(np.all(np.isfinite(out_active)))

        # Bypassed when enhance=False
        p_no_enhance = replace(p_active, enhance=False)
        stage = TonalBalanceStage.from_params(p_no_enhance)
        self.assertFalse(stage.is_enabled())

    def test_transient_punch_stage(self):
        """Verify that TransientPunchStage expands transient attacks."""
        sr = 44100
        # Create audio with an abrupt click / impulse at index 1000
        audio = np.zeros((4096, 2), dtype=np.float32)
        audio[1000, :] = 0.9

        p_off = Params(transient_punch=0.0)
        stage_off = TransientPunchStage.from_params(p_off)
        self.assertFalse(stage_off.is_enabled())
        out_off = process_stft(audio, sr, p_off)

        p_punch = Params(transient_punch=0.8, punch_attack_boost_db=3.0)
        out_punch = process_stft(audio, sr, p_punch)

        self.assertTrue(np.all(np.isfinite(out_punch)))
        self.assertFalse(np.array_equal(out_punch, out_off))

    def test_low_end_stage(self):
        """Verify LowEndStage mono-centers low-end frequencies in stereo audio."""
        sr = 44100
        t = np.arange(4096) / sr
        # 60 Hz tone out of phase in L and R
        sub_l = 0.3 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
        sub_r = -0.3 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
        # High 5000 Hz tone stereo
        hi_l = 0.2 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)
        hi_r = 0.2 * np.cos(2 * np.pi * 5000 * t).astype(np.float32)

        stereo = np.column_stack((sub_l + hi_l, sub_r + hi_r))

        p_mono = Params(bass_mono=True, bass_mono_hz=120.0)
        out_mono = process_stft(stereo, sr, p_mono)

        # Below 120 Hz, out-of-phase bass should be collapsed/attenuated on the sides
        side_in = 0.5 * (stereo[:, 0] - stereo[:, 1])
        side_out = 0.5 * (out_mono[:, 0] - out_mono[:, 1])
        self.assertLess(np.std(side_out), np.std(side_in))
        self.assertTrue(np.all(np.isfinite(out_mono)))

    def test_stereo_translation_stage(self):
        """Verify StereoTranslationStage safely adjusts width and preserves mono compatibility."""
        sr = 44100
        rng = np.random.default_rng(88)
        stereo = rng.normal(0, 0.1, (4096, 2)).astype(np.float32)

        p_wide = Params(stereo_width=1.2, stereo_safe_mono=True)
        out_wide = process_stft(stereo, sr, p_wide)
        self.assertFalse(np.array_equal(out_wide, stereo))
        self.assertTrue(np.all(np.isfinite(out_wide)))

    def test_harmonic_enhance_stage(self):
        """Verify HarmonicEnhanceStage adds subtle high-frequency harmonic sheen."""
        sr = 44100
        t = np.arange(4096) / sr
        # 1000 Hz pure tone
        tone = 0.4 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)[:, None]

        p_harm = Params(harmonic_enhance=0.35, harmonic_start_hz=2000.0)
        out_harm = process_stft(tone, sr, p_harm)
        self.assertFalse(np.array_equal(out_harm, tone))
        self.assertTrue(np.all(np.isfinite(out_harm)))

    def test_null_invertibility_with_all_stages(self):
        """Verify exact forensic residual reconstruction contract (x = y + diff) with all sound goodizer stages active."""
        sr = 44100
        rng = np.random.default_rng(999)
        audio = rng.normal(0, 0.1, (4096, 2)).astype(np.float32)

        params = Params(
            tonal_balance=0.5,
            tb_low_mid_db=-1.5,
            tb_air_db=1.5,
            transient_punch=0.4,
            punch_attack_boost_db=2.5,
            bass_mono=True,
            bass_tighten=0.2,
            stereo_width=1.1,
            harmonic_enhance=0.2,
        )

        output, info = process_audio(audio, sr, params=params, include_residuals=True)
        diff = info["residuals"]["total_enhancement_diff"]

        # Null test: output + diff == audio (within floating point precision)
        np.testing.assert_allclose(output + diff, audio, atol=1e-6)

    def test_auto_analyze_diagnostic_layer(self):
        """Verify analyze() extracts perceptual diagnostics for tonal balance, dynamics, and stereo."""
        sr = 44100
        rng = np.random.default_rng(55)
        audio = rng.normal(0, 0.1, (22050, 2)).astype(np.float32)

        regions = [Region(0.0, 0.5, "r1")]
        p, mp, rep = analyze(audio, sr, regions)

        self.assertIsNotNone(rep.tonal_balance_summary)
        self.assertIsNotNone(rep.dynamics_summary)
        self.assertIsNotNone(rep.stereo_summary)
        md = rep.to_markdown()
        self.assertIn("Auto analysis & diagnostic report", md)

    def test_refine_with_sound_goodizer_stages(self):
        """Verify refine() optimizes sound goodizer stages when enabled."""
        sr = 44100
        rng = np.random.default_rng(66)
        audio = rng.normal(0, 0.1, (22050, 2)).astype(np.float32)

        base_params = Params(
            n_fft=512,
            hop=128,
            tonal_balance=0.4,
            transient_punch=0.3,
            bass_tighten=0.2,
            bass_mono=True,
        )
        regions = [Region(0.0, 0.5, "test")]

        selected, summary = refine(
            audio, sr,
            base_params=base_params,
            regions=regions,
            n_trials_per_stage=2,
            refine_dur=0.5,
        )

        stage_names = [s["stage"] for s in summary["stages"]]
        self.assertIn("tonal_balance", stage_names)
        self.assertIn("transient_punch", stage_names)
        self.assertIn("low_end", stage_names)
        self.assertTrue(np.isfinite(summary["final_score"]))
