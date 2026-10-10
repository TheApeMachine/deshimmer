"""Regression tests for Sound Goodizer: Tonal Balance, Punch, Low-End Coherence, and Stereo Translation."""

import unittest
from dataclasses import replace

import numpy as np

from deshimmer.api import process_audio
from deshimmer.autotune import Region, analyze, refine
from deshimmer.autotune.optimizer import _evaluate_params_multi, _evaluate_params_single
from deshimmer.autotune.scoring import _onset_envelope, compute_enhancement_rewards
from deshimmer.autotune.types import Weights
from deshimmer.cli import build_parser
from deshimmer.core import DSPContext, DSPPipeline, Params, process_stft
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
        """Verify StereoTranslationStage preserves exact mono downmix while safely adjusting width."""
        sr = 44100
        rng = np.random.default_rng(88)

        # Exact mathematical contract test on STFT context:
        # Mid is defined as M = (L+R)/2. Since only Side S = (L-R)/2 is scaled:
        # L_out = M + S_eff, R_out = M - S_eff => (L_out + R_out)/2 = M identically.
        p_wide = Params(stereo_width=1.3, stereo_safe_mono=True)
        stage_wide = StereoTranslationStage.from_params(p_wide)
        z_init = (rng.normal(0, 1, (1025, 10, 2)) + 1j * rng.normal(0, 1, (1025, 10, 2))).astype(np.complex64)
        ctx = DSPContext.create(z_init, sr=sr, p=p_wide)
        m_in = 0.5 * (ctx.Z[:, :, 0] + ctx.Z[:, :, 1])
        stage_wide.process(ctx)
        m_out = 0.5 * (ctx.Z[:, :, 0] + ctx.Z[:, :, 1])
        np.testing.assert_allclose(m_out, m_in, atol=1e-6)

        # Anti-phase correlation safety test on stereo audio signal
        l_sig = rng.normal(0, 0.1, 4096).astype(np.float32)
        r_sig = -0.9 * l_sig + 0.1 * rng.normal(0, 0.1, 4096).astype(np.float32)
        anti_stereo = np.column_stack((l_sig, r_sig))

        p_unsafe = Params(stereo_width=1.4, stereo_safe_mono=False, tonal_repair=0.0)
        p_safe = Params(stereo_width=1.4, stereo_safe_mono=True, tonal_repair=0.0)
        out_unsafe = process_stft(anti_stereo, sr, p_unsafe, pipeline=DSPPipeline([StereoTranslationStage.from_params(p_unsafe)]))
        out_safe = process_stft(anti_stereo, sr, p_safe, pipeline=DSPPipeline([StereoTranslationStage.from_params(p_safe)]))

        corr_unsafe = float(np.sum(out_unsafe[:, 0] * out_unsafe[:, 1]))
        corr_safe = float(np.sum(out_safe[:, 0] * out_safe[:, 1]))
        self.assertGreater(corr_safe, corr_unsafe)

    def test_harmonic_enhance_stage(self):
        """Verify HarmonicEnhanceStage synthesizes new overtones where none existed."""
        sr = 44100
        # Exactly 1 second so 1000 Hz falls on exact integer bin 1000 with 0 sidelobe leakage
        t = np.arange(44100) / sr
        tone = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)[:, None]
        f_fft = np.fft.rfftfreq(len(t), d=1.0 / sr)
        hf_idx = np.where(f_fft >= 2500.0)[0]

        # Prior to enhancement, energy above 2500 Hz is 0
        mag_in = np.abs(np.fft.rfft(tone[:, 0]))
        self.assertAlmostEqual(float(np.max(mag_in[hf_idx])), 0.0, places=4)

        # Apply harmonic enhancement targeting harmonics above 2500 Hz
        p_harm = Params(harmonic_enhance=0.45, harmonic_start_hz=2500.0)
        pipe_harm = DSPPipeline([HarmonicEnhanceStage.from_params(p_harm)])
        out_harm = process_stft(tone, sr, p_harm, pipeline=pipe_harm)
        mag_out = np.abs(np.fft.rfft(out_harm[:, 0] if out_harm.ndim == 2 else out_harm))

        # Check newly synthesized harmonics appear above 2500 Hz (3rd harmonic ~3000 Hz, 4th ~4000 Hz)
        h3_idx = np.argmin(np.abs(f_fft - 3000.0))
        h4_idx = np.argmin(np.abs(f_fft - 4000.0))
        self.assertGreater(float(mag_out[h3_idx]), 0.01)
        self.assertGreater(float(mag_out[h4_idx]), 0.005)
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

    def test_low_end_preserves_musical_bass_harmonics(self):
        """Verify that LowEndStage with bass_tighten preserves pitched musical notes with harmonics."""
        sr = 44100
        t = np.arange(8192) / sr
        # Pitched musical bass note (60 Hz fundamental + 120 Hz 2nd harmonic)
        musical_bass = (0.5 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)[:, None]
        p_tight = Params(bass_tighten=0.8, enhance=True, tonal_repair=0.0)
        pipe = DSPPipeline([LowEndStage.from_params(p_tight)])
        out_musical = np.asarray(process_stft(musical_bass, sr, p_tight, pipeline=pipe)).reshape(-1)

        f_fft = np.fft.rfftfreq(len(t), d=1.0 / sr)
        k60 = np.argmin(np.abs(f_fft - 60.0))
        m_mus_60 = np.abs(np.fft.rfft(out_musical))[k60]
        m_in_60 = np.abs(np.fft.rfft(musical_bass[:, 0]))[k60]
        # Pitched bass with harmonics should retain >= 99% of its 60 Hz fundamental energy
        self.assertGreaterEqual(float(m_mus_60 / m_in_60), 0.99)

    def test_pure_sine_synth_bass_preserved(self):
        """Verify LowEndStage does NOT mistake an intentional pure sine synth bass for unwanted boom resonance."""
        sr = 44100
        t = np.arange(8192) / sr
        # Pure 60 Hz sine wave with no harmonics (e.g. 808 sub or clean synth bass)
        sine_bass = (0.5 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)[:, None]

        p_tight = Params(bass_tighten=0.8, enhance=True, tonal_repair=0.0)
        pipe = DSPPipeline([LowEndStage.from_params(p_tight)])
        out_sine = np.asarray(process_stft(sine_bass, sr, p_tight, pipeline=pipe)).reshape(-1)

        f_fft = np.fft.rfftfreq(len(t), d=1.0 / sr)
        k60 = np.argmin(np.abs(f_fft - 60.0))
        m_out_60 = np.abs(np.fft.rfft(out_sine))[k60]
        m_in_60 = np.abs(np.fft.rfft(sine_bass[:, 0]))[k60]
        # A steady pure sine bass note has no transient overhang decay, so it must not be attenuated
        self.assertGreaterEqual(float(m_out_60 / m_in_60), 0.99)

    def test_wide_mix_with_coherent_sub_bass_preserves_stereo(self):
        """Verify analyzer does not falsely trigger bass_mono on wide mixes with coherent center bass."""
        sr = 44100
        rng = np.random.default_rng(101)
        n = 44100 * 2
        t = np.arange(n) / sr

        # Coherent center mono sub-bass at 50 Hz
        sub_bass = 0.15 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
        # Wide, uncorrelated high/mid stereo material (ambient pads, ping-pong synths)
        mid_l = rng.normal(0, 0.35, n).astype(np.float32)
        mid_r = rng.normal(0, 0.35, n).astype(np.float32)

        stereo = np.column_stack((sub_bass + mid_l, sub_bass + mid_r))
        corr = float(np.sum(stereo[:, 0] * stereo[:, 1]) / (np.sqrt(np.sum(stereo[:, 0] ** 2) * np.sum(stereo[:, 1] ** 2)) + 1e-12))
        # Mix is wide (low broadband correlation)
        self.assertLess(corr, 0.35)

        # Analyze audio
        p_diag, _, _ = analyze(stereo, sr)
        # Bass is completely mono, so bass_mono must NOT be activated despite low broadband correlation
        self.assertFalse(p_diag.bass_mono)
        self.assertTrue(p_diag.stereo_safe_mono)

    def test_transient_onset_alignment_with_synthetic_impulses(self):
        """Verify onset envelope frame indices align exactly with sample windowing (hop=256)."""
        sr = 44100
        n_samples = 16384
        audio = np.zeros((n_samples, 1), dtype=np.float32)
        # Place synthetic impulses at known exact sample indices
        impulse_samples = [2048, 6144, 10240]
        for s_idx in impulse_samples:
            audio[s_idx:s_idx + 32, 0] = 0.9

        hop = 256
        env = _onset_envelope(audio, sr, hop=hop)
        # Peak frames in onset envelope must match s_idx // hop
        expected_frames = [s // hop for s in impulse_samples]
        for exp_f in expected_frames:
            # Check maximum in local neighborhood (+/- 1 frame) around expected frame
            local_max_f = exp_f - 1 + int(np.argmax(env[max(0, exp_f - 1):exp_f + 2]))
            self.assertEqual(local_max_f, exp_f)

        # Test transient punch reward on boosted attacks
        audio_boosted = audio.copy()
        for s_idx in impulse_samples:
            audio_boosted[s_idx:s_idx + 32, 0] *= 1.5

        p_punch = Params(transient_punch=0.5)
        rewards = compute_enhancement_rewards(audio, audio_boosted, sr, "transient_punch", p_punch)
        self.assertIn(2, rewards)
        self.assertGreater(rewards[2], 0.0)

    def test_optimizer_consistent_objective_scoring(self):
        """Verify baseline, candidate, and final evaluation use consistent objective scoring."""
        from deshimmer.autotune import weights_from_aggressiveness

        sr = 44100
        audio = np.random.default_rng(55).normal(0, 0.1, (44100, 2)).astype(np.float32)
        regions = [Region(0.0, 0.5, "intro")]
        weights = weights_from_aggressiveness(0.5)
        band = (3000.0, 16000.0)
        p_base = Params(tonal_balance=0.5, tb_low_mid_db=-1.5)

        # Baseline evaluated with stage="tonal_balance"
        base_obj = _evaluate_params_multi(audio, sr, regions, p_base, weights, band, 0.5, stage="tonal_balance")
        # Candidate evaluated with stage="tonal_balance"
        cand_obj = _evaluate_params_multi(audio, sr, regions, p_base, weights, band, 0.5, stage="tonal_balance")
        # Exact mathematical equality on identical parameter configuration
        np.testing.assert_allclose(base_obj, cand_obj, atol=1e-6)

        # Single evaluation with stage="all"
        single_val = _evaluate_params_single(audio, sr, regions, p_base, weights, band, 0.5, stage="all")
        multi_val = sum(_evaluate_params_multi(audio, sr, regions, p_base, weights, band, 0.5, stage="all"))
        self.assertAlmostEqual(single_val, multi_val, places=5)

    def test_optimizer_enhancement_rewards(self):
        """Verify compute_enhancement_rewards gives positive score for genuine acoustic improvements."""
        sr = 44100
        rng = np.random.default_rng(42)
        audio = rng.normal(0, 0.1, (8192, 2)).astype(np.float32)

        # Test tonal_balance reward
        p_tb = Params(tonal_balance=0.5, tb_low_mid_db=-2.0, tb_air_db=2.0)
        y_tb = process_stft(audio, sr, p_tb)
        rewards_tb = compute_enhancement_rewards(audio, y_tb, sr, "tonal_balance", p_tb)
        self.assertIn(4, rewards_tb)
        self.assertGreaterEqual(rewards_tb[4], 0.0)

        # Test transient_punch reward
        p_punch = Params(transient_punch=0.5, punch_attack_boost_db=2.5)
        y_punch = process_stft(audio, sr, p_punch)
        rewards_punch = compute_enhancement_rewards(audio, y_punch, sr, "transient_punch", p_punch)
        self.assertIn(2, rewards_punch)
        self.assertGreaterEqual(rewards_punch[2], 0.0)

    def test_defaults_consistency(self):
        """Verify consistent default stereo_safe_mono across Params and CLI."""
        p = Params()
        self.assertFalse(p.stereo_safe_mono)

        parser = build_parser()
        args = parser.parse_args(["in.wav", "out.wav"])
        self.assertFalse(args.stereo_safe_mono)
