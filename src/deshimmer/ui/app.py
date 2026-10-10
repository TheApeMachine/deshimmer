"""Gradio Web UI application for Deshimmer."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any, Mapping

import gradio as gr
import numpy as np

from ..autotune import Region, analyze, pick_regions, refine
from ..core.params import MasterParams, Params
from .audio import _to_float_audio
from .handlers import render_full_to_files, run_once
from .presets import (
    DEFAULT_PRESETS,
    load_user_presets,
    save_user_preset,
)
from .state import (
    PARAM_KNOB_NAMES,
    PARAM_STATE_NAMES,
    _build_params,
    _default_param_state,
    _merge_param_state,
    _state_from_dataclasses,
    _state_to_component_values,
)


def build_ui() -> Any:
    """Build the full Gradio Blocks application."""
    user_presets_path = os.path.join(os.getcwd(), "ui_presets.json")
    if not os.path.exists(user_presets_path):
        fallback = os.path.join(os.path.dirname(__file__), "ui_presets.json")
        if os.path.exists(fallback):
            user_presets_path = fallback

    all_presets = dict(DEFAULT_PRESETS)
    all_presets.update(load_user_presets(user_presets_path))

    with gr.Blocks(title="Deshimmer UI") as demo:
        gr.Markdown(
            "## Deshimmer UI\n"
            "Load an audio file, pick a preview region, tweak knobs, then listen to input/output/diff and inspect spectrograms."
        )

        with gr.Row():
            preset = gr.Dropdown(
                choices=list(all_presets.keys()),
                value="02 - Default shimmer (recommended start)",
                label="Presets (simple → complex)",
            )
            preset_apply = gr.Button("Apply preset", variant="secondary")
        preset_desc = gr.Markdown()

        with gr.Row():
            audio_in = gr.Audio(label="Input audio", type="numpy")
            with gr.Column():
                full_song_mode = gr.Checkbox(
                    value=False,
                    label="Full Song Mode",
                    info="Process and play back the entire track in the players below (slower). Preview start/duration are ignored. For WAV exports, use Render full & generate downloads.",
                )
                preview_t0 = gr.Slider(
                    0.0, 600.0, value=0.0, step=0.05,
                    label="Preview start (s)",
                    info="Where in the song to start the preview. Move this to test different sections.",
                )
                preview_dur = gr.Slider(
                    0.5, 20.0, value=6.0, step=0.05,
                    label="Preview duration (s)",
                    info="How long the preview clip should be. Shorter = faster processing, longer = hear more context.",
                )
                loop_xfade_ms = gr.Slider(
                    0.0, 250.0, value=35.0, step=1.0,
                    label="Loop crossfade (ms)",
                    info="Smooth the loop point so it doesn't click. Higher = smoother loop but slightly alters the audio.",
                )
                run_btn = gr.Button("Run preview / full song")

        repair_only_diff = gr.Checkbox(
            value=False, label="Listen to repair-only diff",
            info="Compare repair without alignment, balancing, carving, or mastering. Adds a render pass.",
        )

        with gr.Row():
            a_in = gr.Audio(label="Input", type="numpy", interactive=False)
            a_out = gr.Audio(label="Output", type="numpy", interactive=False, autoplay=True)
            a_diff = gr.Audio(label="Diff / removed", type="numpy", interactive=False)

        with gr.Row():
            im_in = gr.Image(label="Spectrogram: input", type="numpy")
            im_out = gr.Image(label="Spectrogram: output", type="numpy")
            im_diff = gr.Image(label="Spectrogram: diff", type="numpy")

        metrics = gr.Markdown()
        params_json = gr.JSON(label="Effective params")

        with gr.Row():
            full_btn = gr.Button("Render full & generate downloads", variant="primary")
            dl_out = gr.File(label="Download: full output.wav")
            dl_diff = gr.File(label="Download: full diff.wav")
            dl_params = gr.File(label="Download: params.json")

        with gr.Accordion("✨ Auto-tune (Analyze + Refine)", open=True):
            gr.Markdown(
                "*Auto-derive sensible defaults from the audio (Analyze), then optionally refine with a "
                "multi-objective search across a few preview regions (Refine). Both write directly into the sliders below.*"
            )
            with gr.Row():
                auto_aggressiveness = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Aggressiveness",
                    info="0 = preserve content (more conservative), 1 = stronger artifact reduction while retaining music-damage penalties. Refine only.",
                )
                auto_n_trials = gr.Slider(
                    4, 30, value=12, step=1,
                    label="Trials per stage",
                    info="Higher = better tuning but slower. ~12 is a good default. Refine only.",
                )
                auto_refine_dur = gr.Slider(
                    1.0, 6.0, value=3.0, step=0.25,
                    label="Refine region duration (s)",
                    info="Shorter regions = faster optimisation. Refine only.",
                )
            with gr.Row():
                auto_use_preview = gr.Checkbox(
                    value=True,
                    label="Use current preview as one region",
                    info="When on, the current preview start/duration is locked in as one of the auto regions.",
                )
                auto_n_regions = gr.Slider(
                    1, 8, value=5, step=1,
                    label="Number of regions",
                    info="How many short windows to sample (auto-picks quiet/loud/transient).",
                )
            with gr.Row():
                analyze_btn = gr.Button("🔍 Set from audio", variant="secondary")
                refine_btn = gr.Button("⚙️ Optimize", variant="primary")
            auto_report = gr.Markdown()

        with gr.Accordion("Tonal self-consistency (active by default)", open=False):
            gr.Markdown("Learns stable partial relationships from this audio. Shared bends and tuning offsets are retained; ambiguous families are left alone.")
            tonal_repair = gr.Slider(
                0.0, 1.0, value=Params().tonal_repair, step=0.01,
                label="Tonal repair amount", info="Optimize searches this amount automatically.",
            )
            tonal_max_shift_cents = gr.Number(
                value=Params().tonal_max_shift_cents, minimum=0.01,
                label="Maximum partial correction (cents)",
                info="Bound on an individual partial's correction before applying the amount.",
            )

        with gr.Accordion("🎯 Shimmer Removal (main tool for AI sparkle/shimmer)", open=True):
            gr.Markdown("*Targets the annoying 'sparkly' or 'shimmery' artifacts common in AI-generated music. Start here!*")
            with gr.Row():
                start_hz = gr.Number(
                    value=5100.0,
                    label="Start frequency (Hz)",
                    info="Lower edge of the shimmer band. AI shimmer typically lives around 5000-7000 Hz. Lower = catch more low-mid harshness.",
                )
                end_hz = gr.Number(
                    value=7200.0,
                    label="End frequency (Hz)",
                    info="Upper edge of the shimmer band. Higher = catch more high-frequency sparkle. Don't go above your audio's limit.",
                )
                edge_hz = gr.Number(
                    value=200.0,
                    label="Edge softness (Hz)",
                    info="How gradually the effect fades at band edges. Higher = smoother transition, less obvious processing.",
                )
            with gr.Row():
                n_fft = gr.Dropdown(
                    [1024, 2048, 4096, 8192], value=2048,
                    label="FFT size",
                    info="Analysis window size. Larger = better frequency detail but slower. 2048 is a good balance.",
                )
                hop = gr.Dropdown(
                    [256, 512, 1024, 2048], value=512,
                    label="Hop size",
                    info="How much the analysis window moves. Smaller = smoother but slower. 512 works well for most cases.",
                )
            with gr.Row():
                flat_start = gr.Slider(
                    0.0, 1.0, value=0.25, step=0.01,
                    label="Noise detection: start",
                    info="How 'noisy' audio must be before processing kicks in. Lower = more aggressive, catches more but may affect wanted sounds.",
                )
                flat_end = gr.Slider(
                    0.0, 1.0, value=0.70, step=0.01,
                    label="Noise detection: full",
                    info="When audio is this 'noisy', full processing is applied. Higher = only process very noisy parts.",
                )
            with gr.Row():
                freq_med_bins = gr.Slider(
                    3, 61, value=9, step=2,
                    label="Peak detection width",
                    info="How wide to look when finding shimmer peaks. Larger = ignore broader peaks (like wanted harmonics), catch only narrow spikes.",
                )
                thr_db = gr.Slider(
                    0.0, 24.0, value=8.0, step=0.1,
                    label="Threshold (dB)",
                    info="How much louder than surroundings a peak must be to get reduced. Lower = more aggressive, higher = only obvious spikes.",
                )
                slope = gr.Slider(
                    0.0, 2.0, value=0.6, step=0.01,
                    label="Reduction strength",
                    info="How hard to push down detected shimmer. Higher = more reduction but risk of dullness. Start around 0.5-0.7.",
                )
            with gr.Row():
                density_lo = gr.Slider(
                    0.0, 0.5, value=0.02, step=0.005,
                    label="Density: sparse",
                    info="When few peaks are detected, apply full processing. Shimmer is usually sparse.",
                )
                density_hi = gr.Slider(
                    0.0, 0.5, value=0.15, step=0.005,
                    label="Density: dense (back off)",
                    info="When many peaks detected, back off—probably real music content, not artifacts.",
                )
            with gr.Row():
                flux_thr_db = gr.Slider(
                    0.0, 24.0, value=6.0, step=0.1,
                    label="Transient protection: threshold",
                    info="Energy jump (in dB) that triggers transient protection. Protects drum hits and attacks from being dulled.",
                )
                flux_range_db = gr.Slider(
                    0.0, 24.0, value=8.0, step=0.1,
                    label="Transient protection: range",
                    info="How gradually transient protection kicks in. Higher = more gradual fade-in of protection.",
                )
            with gr.Row():
                noise_resynth = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Texture softening",
                    info="Blend in random-phase noise to soften 'crystalline' texture. 0 = off, 0.2-0.4 = subtle softening. Can help with metallic artifacts.",
                )
                mix = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.01,
                    label="Wet/dry mix",
                    info="0 = original audio (bypass), 1 = fully processed. Use values in between for subtle blending.",
                )
                delta_listen = gr.Checkbox(
                    value=False,
                    label="Listen to removed audio",
                    info="Hear what's being removed instead of the result. Useful for checking you're not removing wanted sounds!",
                )

        with gr.Accordion("🔇 Smart Denoise (hiss/noise floor reduction)", open=False):
            gr.Markdown("*Reduces background hiss and noise floor artifacts. Good for AI-generated 'swishy' or 'windy' sounds.*")
            with gr.Row():
                denoise = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Denoise strength",
                    info="Overall strength. 0 = off. Start low (0.2-0.3) and increase. Too high = 'underwater' sound.",
                )
                dn_floor_db = gr.Slider(
                    -60.0, 0.0, value=-18.0, step=0.5,
                    label="Noise floor limit (dB)",
                    info="Never reduce below this level. Prevents complete silence and 'pumping'. More negative = deeper cuts allowed.",
                )
            with gr.Row():
                dn_start_hz = gr.Number(
                    value=120.0,
                    label="Start frequency (Hz)",
                    info="Where to start denoising. Keep above ~100 Hz to avoid affecting bass.",
                )
                dn_end_hz = gr.Number(
                    value=16000.0,
                    label="End frequency (Hz)",
                    info="Where to stop denoising. Usually covers most of audible range.",
                )
                dn_edge_hz = gr.Number(
                    value=200.0,
                    label="Edge softness (Hz)",
                    info="Gradual fade at band edges for smoother processing.",
                )
            with gr.Row():
                dn_psd_smooth_ms = gr.Slider(
                    0.0, 300.0, value=50.0, step=1.0,
                    label="Noise tracking speed (ms)",
                    info="How fast noise estimation adapts. Lower = faster tracking, higher = more stable.",
                )
                dn_minwin_ms = gr.Slider(
                    50.0, 2000.0, value=400.0, step=10.0,
                    label="Minimum window (ms)",
                    info="Time window for finding quietest noise level. Longer = better at finding true noise floor but slower to adapt.",
                )
                dn_up_db_per_s = gr.Slider(
                    0.0, 24.0, value=3.0, step=0.1,
                    label="Noise rise rate (dB/s)",
                    info="How fast estimated noise can increase. Prevents sudden jumps. Lower = more conservative.",
                )
            with gr.Row():
                dn_attack_ms = gr.Slider(
                    0.0, 200.0, value=5.0, step=1.0,
                    label="Attack speed (ms)",
                    info="How fast reduction kicks in when noise detected. Lower = faster but may clip transients.",
                )
                dn_release_ms = gr.Slider(
                    0.0, 1000.0, value=120.0, step=5.0,
                    label="Release speed (ms)",
                    info="How fast reduction backs off. Higher = smoother but may 'pump' on dynamic material.",
                )
                dn_freq_smooth_bins = gr.Slider(
                    1, 21, value=3, step=1,
                    label="Frequency smoothing",
                    info="Smooth reduction across frequencies. Higher = less 'musical noise'/chirping but less precise.",
                )

        with gr.Accordion("📢 De-Resonator (ringing/whine removal)", open=False):
            gr.Markdown("*Targets persistent ringing tones and 'whine' artifacts. Great for Suno/Udio's characteristic 3-4kHz whine.*")
            with gr.Row():
                deres = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="De-resonator strength",
                    info="Overall strength. 0 = off. Effective range is 0.3-0.8. Attacks narrow, persistent frequency peaks.",
                )
                deq_max_att_db = gr.Slider(
                    0.0, 24.0, value=8.0, step=0.1,
                    label="Maximum cut (dB)",
                    info="Cap on how much to reduce resonances. Higher = more aggressive cuts. 6-12 dB is typical.",
                )
            with gr.Row():
                deq_start_hz = gr.Number(
                    value=180.0,
                    label="Start frequency (Hz)",
                    info="Where to look for resonances. Suno/Udio whine often around 3000-4000 Hz.",
                )
                deq_end_hz = gr.Number(
                    value=12000.0,
                    label="End frequency (Hz)",
                    info="Upper limit for resonance detection.",
                )
                deq_edge_hz = gr.Number(
                    value=150.0,
                    label="Edge softness (Hz)",
                    info="Gradual fade at band edges.",
                )
            with gr.Row():
                deq_freq_med_bins = gr.Slider(
                    3, 121, value=31, step=2,
                    label="Peak detection width",
                    info="How wide to look for 'normal' level. Larger = catch wider resonances. For narrow whines, try 31-61.",
                )
                deq_thr_db = gr.Slider(
                    0.0, 24.0, value=6.0, step=0.1,
                    label="Threshold (dB)",
                    info="How much louder than neighbors a peak must be to get cut. Lower = more aggressive.",
                )
                deq_slope = gr.Slider(
                    0.0, 2.0, value=0.7, step=0.01,
                    label="Reduction strength",
                    info="How hard to cut detected resonances. Higher = stronger cuts.",
                )
            with gr.Row():
                deq_density_lo = gr.Slider(
                    0.0, 0.5, value=0.03, step=0.005,
                    label="Density: sparse (full processing)",
                    info="When few peaks, apply full reduction—probably artifacts.",
                )
                deq_density_hi = gr.Slider(
                    0.0, 0.5, value=0.20, step=0.005,
                    label="Density: dense (back off)",
                    info="When many peaks, back off—probably real harmonics.",
                )
            with gr.Row():
                deq_persist_ms = gr.Slider(
                    0.0, 5000.0, value=600.0, step=25.0,
                    label="Persistence memory (ms)",
                    info="How long a peak must persist to be considered a resonance. Catches steady whines, ignores moving melodies.",
                )
                deq_persist_thr_db = gr.Slider(
                    0.0, 24.0, value=2.5, step=0.1,
                    label="Persistence threshold (dB)",
                    info="Accumulated level needed before peak is treated as resonance. Higher = more conservative.",
                )
                deq_freq_smooth_bins = gr.Slider(
                    1, 21, value=5, step=1,
                    label="Frequency smoothing",
                    info="Smooth cuts across frequencies. Higher = less surgical but fewer artifacts.",
                )
                deq_tonal_boost_db = gr.Slider(
                    0.0, 24.0, value=6.0, step=0.1,
                    label="Tonal protection (dB)",
                    info="Raise threshold when audio sounds 'tonal' (not noisy). Protects real harmonics.",
                )
            with gr.Row():
                deq_time_floor = gr.Checkbox(
                    value=False,
                    label="Enable time-floor mode",
                    info="Track bins that are ALWAYS loud—catches stationary ringing lines that never quiet down.",
                )
                deq_floor_smooth_ms = gr.Slider(
                    0.0, 500.0, value=80.0, step=5.0,
                    label="Floor tracking speed (ms)",
                    info="How fast the 'always loud' detector adapts. Lower = faster.",
                )
                deq_floor_rise_db_per_s = gr.Slider(
                    0.0, 12.0, value=1.0, step=0.1,
                    label="Floor rise rate (dB/s)",
                    info="How fast the floor estimate can rise. Lower = stricter about what counts as 'always on'.",
                )
                deq_floor_thr_db = gr.Slider(
                    0.0, 12.0, value=3.0, step=0.1,
                    label="Floor excess threshold (dB)",
                    info="How many dB above the local frequency floor a stationary bin must sit before de-resonance acts.",
                )

        with gr.Accordion("🔬 Advanced Tools (experimental - use with caution)", open=False):
            gr.Markdown("*These are more aggressive/experimental tools. Most users won't need them.*")

            gr.Markdown("**Downward Expander** - Push down quiet parts in a frequency band (reduces lingering artifacts)")
            with gr.Row():
                expander = gr.Checkbox(
                    value=False,
                    label="Enable expander",
                    info="Turn on the downward expander. Reduces quiet sounds in the target band.",
                )
                exp_threshold_db = gr.Slider(
                    -90.0, 0.0, value=-45.0, step=0.5,
                    label="Threshold (dB)",
                    info="Level below which sounds get reduced. More negative = only very quiet sounds affected.",
                )
                exp_ratio = gr.Slider(
                    1.0, 6.0, value=2.0, step=0.05,
                    label="Ratio",
                    info="How much to reduce. 2:1 = moderate, 4:1 = aggressive. Higher = more expansion.",
                )
            with gr.Row():
                exp_start_hz = gr.Number(
                    value=3000.0,
                    label="Start frequency (Hz)",
                    info="Lower edge of expander band.",
                )
                exp_end_hz = gr.Number(
                    value=8000.0,
                    label="End frequency (Hz)",
                    info="Upper edge of expander band.",
                )
                exp_attack_ms = gr.Slider(
                    0.0, 200.0, value=10.0, step=1.0,
                    label="Attack (ms)",
                    info="How fast expansion kicks in. Lower = faster but may affect transients.",
                )
                exp_release_ms = gr.Slider(
                    0.0, 1000.0, value=150.0, step=5.0,
                    label="Release (ms)",
                    info="How fast expansion backs off. Higher = smoother.",
                )

            gr.Markdown("**Harmonic/Percussive Separation** - Protect drums/transients from processing")
            with gr.Row():
                hpss = gr.Checkbox(
                    value=False,
                    label="Enable HPSS",
                    info="Separate harmonic (sustained) from percussive (transient) content.",
                )
                hpss_harmonic_only = gr.Checkbox(
                    value=True,
                    label="Process harmonic only",
                    info="Only process the sustained/harmonic parts, leave drums/transients alone.",
                )
            with gr.Row():
                hpss_start_hz = gr.Number(
                    value=3000.0,
                    label="Start frequency (Hz)",
                    info="Where to apply HPSS separation.",
                )
                hpss_end_hz = gr.Number(
                    value=8000.0,
                    label="End frequency (Hz)",
                    info="Upper limit for HPSS.",
                )
                hpss_time_frames = gr.Slider(
                    3, 61, value=21, step=2,
                    label="Time window",
                    info="Median filter over time. Larger = better separation but more smearing.",
                )
                hpss_freq_bins = gr.Slider(
                    3, 61, value=17, step=2,
                    label="Frequency window",
                    info="Median filter over frequency. Larger = better separation but less precise.",
                )
                hpss_protect_percussive = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05,
                    label="Protect percussive (0–1)",
                    info="Reduce repair depth on percussive bins. 0 = off, 1 = strong cymbal/snare protection.",
                )

            gr.Markdown("**Repair philosophy** — inpainting, combined cap, Mid/Side")
            with gr.Row():
                magnitude_inpaint = gr.Checkbox(
                    value=True,
                    label="Magnitude inpainting (shimmer)",
                    info="Cap spikes to local median instead of pure attenuation (preserves air).",
                )
                deq_inpaint = gr.Checkbox(
                    value=True,
                    label="Magnitude inpainting (de-res)",
                    info="Same inpainting approach for persistent resonances.",
                )
                ms_process = gr.Checkbox(
                    value=False,
                    label="Mid/Side processing",
                    info="Apply full repair on Mid; scale repair on Side to preserve stereo width.",
                )
            with gr.Row():
                total_att_cap_db = gr.Slider(
                    3.0, 24.0, value=12.0, step=0.5,
                    label="Combined attenuation cap (dB)",
                    info="Max product attenuation per bin across denoise/deq/shimmer/expander.",
                )
                ms_side_scale = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.05,
                    label="Side channel repair scale",
                    info="When Mid/Side is on, Side gets this fraction of repair depth.",
                )
                nuclear_mode = gr.Checkbox(
                    value=False,
                    label="Nuclear mode (disable cap)",
                    info="Disables combined attenuation cap. Use with HF resynth only when desperate.",
                )

            gr.Markdown("**Phase Blur** - Soften harsh textures by randomizing phase")
            with gr.Row():
                phase_blur = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Amount",
                    info="How much to randomize phase. 0 = off. Subtle values (0.1-0.3) can soften metallic textures.",
                )
                pb_harmonic_only = gr.Checkbox(
                    value=True,
                    label="Harmonic only",
                    info="Only blur harmonic content, preserve transients.",
                )
            with gr.Row():
                pb_start_hz = gr.Number(
                    value=3000.0,
                    label="Start frequency (Hz)",
                    info="Where to apply phase blur.",
                )
                pb_end_hz = gr.Number(
                    value=8000.0,
                    label="End frequency (Hz)",
                    info="Upper limit for phase blur.",
                )

            gr.Markdown("**Swish Repair** — adaptive phase coherence for moving AI 'swish' (not EQ-able)")
            with gr.Row():
                swish_repair = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Swish repair amount",
                    info="Smooths erratic inter-frame/inter-bin phase where instability is high. Start ~0.3–0.5 on tonal AI material.",
                )
                swish_time_amt = gr.Slider(
                    0.0, 1.0, value=0.55, step=0.01,
                    label="Inter-frame smoothing",
                    info="Time-axis phase coherence (targets moving swish).",
                )
                swish_freq_amt = gr.Slider(
                    0.0, 1.0, value=0.30, step=0.01,
                    label="Inter-bin smoothing",
                    info="Frequency-axis phase coherence.",
                )
            with gr.Row():
                swish_start_hz = gr.Number(
                    value=3500.0,
                    label="Swish band start (Hz)",
                    info="Lower edge for phase repair.",
                )
                swish_end_hz = gr.Number(
                    value=14000.0,
                    label="Swish band end (Hz)",
                    info="Upper edge for phase repair.",
                )
            with gr.Row():
                hf_decorrelate = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="HF decorrelate",
                    info="Break synthetic L/R phase lock in upper band. Try 0.15–0.35 on stereo AI exports.",
                )
                hf_dec_start_hz = gr.Number(
                    value=4500.0,
                    label="Decorrelate start (Hz)",
                )
                hf_dec_end_hz = gr.Number(
                    value=16000.0,
                    label="Decorrelate end (Hz)",
                )

            gr.Markdown("**HF Resynthesis** - Nuclear option: remove HF entirely and regenerate from lower frequencies")
            with gr.Row():
                hf_resynth = gr.Checkbox(
                    value=False,
                    label="Enable HF resynthesis",
                    info="⚠️ Destructive! Removes high frequencies and creates new ones from lower bands. Last resort.",
                )
                hf_mix = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.01,
                    label="Mix amount",
                    info="How much regenerated HF to blend in. Lower = subtler.",
                )
                hf_confidence_blend = gr.Checkbox(
                    value=True,
                    label="Confidence-masked blend",
                    info="Only resynthesize HF where artifact confidence is high.",
                )
            with gr.Row():
                hf_lp_hz = gr.Number(
                    value=3000.0,
                    label="Low-pass cutoff (Hz)",
                    info="Keep everything below this frequency, remove above.",
                )
                hf_hp_hz = gr.Number(
                    value=3000.0,
                    label="Regenerated HF starts at (Hz)",
                    info="Where regenerated harmonics begin.",
                )
                hf_drive = gr.Slider(
                    0.1, 10.0, value=2.0, step=0.05,
                    label="Drive",
                    info="Saturation amount for generating harmonics. Higher = more harmonics but harsher.",
                )
            with gr.Row():
                hf_src_lo_hz = gr.Number(
                    value=1000.0,
                    label="Source band start (Hz)",
                    info="Lower edge of the band used to generate new harmonics.",
                )
                hf_src_hi_hz = gr.Number(
                    value=2000.0,
                    label="Source band end (Hz)",
                    info="Upper edge of the source band.",
                )

        with gr.Accordion("✨ Sound Goodizer: Tonal Balance, Punch & Stereo Translation", open=False):
            gr.Markdown("*Automatic musical enhancement engine: self-adapting tonal balance, punch restoration, mono bass, and translation-safe stereo.*")
            with gr.Row():
                tonal_balance = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Tonal balance strength",
                    info="Overall strength of the self-adapting multi-band EQ.",
                )
                transient_punch = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Punch restoration",
                    info="Restores lost attack definition and microdynamic punch in squashed mixes.",
                )
            with gr.Row():
                tb_low_db = gr.Slider(
                    -3.0, 3.0, value=0.0, step=0.1,
                    label="Low balance (dB)",
                    info="Sub and bass balance (~40-250 Hz).",
                )
                tb_low_mid_db = gr.Slider(
                    -3.0, 3.0, value=0.0, step=0.1,
                    label="Low-mid warmth / de-mud (dB)",
                    info="Controls boxiness and muddiness (~200-900 Hz).",
                )
                tb_mid_db = gr.Slider(
                    -3.0, 3.0, value=0.0, step=0.1,
                    label="Mid presence (dB)",
                    info="Controls vocal and body presence (~700-3000 Hz).",
                )
                tb_presence_db = gr.Slider(
                    -3.0, 3.0, value=0.0, step=0.1,
                    label="Presence / de-harsh (dB)",
                    info="Controls sharpness and bite (~2500-8000 Hz).",
                )
                tb_air_db = gr.Slider(
                    -3.0, 3.0, value=0.0, step=0.1,
                    label="Air sheen (dB)",
                    info="Controls top-end openness and air (>7000 Hz).",
                )
            with gr.Row():
                punch_attack_boost_db = gr.Slider(
                    1.0, 4.0, value=2.0, step=0.1,
                    label="Max attack boost (dB)",
                    info="Maximum transient expansion on attacks.",
                )
                punch_low = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Low-end punch focus",
                    info="Focus punch on kick drum attacks.",
                )
                punch_high = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="High-end snap focus",
                    info="Focus punch on snare and pluck transients.",
                )
            with gr.Row():
                bass_mono = gr.Checkbox(
                    value=False,
                    label="Mono bass",
                    info="Collapses sub-bass to mono to eliminate phase cancellation on speakers and clubs.",
                )
                bass_mono_hz = gr.Slider(
                    60.0, 180.0, value=120.0, step=5.0,
                    label="Mono bass cutoff (Hz)",
                    info="Frequencies below this cutoff are centered in mono.",
                )
                bass_tighten = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Bass tighten",
                    info="Attenuates sustained low-end mud/boom overhang while preserving transient kick hits.",
                )
            with gr.Row():
                stereo_width = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.01,
                    label="Stereo width scaling",
                    info="1.0 = neutral. Adjusts side channel with translation safety.",
                )
                stereo_safe_mono = gr.Checkbox(
                    value=False,
                    label="Stereo mono safety",
                    info="Suppresses anti-phase amplification while preserving exact mono downmix.",
                )
            with gr.Row():
                harmonic_enhance = gr.Slider(
                    0.0, 0.5, value=0.0, step=0.01,
                    label="Harmonic sheen exciter",
                    info="Musically coherent overtone enhancement in the upper frequencies.",
                )
                harmonic_start_hz = gr.Number(
                    value=3500.0,
                    label="Harmonic sheen start frequency (Hz)",
                    info="Frequency threshold above which harmonic sheen is blended.",
                )

        with gr.Accordion("🎚️ Delivery Mastering (loudness/limiting)", open=False):
            gr.Markdown("*Final loudness normalization and limiting for delivery. Only enable when you're done tweaking!*")
            master_enabled = gr.Checkbox(
                value=False,
                label="Enable mastering",
                info="Turn on loudness normalization and limiting. Leave off while tweaking parameters.",
            )
            with gr.Row():
                hp_hz = gr.Slider(
                    0.0, 80.0, value=20.0, step=0.5,
                    label="High-pass filter (Hz)",
                    info="Remove sub-bass rumble. 20-30 Hz is typical. Set to 0 to disable.",
                )
                tp_os = gr.Dropdown(
                    [1, 2, 4, 8], value=4,
                    label="True peak oversampling",
                    info="Oversampling for accurate peak detection. 4x is standard, 8x is more accurate but slower.",
                )
            with gr.Row():
                target_lufs = gr.Slider(
                    -30.0, 999.0, value=-14.0, step=0.1,
                    label="Target loudness (LUFS)",
                    info="-14 LUFS = Spotify/YouTube. -16 = Apple. Set >=998 to disable loudness normalization.",
                )
                target_rms_dbfs = gr.Slider(
                    -40.0, -6.0, value=-16.0, step=0.1,
                    label="Fallback RMS target (dBFS)",
                    info="Used if LUFS measurement fails. -16 to -14 is typical.",
                )
            with gr.Row():
                norm_max_gain_db = gr.Slider(
                    0.0, 24.0, value=12.0, step=0.5,
                    label="Max boost (dB)",
                    info="Maximum gain increase allowed. Prevents over-boosting quiet tracks.",
                )
                norm_max_atten_db = gr.Slider(
                    0.0, 60.0, value=24.0, step=0.5,
                    label="Max cut (dB)",
                    info="Maximum gain reduction allowed. Prevents over-cutting loud tracks.",
                )
            with gr.Row():
                ceiling_dbtp = gr.Slider(
                    -12.0, 0.0, value=-1.0, step=0.1,
                    label="True peak ceiling (dBTP)",
                    info="Maximum peak level. -1 dBTP is safe for streaming, -0.5 for CD.",
                )
                lim_lookahead_ms = gr.Slider(
                    0.0, 50.0, value=5.0, step=0.5,
                    label="Limiter lookahead (ms)",
                    info="How far ahead the limiter looks. Higher = cleaner limiting but adds latency.",
                )
                lim_release_ms = gr.Slider(
                    10.0, 1000.0, value=100.0, step=5.0,
                    label="Limiter release (ms)",
                    info="How fast limiter recovers. Lower = more aggressive, higher = smoother.",
                )

        with gr.Accordion("Spectrogram settings (UI only)", open=False):
            with gr.Row():
                spec_n_fft = gr.Dropdown([512, 1024, 2048, 4096], value=2048, label="spec_n_fft")
                spec_hop = gr.Dropdown([128, 256, 512, 1024], value=512, label="spec_hop")
            with gr.Row():
                spec_max_frames = gr.Slider(200, 4000, value=1200, step=50, label="spec_max_frames")
                spec_max_hz = gr.Slider(2000.0, 24000.0, value=20000.0, step=100.0, label="spec_max_hz")

        with gr.Accordion("Preset management", open=False):
            preset_name = gr.Textbox(label="New preset name", placeholder="e.g. My grit killer v1")
            preset_user_desc = gr.Textbox(label="Preset description (optional)")
            preset_save = gr.Button("Save preset from current knobs", variant="primary")
            preset_save_status = gr.Markdown()

        params_state = gr.State(_default_param_state())
        _component_scope = locals()
        param_components: dict[str, Any] = {
            name: _component_scope[name] for name in PARAM_STATE_NAMES
        }
        param_outputs = [param_components[name] for name in PARAM_STATE_NAMES]
        result_outputs = [a_in, a_out, a_diff, im_in, im_out, im_diff, metrics, params_json]
        render_inputs = [audio_in, preview_t0, preview_dur, loop_xfade_ms, full_song_mode, params_state, repair_only_diff]

        def _preset_desc_md(name: str) -> str:
            p = all_presets.get(name, {})
            desc = p.get("desc", "")
            return f"**{name}**  \n{desc}" if desc else f"**{name}**"

        def apply_preset(name: str, state: Mapping[str, object] | None) -> tuple[Any, ...]:
            preset_data = all_presets.get(name, {})
            values = preset_data.get("values", {})
            updates = values if isinstance(values, Mapping) else {}
            new_state = _merge_param_state(state, updates)
            return tuple([new_state] + _state_to_component_values(new_state) + [_preset_desc_md(name)])

        preset.change(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])
        preset_apply.click(
            fn=apply_preset,
            inputs=[preset, params_state],
            outputs=[params_state] + param_outputs + [preset_desc],
        ).then(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        def do_analyze(
            audio_in_v: tuple[int, np.ndarray] | None,
            preview_t0_v: float,
            preview_dur_v: float,
            use_preview_v: bool,
            n_regions_v: float,
            state: Mapping[str, object] | None,
        ) -> tuple[Any, ...]:
            current_state = _merge_param_state(state, {})
            if audio_in_v is None:
                report = "**Auto-analyze:** please load an audio file first."
                return tuple([current_state] + _state_to_component_values(current_state) + [report])

            sr_v, x_v = audio_in_v
            x_v = _to_float_audio(x_v)
            base_p, _base_mp, _base_dp = _build_params(current_state)

            locked = None
            if use_preview_v:
                locked = Region(t0=float(preview_t0_v), dur=float(preview_dur_v), label="user")
            n_reg = max(1, round(float(n_regions_v)))
            regions = pick_regions(x_v, sr_v, n=n_reg, dur=max(2.0, float(preview_dur_v)), locked=locked)
            new_p, new_mp, report = analyze(x_v, sr_v, regions, base_params=base_p)
            new_state = _merge_param_state(current_state, _state_from_dataclasses(new_p, new_mp))
            return tuple([new_state] + _state_to_component_values(new_state) + [report.to_markdown()])

        def do_refine(
            audio_in_v: tuple[int, np.ndarray] | None,
            preview_t0_v: float,
            preview_dur_v: float,
            use_preview_v: bool,
            n_regions_v: float,
            aggressiveness_v: float,
            n_trials_v: float,
            refine_dur_v: float,
            state: Mapping[str, object] | None,
            progress: Any = gr.Progress(track_tqdm=False),
        ) -> tuple[Any, ...]:
            current_state = _merge_param_state(state, {})
            if audio_in_v is None:
                report = "**Auto-refine:** please load an audio file first."
                return tuple([current_state] + _state_to_component_values(current_state) + [report])

            sr_v, x_v = audio_in_v
            x_v = _to_float_audio(x_v)
            base_p, base_mp, _base_dp = _build_params(current_state)

            locked = None
            if use_preview_v:
                locked = Region(t0=float(preview_t0_v), dur=float(preview_dur_v), label="user")
            n_reg = max(1, round(float(n_regions_v)))
            refine_dur = float(max(1.0, min(refine_dur_v, preview_dur_v + 1.0)))
            region_dur = float(max(refine_dur + 0.5, preview_dur_v, 4.0))
            regions = pick_regions(x_v, sr_v, n=n_reg, dur=region_dur, locked=locked)

            def _cb(frac: float, msg: str) -> None:
                try:
                    progress(float(frac), desc=msg)
                except Exception:
                    pass

            progress(0.0, desc="Refining...")
            new_p, summary = refine(
                x_v,
                sr_v,
                base_params=base_p,
                regions=regions,
                aggressiveness=float(aggressiveness_v),
                n_trials_per_stage=int(n_trials_v),
                refine_dur=refine_dur,
                progress_cb=_cb,
            )

            stages_md_lines = ["**Auto-refine results**", ""]
            stages_md_lines.append(
                f"- Aggressiveness: **{float(aggressiveness_v):.2f}**, "
                f"trials/stage: **{int(n_trials_v)}**, regions: {len(regions)}"
            )
            stages_md_lines.append(f"- Final composite score: **{summary.get('final_score', 0.0):+.3f}**")
            for s in summary.get("stages", []):
                stages_md_lines.append(f"- Stage **{s['stage']}**: selected={s.get('selection_mode', 'balanced')}, score={s.get('selected_score', 0.0):+.3f}")
                pareto = s.get("pareto", {})
                if pareto:
                    for mode in ("safe", "balanced", "aggressive"):
                        pt = pareto.get(mode, {})
                        vals = pt.get("values") or []
                        if vals:
                            stages_md_lines.append(
                                f"  - **{mode}**: artifact={vals[0]:+.2f}, "
                                f"music={-vals[1]:+.2f}, stereo={-vals[3]:+.2f}, tonal={vals[5]:+.2f}"
                            )
                bp = s.get("selected_params", s.get("best_params", {}))
                if bp:
                    bp_s = ", ".join(f"`{k}`={v:.3g}" if isinstance(v, float) else f"`{k}`={v}" for k, v in bp.items())
                    stages_md_lines.append(f"  - {bp_s}")
            failure_log = summary.get("failure_log")
            if failure_log:
                stages_md_lines.append(f"- Failed-trial diagnostics: `{failure_log}`")
            stages_md_lines.append("")
            stages_md_lines.append("_(Aggressiveness weights six NSGA-II objectives, including tonal coherence. Selection and final score use their total utility. Current settings are retained if proposals score worse.)_")

            new_state = _merge_param_state(current_state, _state_from_dataclasses(new_p, base_mp))
            return tuple([new_state] + _state_to_component_values(new_state) + ["\n".join(stages_md_lines)])

        analyze_btn.click(
            fn=do_analyze,
            inputs=[audio_in, preview_t0, preview_dur, auto_use_preview, auto_n_regions, params_state],
            outputs=[params_state] + param_outputs + [auto_report],
        ).then(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        refine_btn.click(
            fn=do_refine,
            inputs=[
                audio_in,
                preview_t0,
                preview_dur,
                auto_use_preview,
                auto_n_regions,
                auto_aggressiveness,
                auto_n_trials,
                auto_refine_dur,
                params_state,
            ],
            outputs=[params_state] + param_outputs + [auto_report],
        ).then(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        demo.load(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])
        run_btn.click(fn=run_once, inputs=render_inputs, outputs=result_outputs)
        repair_only_diff.change(fn=run_once, inputs=render_inputs, outputs=result_outputs)
        full_btn.click(fn=render_full_to_files, inputs=[audio_in, params_state, repair_only_diff], outputs=[dl_out, dl_diff, dl_params])

        def _bind_preview_auto(comp: Any) -> None:
            handler = getattr(comp, "release", None)
            if callable(handler):
                handler(fn=run_once, inputs=render_inputs, outputs=result_outputs)
                return
            comp.change(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        def _make_state_updater(key: str):
            def _update(value: object, state: Mapping[str, object] | None) -> dict[str, object]:
                return _merge_param_state(state, {key: value})
            return _update

        def _bind_param_auto(key: str, comp: Any) -> None:
            updater = _make_state_updater(key)
            handler = getattr(comp, "release", None)
            if callable(handler):
                handler(fn=updater, inputs=[comp, params_state], outputs=[params_state]).then(
                    fn=run_once, inputs=render_inputs, outputs=result_outputs
                )
                return
            comp.change(fn=updater, inputs=[comp, params_state], outputs=[params_state]).then(
                fn=run_once, inputs=render_inputs, outputs=result_outputs
            )

        for comp in (preview_t0, preview_dur, loop_xfade_ms, full_song_mode):
            _bind_preview_auto(comp)
        for key, comp in param_components.items():
            _bind_param_auto(key, comp)

        def save_preset_from_state(
            preset_name_v: str,
            desc: str,
            state: Mapping[str, object] | None,
        ) -> tuple[Any, str]:
            nonlocal all_presets
            clean_name = (preset_name_v or "").strip()
            if not clean_name:
                return gr.update(), "Please enter a preset name."
            current_state = _merge_param_state(state, {})
            values = {key: current_state[key] for key in PARAM_KNOB_NAMES}
            save_user_preset(user_presets_path, clean_name, (desc or "").strip(), values)
            all_presets = dict(DEFAULT_PRESETS)
            all_presets.update(load_user_presets(user_presets_path))
            return gr.update(choices=list(all_presets.keys()), value=clean_name), f"Saved preset to `{user_presets_path}`."

        preset_save.click(
            fn=save_preset_from_state,
            inputs=[preset_name, preset_user_desc, params_state],
            outputs=[preset, preset_save_status],
        ).then(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])

        gr.Markdown(
            "### Tip\n"
            "- Use a short preview (3–8s) around a known-problem spot for fast iteration.\n"
            "- When you like it, run the CLI on the full file using the same values."
        )

    return demo


def main() -> int:
    """Launch the Gradio web interface."""
    demo = build_ui()
    demo.queue()
    demo.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
