"""Preset definitions and user preset file management for Deshimmer UI."""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    # --- Simple (few knobs) ---
    "01 - Bypass (no processing)": {
        "desc": "Mix=0.0 (fully dry). Useful sanity check.",
        "values": {"mix": 0.0, "delta_listen": False, "noise_resynth": 0.0, "denoise": 0.0, "deres": 0.0, "master_enabled": False},
    },
    "02 - Default shimmer (recommended start)": {
        "desc": "Conservative shimmer suppression in 5.1–7.2 kHz.",
        "values": {"mix": 1.0, "thr_db": 8.0, "slope": 0.6, "noise_resynth": 0.0, "denoise": 0.0, "deres": 0.0, "master_enabled": False},
    },
    "03 - Gentle shimmer": {
        "desc": "Higher threshold + lower slope; safest for already-good material.",
        "values": {"thr_db": 10.0, "slope": 0.45, "noise_resynth": 0.0, "mix": 1.0},
    },
    "04 - Aggressive shimmer": {
        "desc": "Lower threshold + higher slope. Can dull cymbals; use preview loop.",
        "values": {"thr_db": 6.5, "slope": 0.85, "noise_resynth": 0.0, "mix": 1.0},
    },
    "05 - Shimmer + de-crystallize (noise resynth)": {
        "desc": "Adds subtle random-phase blend in noise-like frames to soften sparkle.",
        "values": {"thr_db": 8.0, "slope": 0.6, "noise_resynth": 0.25, "mix": 1.0},
    },
    # --- Medium (introduce denoise / deres) ---
    "06 - Shimmer + light denoise": {
        "desc": "Taste-neutral noise floor management; good for codec hiss / swish.",
        "values": {"denoise": 0.25, "dn_floor_db": -18.0, "dn_freq_smooth_bins": 3, "mix": 1.0},
    },
    "07 - Shimmer + stronger denoise": {
        "desc": "More denoise depth, more smoothing to avoid musical noise.",
        "values": {"denoise": 0.55, "dn_floor_db": -18.0, "dn_freq_smooth_bins": 5, "dn_release_ms": 180.0, "mix": 1.0},
    },
    "08 - De-resonator (gentle)": {
        "desc": "Dynamic EQ for persistent narrow resonances; conservative.",
        "values": {"deres": 0.35, "deq_thr_db": 7.0, "deq_max_att_db": 6.0, "deq_persist_ms": 800.0, "mix": 1.0},
    },
    "09 - De-resonator (stronger)": {
        "desc": "More active notch behavior on persistent peaks (still capped).",
        "values": {"deres": 0.70, "deq_thr_db": 6.0, "deq_max_att_db": 10.0, "deq_persist_ms": 700.0, "deq_freq_smooth_bins": 7, "mix": 1.0},
    },
    # --- Complex (touch many knobs) ---
    "10 - Full stack (conservative)": {
        "desc": "Shimmer + light denoise + gentle deres. Good 'set-and-forget' baseline.",
        "values": {
            "thr_db": 8.5,
            "slope": 0.55,
            "noise_resynth": 0.10,
            "denoise": 0.25,
            "dn_floor_db": -18.0,
            "dn_psd_smooth_ms": 60.0,
            "dn_minwin_ms": 450.0,
            "dn_release_ms": 140.0,
            "dn_freq_smooth_bins": 5,
            "deres": 0.35,
            "deq_thr_db": 7.0,
            "deq_max_att_db": 7.0,
            "deq_persist_ms": 900.0,
            "deq_persist_thr_db": 2.8,
            "deq_freq_smooth_bins": 7,
            "mix": 1.0,
            "master_enabled": False,
        },
    },
    "11 - Full stack (aggressive repair)": {
        "desc": "More denoise + more shimmer suppression + stronger deres. Watch for dullness.",
        "values": {
            "thr_db": 7.0,
            "slope": 0.85,
            "noise_resynth": 0.20,
            "denoise": 0.60,
            "dn_floor_db": -20.0,
            "dn_psd_smooth_ms": 40.0,
            "dn_minwin_ms": 350.0,
            "dn_release_ms": 220.0,
            "dn_freq_smooth_bins": 7,
            "deres": 0.75,
            "deq_thr_db": 6.0,
            "deq_max_att_db": 12.0,
            "deq_persist_ms": 650.0,
            "deq_persist_thr_db": 2.2,
            "deq_freq_smooth_bins": 9,
            "mix": 1.0,
            "master_enabled": False,
        },
    },
    "12 - Delivery: -14 LUFS, -1 dBTP": {
        "desc": "Enable mastering stage for streaming-ish loudness.",
        "values": {"master_enabled": True, "target_lufs": -14.0, "ceiling_dbtp": -1.0, "tp_os": 4, "hp_hz": 20.0},
    },
    "13 - Delivery: louder (-10 LUFS), -1 dBTP": {
        "desc": "Hotter delivery (can pump/limit more).",
        "values": {"master_enabled": True, "target_lufs": -10.0, "ceiling_dbtp": -1.0, "tp_os": 4, "hp_hz": 20.0},
    },
    # --- Band experiments ---
    "14 - Band experiment: 2k–6k (harshness zone)": {
        "desc": "Useful if the artifact is more 'presence harshness' than 6–7k sparkle.",
        "values": {"start_hz": 2000.0, "end_hz": 6000.0, "thr_db": 8.0, "slope": 0.6, "mix": 1.0},
    },
    "15 - Band experiment: 5.8k–7.8k": {
        "desc": "Shift shimmer band upward a bit.",
        "values": {"start_hz": 5800.0, "end_hz": 7800.0, "thr_db": 8.0, "slope": 0.6, "mix": 1.0},
    },
    "16 - Suno/Udio: 3.5k whine (exact CLI values)": {
        "desc": "Exactly: --start-hz 3000 --end-hz 4200 --deres 0.8 --deq-thr-db 4.0 --deq-max-att-db 12.0 --deq-freq-med-bins 61 --deq-persist-ms 1000 --thr-db 6.0 --slope 0.9",
        "values": {
            "start_hz": 3000.0,
            "end_hz": 4200.0,
            "deres": 0.8,
            "deq_thr_db": 4.0,
            "deq_max_att_db": 12.0,
            "deq_freq_med_bins": 61,
            "deq_persist_ms": 1000.0,
            "thr_db": 6.0,
            "slope": 0.9,
        },
    },
    "17 - Suno/Udio: 3.5k whine + metallic crickets (full recipe)": {
        "desc": "Whine recipe + denoise/noise-resynth for 'wind/crickets'. Also enables time-stabilized floor for stationary lines.",
        "values": {
            "start_hz": 3000.0,
            "end_hz": 4200.0,
            "thr_db": 6.0,
            "slope": 0.9,
            "deres": 0.8,
            "deq_thr_db": 4.0,
            "deq_max_att_db": 12.0,
            "deq_freq_med_bins": 61,
            "deq_persist_ms": 1000.0,
            "deq_time_floor": True,
            "deq_floor_smooth_ms": 80.0,
            "deq_floor_rise_db_per_s": 1.0,
            "deq_floor_thr_db": 3.0,
            "denoise": 0.8,
            "dn_start_hz": 3000.0,
            "dn_end_hz": 16000.0,
            "dn_floor_db": -24.0,
            "dn_minwin_ms": 1000.0,
            "dn_attack_ms": 10.0,
            "dn_release_ms": 150.0,
            "dn_freq_smooth_bins": 5,
            "noise_resynth": 0.35,
            "mix": 1.0,
            "master_enabled": False,
        },
    },
}


def load_user_presets(path: str) -> dict[str, dict[str, Any]]:
    """Read user presets from json file."""
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            vals = v.get("values", {})
            if not isinstance(vals, dict):
                continue
            out[k] = {"desc": str(v.get("desc", "")), "values": vals}
        return out
    except Exception:
        return {}


def save_user_preset(path: str, name: str, desc: str, values: dict[str, Any]) -> None:
    """Save user preset to json file."""
    data = load_user_presets(path)
    data[name] = {"desc": desc, "values": values}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_all_presets(user_presets_path: str) -> dict[str, dict[str, Any]]:
    """Return dictionary of default plus user-defined presets."""
    all_presets = dict(DEFAULT_PRESETS)
    all_presets.update(load_user_presets(user_presets_path))
    return all_presets
