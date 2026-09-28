"""State conversions, default values, and dataclass parameter reconstruction for UI."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from typing import Mapping

from ..core.params import DebugParams, MasterParams, Params

PARAM_KNOB_NAMES: tuple[str, ...] = (
    "start_hz", "end_hz", "edge_hz", "n_fft", "hop",
    "flat_start", "flat_end", "freq_med_bins", "thr_db", "slope",
    "density_lo", "density_hi", "flux_thr_db", "flux_range_db",
    "noise_resynth", "mix", "delta_listen",
    "tonal_repair", "tonal_max_shift_cents",
    "denoise", "dn_start_hz", "dn_end_hz", "dn_edge_hz", "dn_floor_db",
    "dn_psd_smooth_ms", "dn_minwin_ms", "dn_up_db_per_s", "dn_attack_ms",
    "dn_release_ms", "dn_freq_smooth_bins",
    "deres", "deq_start_hz", "deq_end_hz", "deq_edge_hz", "deq_freq_med_bins",
    "deq_thr_db", "deq_slope", "deq_max_att_db", "deq_density_lo",
    "deq_density_hi", "deq_persist_ms", "deq_persist_thr_db",
    "deq_freq_smooth_bins", "deq_tonal_boost_db", "deq_time_floor",
    "deq_floor_smooth_ms", "deq_floor_rise_db_per_s", "deq_floor_thr_db",
    "expander", "exp_start_hz", "exp_end_hz", "exp_threshold_db",
    "exp_ratio", "exp_attack_ms", "exp_release_ms",
    "hpss", "hpss_start_hz", "hpss_end_hz", "hpss_time_frames",
    "hpss_freq_bins", "hpss_harmonic_only", "hpss_protect_percussive",
    "magnitude_inpaint", "deq_inpaint", "total_att_cap_db", "nuclear_mode",
    "ms_process", "ms_side_scale",
    "phase_blur", "pb_start_hz", "pb_end_hz", "pb_harmonic_only",
    "swish_repair", "swish_start_hz", "swish_end_hz",
    "swish_time_amt", "swish_freq_amt",
    "hf_decorrelate", "hf_dec_start_hz", "hf_dec_end_hz",
    "hf_resynth", "hf_lp_hz", "hf_src_lo_hz", "hf_src_hi_hz",
    "hf_drive", "hf_hp_hz", "hf_mix", "hf_confidence_blend",
    "master_enabled", "hp_hz", "target_lufs", "target_rms_dbfs",
    "norm_max_gain_db", "norm_max_atten_db", "ceiling_dbtp",
    "lim_lookahead_ms", "lim_release_ms", "tp_os",
)

SPEC_PARAM_NAMES: tuple[str, ...] = ("spec_n_fft", "spec_hop", "spec_max_frames", "spec_max_hz")
PARAM_STATE_NAMES: tuple[str, ...] = PARAM_KNOB_NAMES + SPEC_PARAM_NAMES

MASTER_TO_UI_NAME = {
    "enabled": "master_enabled",
    "lookahead_ms": "lim_lookahead_ms",
    "release_ms": "lim_release_ms",
    "os_factor": "tp_os",
}
UI_TO_MASTER_NAME = {v: k for k, v in MASTER_TO_UI_NAME.items()}

UI_DEFAULT_OVERRIDES: dict[str, object] = {
    "spec_n_fft": 2048,
    "spec_hop": 512,
    "spec_max_frames": 1200,
    "spec_max_hz": 20000.0,
}


def _coerce_like(value: object, default: object) -> object:
    """Coerce value to the same primitive type as default."""
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(round(float(value)))  # type: ignore
    if isinstance(default, float) or default is None:
        return float(value)  # type: ignore
    return value


def _default_param_state() -> dict[str, object]:
    """Build default flat dictionary of UI parameter states."""
    p = Params()
    mp = MasterParams()
    state: dict[str, object] = {}
    p_values = asdict(p)
    mp_values = asdict(mp)
    for name in PARAM_KNOB_NAMES:
        if name in p_values:
            state[name] = p_values[name]
            continue
        master_name = UI_TO_MASTER_NAME.get(name, name)
        if master_name in mp_values:
            value = mp_values[master_name]
            if name == "target_lufs" and value is None:
                value = 999.0
            state[name] = value
    state.update(UI_DEFAULT_OVERRIDES)
    return state


def _merge_param_state(
    state: Mapping[str, object] | None,
    updates: Mapping[str, object],
) -> dict[str, object]:
    """Merge partial parameter updates into current or default state."""
    merged = _default_param_state()
    if isinstance(state, Mapping):
        for name in PARAM_STATE_NAMES:
            if name in state:
                merged[name] = state[name]
    for name, value in updates.items():
        if name in PARAM_STATE_NAMES:
            merged[name] = value
    return merged


def _state_from_dataclasses(
    p: Params,
    mp: MasterParams,
    dp: DebugParams | None = None,
) -> dict[str, object]:
    """Construct flat UI state dict from Params, MasterParams, and DebugParams dataclasses."""
    state = _default_param_state()
    p_values = asdict(p)
    mp_values = asdict(mp)
    for name in PARAM_KNOB_NAMES:
        if name in p_values:
            state[name] = p_values[name]
            continue
        master_name = UI_TO_MASTER_NAME.get(name, name)
        if master_name in mp_values:
            value = mp_values[master_name]
            if name == "target_lufs" and value is None:
                value = 999.0
            state[name] = value
    if dp is not None:
        dp_values = asdict(dp)
        for name in SPEC_PARAM_NAMES:
            field_name = name.removeprefix("spec_")
            if name in dp_values:
                state[name] = dp_values[name]
            elif field_name in dp_values:
                state[name] = dp_values[field_name]
    return state


def _state_to_component_values(state: Mapping[str, object] | None) -> list[object]:
    """Return ordered list of component values matching PARAM_STATE_NAMES."""
    merged = _merge_param_state(state, {})
    return [merged[name] for name in PARAM_STATE_NAMES]


def _build_params(
    param_state: Mapping[str, object] | None,
) -> tuple[Params, MasterParams, DebugParams]:
    """Convert flat UI parameter state into typed Params, MasterParams, and DebugParams."""
    state = _merge_param_state(param_state, {})

    p = Params()
    p_updates: dict[str, object] = {}
    for f in fields(Params):
        if f.name not in state:
            continue
        p_updates[f.name] = _coerce_like(state[f.name], getattr(p, f.name))
    p_updates.update({"pad": True, "fade_ms": 5.0, "seed": 0})
    p = replace(p, **p_updates)

    mp = MasterParams()
    mp_updates: dict[str, object] = {}
    for f in fields(MasterParams):
        ui_name = MASTER_TO_UI_NAME.get(f.name, f.name)
        if ui_name not in state:
            continue
        if f.name == "target_lufs":
            raw_lufs = float(state[ui_name])  # type: ignore
            mp_updates[f.name] = None if raw_lufs >= 998.0 else raw_lufs
        else:
            mp_updates[f.name] = _coerce_like(state[ui_name], getattr(mp, f.name))
    mp = replace(mp, **mp_updates)

    dp = DebugParams(
        enabled=False,
        spec_n_fft=int(state["spec_n_fft"]),  # type: ignore
        spec_hop=int(state["spec_hop"]),  # type: ignore
        spec_max_frames=int(state["spec_max_frames"]),  # type: ignore
        spec_max_hz=float(state["spec_max_hz"]),  # type: ignore
    )
    return p, mp, dp
