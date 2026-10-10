"""Staged multi-objective optimization using Optuna NSGA-II."""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Any, Callable, Optional
import warnings

import numpy as np

from ..api import preview_context_seconds, process_audio, slice_with_context
from ..core.measurements import _as_2d
from ..core.params import DebugParams, MasterParams, Params
from .analyzer import pick_regions
from .scoring import (
    build_synthetic_transition,
    score_objectives,
    score_transition_boundary,
    weights_from_aggressiveness,
)
from .types import (
    MATH_FAILURES,
    OBJECTIVE_NAMES,
    EvaluationFailureLog,
    Region,
    Weights,
    _SnapshotTrial,
)


def _evaluate_params_multi(
    x: np.ndarray,
    sr: int,
    regions: list[Region],
    p: Params,
    weights: Weights,
    band: tuple[float, float],
    refine_dur: float,
    *,
    failure_log: EvaluationFailureLog | None = None,
    stage: str = "refine",
    trial_number: int | None = None,
    transition_aware: bool = False,
    variance_weight: float = 0.15,
) -> tuple[float, ...]:
    mp = MasterParams(enabled=False)
    dp = DebugParams(enabled=False)
    p_use = replace(p, delta_listen=False)
    context_s = preview_context_seconds(p_use)
    objs_list: list[tuple[float, ...]] = []
    seg_rms_in: list[float] = []
    seg_rms_out: list[float] = []
    count = 0
    for r in regions:
        inset = max(0.0, (r.dur - refine_dur) * 0.5)
        seg_t0 = r.t0 + inset
        seg_dur = min(r.dur, refine_dur)
        if seg_dur <= 0.0:
            raise ValueError(f"region '{r.label}' has no audio to score")
        x_ctx, target_s0, target_s1, ctx_s0 = slice_with_context(
            x, sr, seg_t0, seg_dur, context_s=context_s, params=p_use,
        )
        if target_s1 - target_s0 < max(2048, int(0.5 * sr)):
            raise ValueError(f"region '{r.label}' requires at least 0.5 seconds and 2048 samples")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
                    y_ctx, _info = process_audio(x_ctx, sr, params=p_use, master_params=mp, debug_params=dp)
                    trim0 = target_s0 - ctx_s0
                    trim1 = target_s1 - ctx_s0
                    y = np.asarray(y_ctx)[trim0:trim1]
                    seg = x[target_s0:target_s1]

                    if seg.ndim == 2:
                        y = _as_2d(y)

                    objectives = score_objectives(
                        seg, y, sr, band=band,
                        swish_band=(p_use.swish_start_hz, p_use.swish_end_hz),
                        weights=weights,
                    )

                    if not np.all(np.isfinite(objectives)):
                        raise ValueError("scoring produced non-finite objectives")

                    seg_rms_in.append(float(np.sqrt(np.mean(seg ** 2)) + 1e-12))
                    seg_rms_out.append(float(np.sqrt(np.mean(y ** 2)) + 1e-12))
        except MATH_FAILURES as exc:
            if failure_log is not None:
                fatal = failure_log.record(
                    stage=stage,
                    trial_number=trial_number,
                    region=r,
                    params=p_use,
                    exc=exc,
                )
                if fatal:
                    raise RuntimeError(
                        "Optimization failed due to persistent mathematical instability. "
                        f"See {failure_log.path}"
                    ) from exc
                import optuna
                raise optuna.TrialPruned(f"invalid DSP evaluation: {exc}") from exc

            raise

        objs_list.append(objectives)
        count += 1

    if count == 0:
        raise ValueError("evaluation requires at least one scorable region")
    if failure_log is not None:
        failure_log.success(stage)

    mean = np.mean(objs_list, axis=0)

    if transition_aware and len(objs_list) > 1:
        var_objs = np.var(objs_list, axis=0)
        penalized = mean - variance_weight * var_objs

        # Section-wise musical contrast preservation:
        # Penalize over-homogenizing dynamic contrast across sections
        if len(seg_rms_in) > 1 and len(seg_rms_out) > 1:
            in_spread = float(np.std(seg_rms_in) / (np.mean(seg_rms_in) + 1e-6))
            out_spread = float(np.std(seg_rms_out) / (np.mean(seg_rms_out) + 1e-6))
            contrast_loss = max(0.0, in_spread - out_spread)
            penalized[1] -= 0.4 * contrast_loss
            penalized[4] -= 0.4 * contrast_loss

        trans_data = build_synthetic_transition(x, sr, regions, transition_dur=min(1.0, refine_dur * 0.5))
        if trans_data is not None:
            x_trans, seam_idx = trans_data
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_trans, _ = process_audio(x_trans, sr, params=p_use, master_params=mp, debug_params=dp)
                    b_metrics = score_transition_boundary(x_trans, y_trans, sr, seam_idx)
                    boundary_penalty = (
                        0.5 * b_metrics["boundary_step_error"]
                        + 0.3 * b_metrics["boundary_pumping"]
                        + 0.2 * b_metrics["boundary_stereo_delta"]
                    )
                    # Apply boundary penalty to loudness_tilt_preservation (index 4) and music_preservation (index 1)
                    penalized[4] -= 0.5 * boundary_penalty
                    penalized[1] -= 0.5 * boundary_penalty
            except Exception:
                pass

        return tuple(float(value) for value in penalized)

    return tuple(float(value) for value in mean)


def _evaluate_params_single(
    x: np.ndarray,
    sr: int,
    regions: list[Region],
    p: Params,
    weights: Weights,
    band: tuple[float, float],
    refine_dur: float,
) -> float:
    obj = _evaluate_params_multi(x, sr, regions, p, weights, band, refine_dur)
    # Each metric contributes to exactly one weighted objective.
    return float(sum(obj))


def _make_stage(name: str, p: Params, trial: Any) -> Params:
    """Search-space configuration for Optuna."""
    if name == "tonal":
        return replace(p, tonal_repair=float(trial.suggest_float("tonal_repair", 0.0, 1.0)))

    if name == "shimmer":
        return replace(
            p,
            thr_db=float(trial.suggest_float("thr_db", 4.0, 14.0)),
            slope=float(trial.suggest_float("slope", 0.3, 1.2)),
            density_lo=float(trial.suggest_float("density_lo", 0.005, 0.06)),
            density_hi=float(trial.suggest_float("density_hi", 0.08, 0.30)),
            noise_resynth=float(trial.suggest_float("noise_resynth", 0.0, 0.40)),
        )
    if name == "denoise":
        return replace(
            p,
            denoise=float(trial.suggest_float("denoise", 0.0, 0.85)),
            dn_floor_db=float(trial.suggest_float("dn_floor_db", -36.0, -12.0)),
            dn_freq_smooth_bins=int(trial.suggest_int("dn_freq_smooth_bins", 1, 9, step=2)),
            dn_release_ms=float(trial.suggest_float("dn_release_ms", 60.0, 300.0)),
            dn_attack_ms=float(trial.suggest_float("dn_attack_ms", 2.0, 20.0)),
            dn_minwin_ms=float(trial.suggest_float("dn_minwin_ms", 200.0, 1000.0)),
        )
    if name == "deres":
        return replace(
            p,
            deres=float(trial.suggest_float("deres", 0.0, 0.9)),
            deq_thr_db=float(trial.suggest_float("deq_thr_db", 3.0, 10.0)),
            deq_slope=float(trial.suggest_float("deq_slope", 0.4, 1.2)),
            deq_persist_ms=float(trial.suggest_float("deq_persist_ms", 300.0, 1500.0)),
            deq_max_att_db=float(trial.suggest_float("deq_max_att_db", 4.0, 14.0)),
            deq_time_floor=bool(trial.suggest_categorical("deq_time_floor", [True, False])),
        )
    if name == "swish":
        return replace(
            p,
            swish_repair=float(trial.suggest_float("swish_repair", 0.0, 0.75)),
            swish_time_amt=float(trial.suggest_float("swish_time_amt", 0.15, 0.85)),
            swish_freq_amt=float(trial.suggest_float("swish_freq_amt", 0.0, 0.55)),
            hf_decorrelate=float(trial.suggest_float("hf_decorrelate", 0.0, 0.55)),
            ms_process=bool(trial.suggest_categorical("ms_process", [True, False])),
            ms_side_scale=float(trial.suggest_float("ms_side_scale", 0.15, 0.65)),
        )
    if name == "tonal_balance":
        return replace(
            p,
            tonal_balance=float(trial.suggest_float("tonal_balance", 0.0, 0.75)),
            tb_low_db=float(trial.suggest_float("tb_low_db", -2.5, 2.5)),
            tb_low_mid_db=float(trial.suggest_float("tb_low_mid_db", -2.5, 2.5)),
            tb_mid_db=float(trial.suggest_float("tb_mid_db", -2.0, 2.0)),
            tb_presence_db=float(trial.suggest_float("tb_presence_db", -2.5, 2.5)),
            tb_air_db=float(trial.suggest_float("tb_air_db", -2.0, 2.5)),
        )
    if name == "transient_punch":
        return replace(
            p,
            transient_punch=float(trial.suggest_float("transient_punch", 0.0, 0.65)),
            punch_attack_boost_db=float(trial.suggest_float("punch_attack_boost_db", 1.0, 3.5)),
            punch_low=float(trial.suggest_float("punch_low", 0.2, 0.8)),
            punch_high=float(trial.suggest_float("punch_high", 0.2, 0.8)),
        )
    if name == "low_end":
        return replace(
            p,
            bass_mono=bool(trial.suggest_categorical("bass_mono", [True, False])),
            bass_mono_hz=float(trial.suggest_float("bass_mono_hz", 80.0, 160.0)),
            bass_tighten=float(trial.suggest_float("bass_tighten", 0.0, 0.45)),
        )
    if name == "stereo_enhance":
        return replace(
            p,
            stereo_width=float(trial.suggest_float("stereo_width", 0.9, 1.35)),
            stereo_safe_mono=bool(trial.suggest_categorical("stereo_safe_mono", [True])),
        )
    if name == "harmonic":
        return replace(
            p,
            harmonic_enhance=float(trial.suggest_float("harmonic_enhance", 0.0, 0.4)),
            harmonic_start_hz=float(trial.suggest_float("harmonic_start_hz", 3000.0, 6000.0)),
        )
    raise ValueError(f"unknown stage: {name}")


def _pick_pareto_trial(study: Any, mode: str = "balanced") -> Any:
    """Select weighted utility on the study's maximizing Pareto front."""
    front_trials = study.best_trials

    if not front_trials:
        raise RuntimeError("no completed, valid trials")

    if not all(np.all(np.isfinite(trial.values)) for trial in front_trials):
        raise ValueError("Pareto selection requires finite objectives")

    if mode == "safe":
        return max(front_trials, key=lambda trial: sum(trial.values[1:5]))

    if mode == "aggressive":
        return max(front_trials, key=lambda trial: trial.values[0] + sum(trial.values[5:]))

    if mode != "balanced":
        raise ValueError(f"unknown Pareto selection mode: {mode}")

    return max(front_trials, key=lambda trial: sum(trial.values))


def refine(
    x: np.ndarray,
    sr: int,
    *,
    base_params: Params,
    regions: Optional[list[Region]] = None,
    aggressiveness: float = 0.5,
    n_trials_per_stage: int = 12,
    refine_dur: float = 3.0,
    debug_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    transition_aware: bool = True,
) -> tuple[Params, dict[str, Any]]:
    """Refine base parameters using multi-objective Bayesian optimization across preview regions."""
    try:
        import optuna
        from optuna.samplers import NSGAIISampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise RuntimeError(
            "optuna is required for refine(). Install with: pip install optuna"
        ) from e

    xm = np.asarray(x)
    if regions is None or len(regions) == 0:
        regions = pick_regions(xm, sr, n=5, dur=max(refine_dur + 0.5, 5.0))

    if n_trials_per_stage < 1 or refine_dur <= 0.0:
        raise ValueError("refinement requires positive trial count and duration")

    selection_weights = weights_from_aggressiveness(aggressiveness)
    objective_weights = selection_weights
    band = (float(base_params.start_hz), float(base_params.end_hz))
    failure_log = EvaluationFailureLog(
        debug_dir or os.path.join(os.getcwd(), "auto_tune_debug"),
        planned_trials_per_stage=n_trials_per_stage,
        max_consecutive=5,
        max_failure_rate=0.20,
    )

    # Refining from bypass must return the wet mix that was actually evaluated.
    p_cur = replace(base_params, mix=1.0 if base_params.mix == 0.0 else base_params.mix)
    summary: dict[str, Any] = {
        "aggressiveness": float(aggressiveness),
        "transition_aware": transition_aware,
        "regions": [{"t0": r.t0, "dur": r.dur, "label": r.label} for r in regions],
        "objective_weights": objective_weights.__dict__,
        "selection_weights": selection_weights.__dict__,
        "objective_names": list(OBJECTIVE_NAMES),
        "failure_log": failure_log.path,
        "stages": [],
    }

    stages: list[str] = ["shimmer"]
    if p_cur.denoise > 1e-3 or aggressiveness >= 0.4:
        stages.append("denoise")
    if p_cur.deres > 1e-3:
        stages.append("deres")
    if (
        float(getattr(p_cur, "swish_repair", 0.0)) > 1e-3
        or float(getattr(p_cur, "hf_decorrelate", 0.0)) > 1e-3
        or aggressiveness >= 0.35
    ):
        stages.append("swish")

    stages.append("tonal")

    if float(getattr(p_cur, "tonal_balance", 0.0)) > 1e-3:
        stages.append("tonal_balance")

    if float(getattr(p_cur, "transient_punch", 0.0)) > 1e-3:
        stages.append("transient_punch")

    if (
        bool(getattr(p_cur, "bass_mono", False))
        or float(getattr(p_cur, "bass_tighten", 0.0)) > 1e-3
    ) and xm.ndim == 2 and xm.shape[1] >= 2:
        stages.append("low_end")

    if (
        abs(float(getattr(p_cur, "stereo_width", 1.0)) - 1.0) > 1e-3
    ) and xm.ndim == 2 and xm.shape[1] >= 2:
        stages.append("stereo_enhance")

    if float(getattr(p_cur, "harmonic_enhance", 0.0)) > 1e-3:
        stages.append("harmonic")

    total_trials = max(1, n_trials_per_stage * len(stages))
    done = 0

    def _emit(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(float(np.clip(frac, 0.0, 1.0)), msg)

    _emit(0.0, "Starting refine")

    for stage in stages:
        sampler = NSGAIISampler(seed=42)
        study = optuna.create_study(
            directions=["maximize"] * len(OBJECTIVE_NAMES),
            sampler=sampler,
        )

        # Retain current settings when proposed changes score worse
        baseline = _evaluate_params_multi(
            xm, sr, regions, p_cur, objective_weights, band, refine_dur,
            transition_aware=transition_aware,
        )
        study.add_trial(optuna.trial.create_trial(values=baseline))

        def _objective(trial: Any, _stage: str = stage, _p: Params = p_cur) -> tuple[float, ...]:
            p_try = _make_stage(_stage, _p, trial)
            return _evaluate_params_multi(
                xm,
                sr,
                regions,
                p_try,
                objective_weights,
                band,
                refine_dur,
                failure_log=failure_log,
                stage=_stage,
                trial_number=int(getattr(trial, "number", -1)),
                transition_aware=transition_aware,
            )

        trial_scores: list[float] = []
        for _ in range(n_trials_per_stage):
            failure_log.begin_trial(stage)
            study.optimize(_objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)
            best_bal = _pick_pareto_trial(study, mode="balanced")
            trial_scores.append(float(sum(best_bal.values)))
            done += 1
            _emit(done / total_trials, f"Stage '{stage}': trial {done}/{total_trials}, score={trial_scores[-1]:.3f}")

        best_trial = _pick_pareto_trial(study, mode="balanced")
        safe_trial = _pick_pareto_trial(study, mode="safe")
        agg_trial = _pick_pareto_trial(study, mode="aggressive")
        if best_trial.params:
            p_cur = _make_stage(stage, p_cur, _SnapshotTrial(best_trial.params))
        summary["stages"].append({
            "stage": stage,
            "n_trials": n_trials_per_stage,
            "selected_score": float(sum(best_trial.values)),
            "selected_params": dict(best_trial.params),
            "selection_mode": "weighted_utility",
            "baseline_score": float(sum(baseline)),
            "retained_baseline": not bool(best_trial.params),
            "best_score": float(sum(best_trial.values)),
            "best_params": dict(best_trial.params),
            "trial_scores": trial_scores,
            "pareto": {
                "balanced": {"values": list(best_trial.values or []), "params": dict(best_trial.params)},
                "safe": {"values": list(safe_trial.values or []), "params": dict(safe_trial.params)},
                "aggressive": {"values": list(agg_trial.values or []), "params": dict(agg_trial.params)},
            },
        })

    final_score = _evaluate_params_single(xm, sr, regions, p_cur, selection_weights, band, refine_dur)
    summary["final_score"] = float(final_score)
    _emit(1.0, f"Done. Final score={final_score:.3f}")
    return p_cur, summary
