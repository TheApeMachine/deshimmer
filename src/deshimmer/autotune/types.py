"""Data structures, weights, failure logs, and trial types for auto-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Tuple
import numpy as np

OBJECTIVE_NAMES = (
    "artifact_reduction",
    "music_preservation",
    "diff_preservation",
    "stereo_preservation",
    "loudness_tilt_preservation",
    "tonal_coherence",
)

MATH_FAILURES = (ValueError, FloatingPointError, RuntimeWarning, ArithmeticError, np.linalg.LinAlgError)


@dataclass
class Region:
    t0: float
    dur: float
    label: str = "auto"


@dataclass
class AnalysisReport:
    detected_band: tuple[float, float]
    band_db_diff: float = 0.0
    spectral_entropy: float = 0.0
    detected_whines: list[float] = field(default_factory=list)
    measured_noise_floor_db: float = -60.0
    notes: list[str] = field(default_factory=list)
    tonal_regions: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["**Auto analysis report**", ""]
        b0, b1 = self.detected_band
        lines.append(f"- Shimmer band detected: **{b0:.0f} – {b1:.0f} Hz**")
        lines.append(f"- Estimated noise floor: **{self.measured_noise_floor_db:+.1f} dBFS**")
        if self.detected_whines:
            whines_s = ", ".join(f"{w:.0f} Hz" for w in self.detected_whines[:6])
            lines.append(f"- Identified stationary whines: **{len(self.detected_whines)}** ({whines_s})")
        else:
            lines.append("- Identified stationary whines: none detected")
        for n in self.notes:
            lines.append(f"- {n}")

        for region in self.tonal_regions:
            disagreement = region["p95_disagreement_cents"]
            measured = f"{disagreement:.2f} cents p95 disagreement" if disagreement is not None else "insufficient family evidence"
            lines.append(f"- Tonal region **{region['label']}**: {region['families']} families, {measured}.")

        return "\n".join(lines)


@dataclass
class Weights:
    band_reduction: float
    swish_reduction: float
    out_of_band_change: float
    musical_noise: float
    transient_leak: float
    harmonic_leak: float
    loudness_loss: float
    band_energy_loss: float
    spectral_tilt: float
    centroid_shift: float
    diff_onset_corr: float
    stereo_width: float
    tonal_coherence: float = 1.0

    def objectives(self, metrics: dict[str, float]) -> tuple[float, ...]:
        """Partition weighted utility into six objectives, counting each term once."""
        return (
            self.band_reduction * metrics["band_reduction_db"]
            + self.swish_reduction * metrics["swish_reduction"],
            -(self.out_of_band_change * metrics["out_of_band_change_db"]
              + self.musical_noise * metrics["musical_noise"]
              + self.band_energy_loss * metrics["band_energy_loss_db"]
              + self.centroid_shift * metrics["centroid_shift"]),
            -(self.transient_leak * metrics["transient_leak"]
              + self.harmonic_leak * metrics["harmonic_leak"]
              + self.diff_onset_corr * metrics["diff_onset_corr"]),
            -self.stereo_width * metrics["stereo_width_delta"],
            -(self.loudness_loss * metrics["loudness_loss_db"]
              + self.spectral_tilt * metrics["spectral_tilt_delta"]),
            self.tonal_coherence * metrics["tonal_coherence_gain"],
        )


class EvaluationFailureLog:
    def __init__(
        self,
        debug_dir: str,
        *,
        planned_trials_per_stage: int,
        max_consecutive: int = 5,
        max_failure_rate: float = 0.20,
    ):
        self.debug_dir = debug_dir
        self.path = os.path.join(debug_dir, "failed_trials.json")
        self.planned_trials_per_stage = max(1, planned_trials_per_stage)
        self.max_consecutive = max(1, max_consecutive)
        self.max_failure_rate = float(max(0.0, max_failure_rate))
        self.consecutive = 0
        self.total_failures = 0
        self.stage_attempts: dict[str, int] = {}
        self.stage_failures: dict[str, int] = {}
        self._events: list[dict[str, Any]] = []

    def begin_trial(self, stage: str) -> None:
        self.stage_attempts[stage] = self.stage_attempts.get(stage, 0) + 1

    def success(self, stage: str) -> None:
        self.consecutive = 0

    def record(
        self,
        *,
        stage: str,
        trial_number: int | None,
        region: Region,
        params: Any,
        exc: BaseException,
    ) -> bool:
        self.consecutive += 1
        self.total_failures += 1
        self.stage_failures[stage] = self.stage_failures.get(stage, 0) + 1
        event = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage": stage,
            "trial_number": trial_number,
            "region": asdict(region),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "params": asdict(params) if hasattr(params, "__dataclass_fields__") else str(params),
        }
        self._events.append(event)
        os.makedirs(self.debug_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2, sort_keys=True)

        print(
            f"[auto_tune] stage={stage} trial={trial_number} failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        allowed_stage_failures = math.floor(self.max_failure_rate * self.planned_trials_per_stage)
        return self.consecutive >= self.max_consecutive or (
            self.stage_failures.get(stage, 0) > max(1, allowed_stage_failures)
        )


class _SnapshotTrial:
    def __init__(self, params: dict[str, Any]):
        self._p = params

    def suggest_float(self, name: str, lo: float, hi: float, **_: Any) -> float:
        return float(self._p[name])

    def suggest_int(self, name: str, lo: int, hi: int, **_: Any) -> int:
        return int(self._p[name])

    def suggest_categorical(self, name: str, choices: list[Any], **_: Any) -> Any:
        return self._p[name]
