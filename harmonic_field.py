"""Robust families of partials with learned ratios and a shared pitch trajectory.

Integer ratios propose families; the fitted model uses independent, learned
log-frequency offsets. No equal-temperament grid or fixed tuning is a target.
Only intervals with simultaneous observations of every family member are used.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tonal_tracking import PartialTrack, TonalConfig


@dataclass
class HarmonicFamily:
    members: tuple[int, ...]
    first_frame: int
    observed: np.ndarray
    expected: np.ndarray
    power: np.ndarray
    tolerance: np.ndarray
    stable_members: int

    @property
    def disagreement(self):
        return 1200 * np.log2(self.observed / self.expected)


class HarmonicField:
    """Own inferred family relationships and their evidence, not instrument labels."""

    def __init__(self, tracks: list[PartialTrack], config: TonalConfig, sample_rate: int):
        self.tracks = sorted(tracks, key=lambda track: track.frames[0])
        self.config = config
        self.sample_rate = sample_rate
        self.families: list[HarmonicFamily] = []
        self.stable_neighbors: list[set[int]] = [set() for _ in tracks]
        self.ambiguous_members: set[int] = set()

    def connect(self):
        """A component with its own agreeing peers is not another voice's outlier."""
        for left, track in enumerate(self.tracks):
            for right in range(left + 1, len(self.tracks)):
                other = self.tracks[right]

                if other.frames[0] > track.frames[-1]:
                    break

                first = max(track.frames[0], other.frames[0])
                last = min(track.frames[-1], other.frames[-1])

                if last - first + 1 < self.config.min_track_frames:
                    continue

                first_values = np.asarray(track.frequencies)[first-track.frames[0]:last-track.frames[0]+1]
                second_values = np.asarray(other.frequencies)[first-other.frames[0]:last-other.frames[0]+1]
                ratio = 1200 * np.log2(first_values / second_values)

                if np.quantile(np.abs(ratio - np.median(ratio)), 0.95) <= self.config.consistency_cents:
                    self.stable_neighbors[left].add(right)
                    self.stable_neighbors[right].add(left)

    def fit(self):
        if len(self.tracks) < self.config.min_family_partials:
            return self

        self.connect()

        centers = np.array([np.median(track.frequencies) for track in self.tracks])
        first = np.array([track.frames[0] for track in self.tracks])
        last = np.array([track.frames[-1] for track in self.tracks])
        candidates: list[HarmonicFamily] = []
        seen = set()

        for anchor in range(len(self.tracks)):
            overlap = np.minimum(last, last[anchor]) - np.maximum(first, first[anchor]) + 1
            eligible = np.flatnonzero(overlap >= self.config.min_track_frames)

            for divisor in range(1, int(centers[anchor] / self.config.min_hz) + 1):
                fundamental = centers[anchor] / divisor
                order = np.rint(centers[eligible] / fundamental).astype(int)
                error = np.abs(1200 * np.log2(centers[eligible] / (np.maximum(order, 1) * fundamental)))
                selected = eligible[(order > 0) & (error <= self.config.family_tolerance_cents)]
                selected_orders = np.rint(centers[selected] / fundamental).astype(int)

                # Ambiguous/unresolved orders provide no evidence; other orders can.
                orders, counts = np.unique(selected_orders, return_counts=True)
                selected = selected[np.isin(selected_orders, orders[counts == 1])]

                members = tuple(int(index) for index in selected)

                if len(members) < self.config.min_family_partials or members in seen:
                    continue

                seen.add(members)
                family = self.fit_family(members)

                if family is not None:
                    candidates.append(family)

        self.mark_ambiguity(candidates)
        assigned: set[int] = set()
        candidates.sort(key=lambda family: (family.stable_members, np.sum(family.power)), reverse=True)

        for family in candidates:
            if assigned.isdisjoint(family.members):
                self.families.append(family)
                assigned.update(family.members)

        return self

    def mark_ambiguity(self, candidates: list[HarmonicFamily]):
        """A shared/overlapping partial cannot be assigned to competing stable cores."""
        support: dict[int, list[tuple[set[int], int, int]]] = {}

        for family in candidates:
            stable = np.quantile(np.abs(family.disagreement), 0.95, axis=1) <= family.tolerance
            core = {member for row, member in enumerate(family.members) if stable[row]}
            first = family.first_frame
            last = first + family.observed.shape[1] - 1

            for member in family.members:
                peers = core - {member}

                if len(peers) < self.config.min_family_partials - 1:
                    continue

                previous = support.setdefault(member, [])
                conflict = any(peers.isdisjoint(other) and min(last, stop) - max(first, start) + 1 >= self.config.min_track_frames
                               for other, start, stop in previous)

                if conflict:
                    self.ambiguous_members.add(member)

                previous.append((peers, first, last))

    def fit_family(self, members: tuple[int, ...]) -> HarmonicFamily | None:
        first = max(self.tracks[index].frames[0] for index in members)
        last = min(self.tracks[index].frames[-1] for index in members)

        if last - first + 1 < self.config.min_track_frames:
            return None

        observed, power, tolerance = [], [], []

        for index in members:
            track = self.tracks[index]
            selection = slice(first - track.frames[0], last - track.frames[0] + 1)
            observed.append(np.asarray(track.frequencies)[selection])
            power.append(np.sum(np.abs(np.asarray(track.coefficients)[selection]) ** 2, axis=1))
            tolerance.append(max(self.config.consistency_cents, track.uncertainty_cents(self.sample_rate, self.config.hop)))

        observed = np.asarray(observed)
        logarithm = np.log2(observed)
        offsets = np.median(logarithm, axis=1)

        # Alternating medians minimize absolute log-frequency disagreement.
        for _ in range(self.config.fit_iterations):
            motion = np.median(logarithm - offsets[:, None], axis=0)
            offsets = np.median(logarithm - motion[None, :], axis=1)

        expected = np.exp2(offsets[:, None] + motion[None, :])
        error = np.abs(1200 * np.log2(observed / expected))
        stable = np.quantile(error, 0.95, axis=1) <= np.asarray(tolerance)

        if np.mean(stable) < self.config.stable_fraction:
            return None

        for row, member in enumerate(members):
            if not stable[row] and len(self.stable_neighbors[member]) >= self.config.min_family_partials - 1:
                return None

        # Use the agreeing core as the trajectory reference, excluding outliers.
        motion = np.median(logarithm[stable] - offsets[stable, None], axis=0)
        expected = np.exp2(offsets[:, None] + motion[None, :])
        return HarmonicFamily(members, first, observed, expected, np.asarray(power),
                              np.asarray(tolerance), int(np.sum(stable)))

    def metrics(self) -> dict:
        tracked_power = sum(float(np.sum(np.abs(track.coefficients) ** 2)) for track in self.tracks)
        eligible = [(family, np.array([member not in self.ambiguous_members for member in family.members]))
                    for family in self.families]
        family_power = sum(float(np.sum(family.power[mask])) for family, mask in eligible)
        disagreement = np.concatenate([np.abs(family.disagreement[mask]).ravel() for family, mask in eligible]) if eligible else np.array([])
        fraction = min(1.0, family_power / tracked_power) if tracked_power > 0 else None
        return {
            "status": "supported" if self.families else "no_supported_families",
            "tracked_partials": len(self.tracks),
            "families": len(self.families),
            "ambiguous_partials": len(self.ambiguous_members),
            "family_partial_energy_fraction": fraction,
            "unattached_partial_energy_fraction": 1 - fraction if fraction is not None else None,
            "median_disagreement_cents": float(np.median(disagreement)) if disagreement.size else None,
            "p95_disagreement_cents": float(np.quantile(disagreement, 0.95)) if disagreement.size else None,
        }

    def compare(self, output: "HarmonicField") -> float:
        """Input-anchored improvement, penalizing loss of the original partials.

        Reuse input families/targets so destroying or relabeling a family cannot
        improve its score. Missing output partials incur full energy-loss cost.
        Unsupported input has no measurable correction opportunity and returns 0.
        """
        if not self.families:
            return 0.0

        input_centers = np.array([np.median(track.frequencies) for track in self.tracks])
        output_centers = np.array([np.median(track.frequencies) for track in output.tracks])
        assignments = {}

        if output_centers.size:
            cost = np.abs(1200 * np.log2(output_centers[None, :] / input_centers[:, None]))
            input_start = np.array([track.frames[0] for track in self.tracks])[:, None]
            input_stop = np.array([track.frames[-1] for track in self.tracks])[:, None]
            output_start = np.array([track.frames[0] for track in output.tracks])[None, :]
            output_stop = np.array([track.frames[-1] for track in output.tracks])[None, :]
            overlap = np.maximum(0, np.minimum(input_stop, output_stop) - np.maximum(input_start, output_start) + 1)
            shorter = np.minimum(input_stop - input_start + 1, output_stop - output_start + 1)
            assignment_cost = cost + self.config.family_tolerance_cents * (1 - overlap / shorter)
            rows, columns = linear_sum_assignment(assignment_cost)
            assignments = {int(row): int(column) for row, column in zip(rows, columns)
                           if cost[row, column] <= self.config.family_tolerance_cents
                           and overlap[row, column] >= self.config.min_track_frames}

        improvement, weight = 0.0, 0.0

        for family in self.families:
            input_error = np.abs(family.disagreement)
            scale = np.average(input_error, weights=family.power) + self.config.consistency_cents

            for row, member in enumerate(family.members):
                if member in self.ambiguous_members:
                    continue

                retained = np.zeros(family.observed.shape[1])
                output_error = input_error[row].copy()

                if member in assignments:
                    track = output.tracks[assignments[member]]
                    frames = np.arange(family.first_frame, family.first_frame + len(retained))
                    valid = (frames >= track.frames[0]) & (frames <= track.frames[-1])
                    positions = frames[valid] - track.frames[0]
                    frequencies = np.asarray(track.frequencies)[positions]
                    powers = np.sum(np.abs(np.asarray(track.coefficients)[positions]) ** 2, axis=1)
                    retained[valid] = np.minimum(1.0, powers / family.power[row, valid])
                    output_error[valid] = np.abs(1200 * np.log2(frequencies / family.expected[row, valid]))

                benefit = retained * (input_error[row] - output_error) / scale - (1 - retained)
                improvement += float(np.sum(family.power[row] * benefit))
                weight += float(np.sum(family.power[row]))

        return improvement / weight if weight > 0 else 0.0
