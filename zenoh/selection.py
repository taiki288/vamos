"""Choose one VAMOS candidate path after flat-ground projection.

The original VAMOS ranks candidates with an affordance/value map built from a
height map, which a Go2 with only a front camera does not provide. Instead:

  index   use one fixed candidate index (the earlier demo behaviour)
  medoid  among geometrically usable candidates, use the one with the smallest
          mean distance to the others: the consensus of the sampled paths,
          which discards outliers without claiming obstacle awareness

Either way, an unusable selection is reported, never silently replaced, and
the decision is explainable from the returned per-candidate report.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

from projection import GroundProjector, ProjectedPath

Point2D = tuple[float, float]
Strategy = Literal["index", "medoid"]
RESAMPLE_POINTS = 16


@dataclass(frozen=True)
class CandidateReport:
    index: int
    usable: bool
    length_m: float
    truncated: bool
    mean_distance_m: float | None
    rejected: str | None

    def as_dict(self) -> dict:
        return {"index": self.index, "usable": self.usable, "length_m": self.length_m,
                "truncated": self.truncated, "mean_distance_m": self.mean_distance_m, "rejected": self.rejected}


@dataclass(frozen=True)
class Selection:
    strategy: Strategy
    index: int | None
    path: ProjectedPath | None
    reason: str
    candidates: tuple[CandidateReport, ...]

    @property
    def usable(self) -> bool:
        return self.path is not None


def path_length(points: Sequence[Point2D]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def resample(points: Sequence[Point2D], count: int = RESAMPLE_POINTS) -> list[Point2D]:
    """Evenly spaced points by arc length, including both endpoints."""
    total = path_length(points)
    if total <= 0:
        return [points[0]] * count
    result, segment, walked = [], 0, 0.0
    for step in range(count):
        target = total * step / (count - 1)
        while segment < len(points) - 2 and walked + math.dist(points[segment], points[segment + 1]) < target:
            walked += math.dist(points[segment], points[segment + 1])
            segment += 1
        start, end = points[segment], points[segment + 1]
        length = math.dist(start, end)
        ratio = 0.0 if length == 0 else min(1.0, max(0.0, (target - walked) / length))
        result.append((start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio))
    return result


def path_distance(a: Sequence[Point2D], b: Sequence[Point2D]) -> float:
    return sum(math.dist(p, q) for p, q in zip(resample(a), resample(b))) / RESAMPLE_POINTS


def select_candidate(trajectories: Sequence[Sequence[Sequence[float]]], projector: GroundProjector, *,
                     strategy: Strategy = "medoid", index: int = 0, min_length_m: float = 0.05) -> Selection:
    if strategy not in ("index", "medoid"):
        raise ValueError("strategy must be index or medoid")
    if type(index) is not int or index < 0:
        raise ValueError("candidate index must be a nonnegative integer")
    if not math.isfinite(min_length_m) or min_length_m < 0:
        raise ValueError("min_length_m must be finite and nonnegative")
    projected = projector.project_paths(trajectories)
    rejected = {}
    for path in projected:
        if not path.usable:
            rejected[path.candidate_index] = path.stop_reason or "fewer than two distinct points"
        elif path_length(path.points_robot_m) < min_length_m:
            rejected[path.candidate_index] = "shorter_than_minimum"
    usable = [path for path in projected if path.candidate_index not in rejected]
    mean_distance = {}
    if strategy == "medoid" and len(usable) > 1:
        for path in usable:
            others = [path_distance(path.points_robot_m, other.points_robot_m)
                      for other in usable if other is not path]
            mean_distance[path.candidate_index] = sum(others) / len(others)
    reports = tuple(CandidateReport(path.candidate_index, path.candidate_index not in rejected,
                                    path_length(path.points_robot_m), path.truncated,
                                    mean_distance.get(path.candidate_index), rejected.get(path.candidate_index))
                    for path in projected)

    if strategy == "index":
        if index >= len(projected):
            return Selection(strategy, None, None, f"candidate {index} is missing", reports)
        if index in rejected:
            return Selection(strategy, None, None, f"candidate {index} unusable: {rejected[index]}", reports)
        return Selection(strategy, index, projected[index], f"fixed candidate {index}", reports)
    if not usable:
        return Selection(strategy, None, None, "no usable candidate", reports)
    if len(usable) == 1:
        chosen = usable[0]
        return Selection(strategy, chosen.candidate_index, chosen, "only usable candidate", reports)
    chosen = min(usable, key=lambda path: (mean_distance[path.candidate_index], path.candidate_index))
    return Selection(strategy, chosen.candidate_index, chosen,
                     f"medoid of {len(usable)} usable candidates "
                     f"(mean distance {mean_distance[chosen.candidate_index]:.3f} m)", reports)
