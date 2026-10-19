"""Pure Pursuit for ordered metric paths, independent of Zenoh and VAMOS.

Path points and Pose2D use the same fixed world frame (metres/radians).
Commands use the robot frame: +x forward, +y left, +yaw counter-clockwise.
This controller handles simple, non-self-intersecting paths. It does not perform
obstacle avoidance, pixel projection, freshness checks, or robot communication.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from numbers import Real
from typing import Literal, Sequence


Point2D = tuple[float, float]
_EPS = 1e-9


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("expected a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite number")
    return result


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # Radians, counter-clockwise from world +x.


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_distance_m: float = 0.5
    forward_speed_mps: float = 0.2
    max_yaw_rate_rps: float = 0.8
    goal_tolerance_m: float = 0.1
    slowdown_distance_m: float = 0.5
    rotate_threshold_rad: float = math.pi / 4
    rotate_gain: float = 1.5

    def __post_init__(self):
        for field in fields(self):
            value = _number(getattr(self, field.name))
            if value <= 0:
                raise ValueError(f"{field.name} must be positive")
            object.__setattr__(self, field.name, value)
        if self.forward_speed_mps > 3.8 or self.max_yaw_rate_rps > 4.0:
            raise ValueError("velocity limits exceed the Go2 command contract")
        if self.rotate_threshold_rad >= math.pi / 2:
            raise ValueError("rotate_threshold_rad must be less than pi/2")
        if self.slowdown_distance_m <= self.goal_tolerance_m:
            raise ValueError("slowdown_distance_m must exceed goal_tolerance_m")


@dataclass(frozen=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0

    def as_message(self) -> dict[str, str | float]:
        """Go2-compatible JSON data; calling this never sends a command."""
        return {"type": "velocity", "vx": self.vx, "vy": self.vy, "vyaw": self.vyaw}


TrackingStatus = Literal["tracking", "rotating", "reached", "no_path", "invalid_path", "invalid_pose"]


@dataclass(frozen=True)
class TrackingResult:
    command: VelocityCommand
    status: TrackingStatus
    target_world: Point2D | None = None
    target_robot: Point2D | None = None
    distance_to_end_m: float | None = None
    remaining_distance_m: float | None = None
    cross_track_error_m: float | None = None


def _metric_points(path: Sequence[Sequence[float]]) -> tuple[Point2D, ...]:
    points = []
    for point in path:
        if len(point) != 2:
            raise ValueError("path points must have exactly two coordinates")
        parsed = (_number(point[0]), _number(point[1]))
        if not points or parsed != points[-1]:
            points.append(parsed)
    return tuple(points)


def _path_target(points: tuple[Point2D, ...], pose: Pose2D, lookahead: float):
    """Project onto the closest segment, then advance along the polyline."""
    if len(points) == 1:
        return points[0], 0.0, math.hypot(points[0][0] - pose.x, points[0][1] - pose.y)
    lengths, cumulative = [], [0.0]
    best_distance, best_progress = math.inf, 0.0
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if not math.isfinite(length) or length <= 0:
            raise ValueError("invalid metric segment")
        ux, uy = dx / length, dy / length
        projection = (pose.x - start[0]) * ux + (pose.y - start[1]) * uy
        if not math.isfinite(projection):
            raise ValueError("invalid path projection")
        along = min(length, max(0.0, projection))
        nearest = start[0] + along * ux, start[1] + along * uy
        distance = math.hypot(pose.x - nearest[0], pose.y - nearest[1])
        if distance < best_distance:
            best_distance, best_progress = distance, cumulative[-1] + along
        lengths.append(length)
        cumulative.append(cumulative[-1] + length)
    if not math.isfinite(cumulative[-1]) or not math.isfinite(best_distance):
        raise ValueError("invalid total path length")
    target_progress = min(cumulative[-1], best_progress + lookahead)
    for index, length in enumerate(lengths):
        if target_progress <= cumulative[index + 1]:
            fraction = (target_progress - cumulative[index]) / length
            start, end = points[index], points[index + 1]
            target = (start[0] + fraction * (end[0] - start[0]),
                      start[1] + fraction * (end[1] - start[1]))
            return target, max(0.0, cumulative[-1] - best_progress), best_distance
    return points[-1], 0.0, best_distance


class PurePursuitController:
    def __init__(self, config: PurePursuitConfig | None = None):
        self.config = config or PurePursuitConfig()

    def compute(self, path: Sequence[Sequence[float]], pose: Pose2D) -> TrackingResult:
        """Compute one command from a world-frame path and current world pose.

        Recompute on each control tick using the new pose. A path endpoint is a
        local segment endpoint, not evidence that a VLA navigation task is done.
        Invalid runtime observations produce zero velocity, not an exception.
        """
        zero = VelocityCommand()
        try:
            pose = Pose2D(_number(pose.x), _number(pose.y), _number(pose.yaw))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return TrackingResult(zero, "invalid_pose")
        try:
            points = _metric_points(path)
        except (TypeError, ValueError, OverflowError):
            return TrackingResult(zero, "invalid_path")
        if not points:
            return TrackingResult(zero, "no_path")

        try:
            target, remaining, cross_track = _path_target(points, pose, self.config.lookahead_distance_m)
            distance_to_end = math.hypot(points[-1][0] - pose.x, points[-1][1] - pose.y)
            dx, dy = target[0] - pose.x, target[1] - pose.y
            cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
            x, y = cosine * dx + sine * dy, -sine * dx + cosine * dy
            if not all(math.isfinite(v) for v in (*target, remaining, cross_track, distance_to_end, x, y)):
                raise ValueError("invalid target geometry")
        except (ValueError, OverflowError):
            return TrackingResult(zero, "invalid_path")

        details = dict(target_world=target, target_robot=(x, y), distance_to_end_m=distance_to_end,
                       remaining_distance_m=remaining, cross_track_error_m=cross_track)
        if distance_to_end <= self.config.goal_tolerance_m and remaining <= self.config.goal_tolerance_m:
            return TrackingResult(zero, "reached", **details)
        radius = math.hypot(x, y)
        if radius <= _EPS:
            return TrackingResult(zero, "invalid_path", **details)
        heading_error = math.atan2(y, x)
        if x <= 0 or abs(heading_error) > self.config.rotate_threshold_rad:
            yaw_rate = max(-self.config.max_yaw_rate_rps,
                           min(self.config.max_yaw_rate_rps, self.config.rotate_gain * heading_error))
            return TrackingResult(VelocityCommand(vyaw=yaw_rate), "rotating", **details)

        # Slow down near the end, including when recovering from overshoot or
        # lateral displacement. Slow forward motion to preserve curvature when
        # the turn-rate limit would otherwise be exceeded.
        approach_distance = remaining + cross_track
        speed = self.config.forward_speed_mps * min(1.0, approach_distance / self.config.slowdown_distance_m)
        curvature = 2.0 * (y / radius) / radius
        if abs(curvature) > _EPS:
            speed = min(speed, self.config.max_yaw_rate_rps / abs(curvature))
        return TrackingResult(VelocityCommand(vx=speed, vyaw=speed * curvature), "tracking", **details)


def main():
    """Print sample commands only; no network connection or robot is used."""
    controller = PurePursuitController()
    cases = (
        ("straight", [(0, 0), (2, 0)], Pose2D(0, 0, 0)),
        ("left", [(0, 0), (2, 1)], Pose2D(0, 0, 0)),
        ("right", [(0, 0), (2, -1)], Pose2D(0, 0, 0)),
        ("behind", [(0, 0), (-2, 0)], Pose2D(0, 0, 0)),
        ("end", [(0, 0), (2, 0)], Pose2D(2, 0, 0)),
    )
    for name, path, pose in cases:
        result = controller.compute(path, pose)
        print(json.dumps({"case": name, "status": result.status, "target_robot": result.target_robot,
                          "command": result.command.as_message()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
