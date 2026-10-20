"""Camera pixels -> flat ground in the image-capture robot frame.

Uses the same ray/plane intersection as Projector.pixels_to_ground in the
original trajectory_projection package, with explicit axes and invalid-ray
handling. Pixels from a distorted lens are first undistorted with the OpenCV
pinhole ("opencv") or equidistant ("fisheye") model. No depth, ROS, Zenoh,
model or robot connection is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Literal, Sequence

from controller import Pose2D


Point2D = tuple[float, float]
ProjectionStatus = Literal["ok", "invalid_pixel", "outside_image", "not_ground_facing",
                           "behind_robot", "out_of_range"]
DistortionModel = Literal["none", "opencv", "fisheye"]
# cv2.calibrateCamera returns 4, 5, 8, 12 or 14 coefficients; fisheye uses k1..k4.
_OPENCV_COEFFICIENT_COUNTS = (4, 5, 8, 12, 14)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class GroundCamera:
    """Image dimensions must match the image used for inference.

    Optical frame: right/down/forward. Robot: forward/left/up. Zero pitch and
    yaw mean a level, forward-facing camera with no roll. Positive pitch tilts
    down; positive yaw turns left. Offsets and height locate the optical centre
    relative to the ground-projected robot origin. Ground is z=0.
    The distortion model/coefficients describe the inference image itself;
    "none" means it is already rectified.
    """

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float
    pitch_down_degrees: float = 0.0
    yaw_left_degrees: float = 0.0
    forward_offset_m: float = 0.0
    left_offset_m: float = 0.0
    max_distance_m: float = 8.0
    max_segment_length_m: float = 3.0
    distortion_model: DistortionModel = "none"
    distortion_coefficients: tuple[float, ...] = ()
    calibration_status: Literal["assumed", "calibrated"] = "assumed"
    notes: str = ""

    def __post_init__(self):
        for name in ("image_width", "image_height"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fx", "fy", "cx", "cy", "height_m", "pitch_down_degrees",
                     "yaw_left_degrees", "forward_offset_m", "left_offset_m",
                     "max_distance_m", "max_segment_length_m"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if any(getattr(self, name) <= 0 for name in
               ("fx", "fy", "height_m", "max_distance_m", "max_segment_length_m")):
            raise ValueError("focal lengths, height and distance limits must be positive")
        if not (0 <= self.cx < self.image_width and 0 <= self.cy < self.image_height):
            raise ValueError("principal point must be inside the source image")
        if not -89 < self.pitch_down_degrees < 89 or not -180 <= self.yaw_left_degrees <= 180:
            raise ValueError("pitch must be in (-89, 89), yaw in [-180, 180] degrees")
        if self.calibration_status not in ("assumed", "calibrated") or not isinstance(self.notes, str):
            raise ValueError("invalid calibration metadata")
        coefficients = tuple(_finite(value, "distortion coefficient") for value in self.distortion_coefficients)
        object.__setattr__(self, "distortion_coefficients", coefficients)
        expected = {"none": (0,), "opencv": _OPENCV_COEFFICIENT_COUNTS, "fisheye": (4,)}.get(self.distortion_model)
        if expected is None:
            raise ValueError("distortion_model must be none, opencv or fisheye")
        if len(coefficients) not in expected:
            raise ValueError(f"{self.distortion_model} distortion needs {' or '.join(map(str, expected))} coefficients")

    @property
    def matrix(self):
        import numpy as np

        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)

    def check_image_size(self, width: int, height: int):
        if (width, height) != (self.image_width, self.image_height):
            raise ValueError("camera calibration dimensions do not match the source image; "
                             "resize/crop intrinsics explicitly rather than silently rescaling")

    @classmethod
    def load(cls, path: Path) -> GroundCamera:
        values = json.loads(path.read_text(encoding="utf-8"))
        if "distortion_coefficients" in values:
            values["distortion_coefficients"] = tuple(values["distortion_coefficients"])
        return cls(**values)

    def as_json(self) -> dict:
        return {**asdict(self), "distortion_coefficients": list(self.distortion_coefficients)}


@dataclass(frozen=True)
class ProjectedPoint:
    status: ProjectionStatus
    point_robot_m: Point2D | None = None


@dataclass(frozen=True)
class ProjectedPath:
    candidate_index: int
    points_robot_m: tuple[Point2D, ...]
    point_statuses: tuple[str, ...]
    retained_source_indices: tuple[int, ...]
    stop_reason: str | None

    @property
    def usable(self) -> bool:
        return len(self.points_robot_m) >= 2

    @property
    def truncated(self) -> bool:
        return self.stop_reason is not None

    def as_dict(self) -> dict:
        return {**asdict(self), "usable": self.usable, "truncated": self.truncated}


class GroundProjector:
    def __init__(self, camera: GroundCamera):
        self.camera = camera
        pitch = math.radians(camera.pitch_down_degrees)
        yaw = math.radians(camera.yaw_left_degrees)
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)
        self.cy, self.sy = math.cos(yaw), math.sin(yaw)

    def normalized_ray(self, u: float, v: float) -> Point2D:
        """K^-1 [u, v, 1] after removing lens distortion."""
        c = self.camera
        if c.distortion_model == "none":
            return (u - c.cx) / c.fx, (v - c.cy) / c.fy
        import cv2
        import numpy as np

        source = np.array([[[u, v]]], dtype=float)
        coefficients = np.array(c.distortion_coefficients, dtype=float)
        if c.distortion_model == "fisheye":
            result = cv2.fisheye.undistortPoints(source, c.matrix, coefficients)
        else:
            result = cv2.undistortPoints(source, c.matrix, coefficients)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    def rotation_optical_from_robot(self):
        """Rows are the optical right/down/forward axes in the robot FLU frame."""
        import numpy as np

        cp, sp, cy, sy = self.cp, self.sp, self.cy, self.sy
        return np.array([[sy, -cy, 0.0], [-sp * cy, -sp * sy, -cp], [cp * cy, cp * sy, -sp]])

    def ground_to_pixel(self, points_robot_m: Sequence[Point2D]) -> list[Point2D | None]:
        """Inverse of project_pixel for calibration checks; None when not visible."""
        import cv2
        import numpy as np

        c = self.camera
        centre = np.array([c.forward_offset_m, c.left_offset_m, c.height_m])
        rotation = self.rotation_optical_from_robot()
        result = []
        for x, y in points_robot_m:
            optical = rotation @ (np.array([_finite(x, "x"), _finite(y, "y"), 0.0]) - centre)
            if optical[2] <= 1e-9:
                result.append(None)
                continue
            source = optical.reshape(1, 1, 3)
            zero = np.zeros(3)
            coefficients = np.array(c.distortion_coefficients, dtype=float)
            if c.distortion_model == "fisheye":
                pixels, _ = cv2.fisheye.projectPoints(source, zero, zero, c.matrix, coefficients)
            else:
                pixels, _ = cv2.projectPoints(source, zero, zero, c.matrix,
                                              coefficients if c.distortion_model == "opencv" else None)
            u, v = (float(value) for value in pixels.reshape(2))
            result.append((u, v) if 0 <= u < c.image_width and 0 <= v < c.image_height else None)
        return result

    def project_pixel(self, pixel: Sequence[float]) -> ProjectedPoint:
        try:
            if len(pixel) != 2:
                raise ValueError("expected u, v")
            u, v = _finite(pixel[0], "u"), _finite(pixel[1], "v")
        except (TypeError, ValueError, OverflowError):
            return ProjectedPoint("invalid_pixel")
        c = self.camera
        if not (0 <= u < c.image_width and 0 <= v < c.image_height):
            return ProjectedPoint("outside_image")
        # K^-1 [u,v,1] (undistorted), then optical(right/down/forward) -> robot FLU.
        rx, ry = self.normalized_ray(u, v)
        dx, dy, dz = self.cp - self.sp * ry, -rx, -self.sp - self.cp * ry
        dx, dy = self.cy * dx - self.sy * dy, self.sy * dx + self.cy * dy
        norm = math.hypot(dx, dy, dz)
        if not math.isfinite(norm) or norm == 0 or dz / norm >= -1e-6:
            return ProjectedPoint("not_ground_facing")
        scale = -c.height_m / dz
        x, y = c.forward_offset_m + scale * dx, c.left_offset_m + scale * dy
        if not math.isfinite(x) or not math.isfinite(y):
            return ProjectedPoint("out_of_range")
        if x <= 0:
            return ProjectedPoint("behind_robot")
        if math.hypot(x, y) > c.max_distance_m:
            return ProjectedPoint("out_of_range")
        return ProjectedPoint("ok", (x, y))

    def project_path(self, pixels: Sequence[Sequence[float]], candidate_index: int = 0) -> ProjectedPath:
        """Keep only the consecutive valid prefix in the original model order.

        Never reconnect points across an invalid ray, sort points, or reverse
        their order. Oversized segments also end the prefix. An empty/one-point
        prefix cannot be selected for simulation. This is geometry validation,
        not obstacle detection or a navigation-safety assessment.
        """
        projected = [self.project_pixel(pixel) for pixel in pixels]
        points, indices = [], []
        stop_reason = None
        for index, result in enumerate(projected):
            if result.status != "ok":
                stop_reason = f"point {index}: {result.status}"
                break
            point = result.point_robot_m
            if points and math.dist(points[-1], point) > self.camera.max_segment_length_m:
                stop_reason = f"point {index}: segment_too_long"
                break
            if not points or math.dist(points[-1], point) > 1e-9:
                points.append(point)
                indices.append(index)
        if not projected:
            stop_reason = "empty_path"
        return ProjectedPath(candidate_index, tuple(points), tuple(p.status for p in projected),
                             tuple(indices), stop_reason)

    def project_paths(self, paths: Sequence[Sequence[Sequence[float]]]) -> list[ProjectedPath]:
        return [self.project_path(path, index) for index, path in enumerate(paths)]


def path_to_world(points_robot_m: Sequence[Point2D], capture_pose: Pose2D) -> tuple[Point2D, ...]:
    """Anchor a frozen image's local ground path at its image-capture pose.

    Do not re-anchor the path to the moving robot at each control tick. Real
    streaming navigation will require timestamped image-capture/current poses.
    """
    x = _finite(capture_pose.x, "capture_pose.x")
    y = _finite(capture_pose.y, "capture_pose.y")
    yaw = _finite(capture_pose.yaw, "capture_pose.yaw")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    result = []
    for point in points_robot_m:
        px, py = _finite(point[0], "point.x"), _finite(point[1], "point.y")
        wx, wy = x + cosine * px - sine * py, y + sine * px + cosine * py
        if not math.isfinite(wx) or not math.isfinite(wy):
            raise ValueError("world transform overflow")
        result.append((wx, wy))
    return tuple(result)
