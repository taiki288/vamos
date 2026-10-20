"""Calibrate and verify the ground-projection camera model.

Three steps, each writing or checking a GroundCamera JSON file:

  intrinsics  checkerboard images -> fx, fy, cx, cy and lens distortion
  mount       ground markers (pixel <-> measured robot-frame metres) -> camera
              height, pitch, yaw and offsets
  verify      ground markers -> per-marker projection error; optionally marks
              the file "calibrated" when every error is within tolerance

Marker CSV columns: u,v,x_m,y_m. (u, v) are pixels in the inference image;
(x_m, y_m) are the measured marker position on flat ground in the robot frame
(x forward, y left, origin = ground point below the robot's reference point).
No robot, Zenoh session or model is needed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from projection import GroundCamera, GroundProjector

Point2D = tuple[float, float]
MAX_MODELLED_ROLL_DEGREES = 2.0


@dataclass(frozen=True)
class Marker:
    pixel: Point2D
    ground_m: Point2D


@dataclass(frozen=True)
class IntrinsicsResult:
    camera: GroundCamera
    rms_reprojection_px: float
    images_used: int


@dataclass(frozen=True)
class MountResult:
    camera: GroundCamera
    roll_degrees: float
    rms_reprojection_px: float


@dataclass(frozen=True)
class MarkerError:
    marker: Marker
    projected_m: Point2D | None
    status: str
    error_m: float | None


def load_markers(path: Path) -> list[Marker]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing = {"u", "v", "x_m", "y_m"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"marker CSV needs columns u,v,x_m,y_m (missing {', '.join(sorted(missing))})")
    markers = []
    for row in rows:
        values = [float(row[name]) for name in ("u", "v", "x_m", "y_m")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("marker values must be finite")
        markers.append(Marker((values[0], values[1]), (values[2], values[3])))
    return markers


def calibrate_intrinsics(object_points: Sequence, image_points: Sequence, image_size: tuple[int, int],
                         model: str, base: GroundCamera | None = None) -> IntrinsicsResult:
    """Run OpenCV calibration on matched checkerboard views of one image size."""
    import cv2
    import numpy as np

    if len(object_points) < 3 or len(object_points) != len(image_points):
        raise ValueError("at least three matched checkerboard views are required")
    width, height = image_size
    if model == "fisheye":
        objects = [np.asarray(points, dtype=np.float64).reshape(-1, 1, 3) for points in object_points]
        images = [np.asarray(points, dtype=np.float64).reshape(-1, 1, 2) for points in image_points]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, matrix, coefficients, _, _ = cv2.fisheye.calibrate(
            objects, images, (width, height), np.eye(3), np.zeros(4), flags=flags,
            criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-9))
    elif model == "opencv":
        objects = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in object_points]
        images = [np.asarray(points, dtype=np.float32).reshape(-1, 2) for points in image_points]
        rms, matrix, coefficients, _, _ = cv2.calibrateCamera(objects, images, (width, height), None, None)
    else:
        raise ValueError("model must be opencv or fisheye")
    base = base or GroundCamera(width, height, 1.0, 1.0, width / 2, height / 2, 0.5)
    if (base.image_width, base.image_height) != (width, height):
        raise ValueError("base camera dimensions do not match the checkerboard images")
    camera = replace(base, fx=float(matrix[0, 0]), fy=float(matrix[1, 1]), cx=float(matrix[0, 2]),
                     cy=float(matrix[1, 2]), distortion_model=model,
                     distortion_coefficients=tuple(float(value) for value in np.ravel(coefficients)),
                     calibration_status="assumed",
                     notes=f"Intrinsics from {len(object_points)} checkerboard views, RMS {rms:.3f} px. "
                           "Mounting not yet estimated/verified.")
    return IntrinsicsResult(camera, float(rms), len(object_points))


def find_checkerboards(paths: Sequence[Path], columns: int, rows: int, square_m: float):
    """Detect inner corners; returns object points, image points and image size."""
    import cv2
    import numpy as np

    grid = np.zeros((rows * columns, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    objects, images, size, skipped = [], [], None, []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped.append(f"{path.name}: unreadable")
            continue
        current = (gray.shape[1], gray.shape[0])
        if size is not None and current != size:
            raise ValueError(f"{path.name} is {current}, expected {size}; use one image size")
        size = current
        found, corners = cv2.findChessboardCorners(gray, (columns, rows))
        if not found:
            skipped.append(f"{path.name}: board not found")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objects.append(grid.copy())
        images.append(corners.reshape(-1, 2))
    if size is None:
        raise ValueError("no readable images")
    return objects, images, size, skipped


def estimate_mount(camera: GroundCamera, markers: Sequence[Marker]) -> MountResult:
    """Solve the camera pose from >= 4 non-collinear ground markers.

    Uses the camera's intrinsics/distortion as fixed. The projection model has
    no roll, so roll is reported separately and must stay small.
    """
    import cv2
    import numpy as np

    if len(markers) < 4:
        raise ValueError("at least four ground markers are required")
    projector = GroundProjector(camera)
    normalized = np.array([projector.normalized_ray(*marker.pixel) for marker in markers], dtype=np.float64)
    ground = np.array([[*marker.ground_m, 0.0] for marker in markers], dtype=np.float64)
    if np.linalg.matrix_rank(ground[:, :2] - ground[:, :2].mean(axis=0)) < 2:
        raise ValueError("ground markers must not be collinear")
    ok, rvec, tvec = cv2.solvePnP(ground, normalized.reshape(-1, 1, 2), np.eye(3), None,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("could not solve the camera pose from these markers")
    ok, rvec, tvec = cv2.solvePnP(ground, normalized.reshape(-1, 1, 2), np.eye(3), None, rvec, tvec,
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation, _ = cv2.Rodrigues(rvec)  # optical <- robot
    centre = (-rotation.T @ tvec).ravel()
    right, _, forward = rotation  # Rows: optical axes expressed in the robot frame.
    if centre[2] <= 0:
        raise ValueError("solved camera is below the ground plane; check marker axes")
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -forward[2]))))
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, -right[2]))))
    mounted = replace(camera, height_m=float(centre[2]), pitch_down_degrees=pitch, yaw_left_degrees=yaw,
                      forward_offset_m=float(centre[0]), left_offset_m=float(centre[1]),
                      calibration_status="assumed",
                      notes=f"{camera.notes} Mount solved from {len(markers)} ground markers.".strip())
    reprojected = GroundProjector(mounted).ground_to_pixel([marker.ground_m for marker in markers])
    residuals = [math.dist(marker.pixel, pixel) for marker, pixel in zip(markers, reprojected) if pixel]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals)) if residuals else math.inf
    return MountResult(mounted, roll, rms)


def verify_markers(camera: GroundCamera, markers: Sequence[Marker]) -> list[MarkerError]:
    projector = GroundProjector(camera)
    errors = []
    for marker in markers:
        point = projector.project_pixel(marker.pixel)
        distance = None if point.point_robot_m is None else math.dist(point.point_robot_m, marker.ground_m)
        errors.append(MarkerError(marker, point.point_robot_m, point.status, distance))
    return errors


def _write(camera: GroundCamera, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(camera.as_json(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    intrinsics = commands.add_parser("intrinsics", help="checkerboard images -> intrinsics and distortion")
    intrinsics.add_argument("--images", type=Path, required=True, help="directory of checkerboard images")
    intrinsics.add_argument("--board", required=True, help="inner corners as COLUMNSxROWS, e.g. 9x6")
    intrinsics.add_argument("--square-m", type=float, required=True, help="checker square size in metres")
    intrinsics.add_argument("--model", choices=("opencv", "fisheye"), default="opencv")
    intrinsics.add_argument("--base", type=Path, help="existing camera JSON whose mount values are kept")
    intrinsics.add_argument("--output", type=Path, required=True)

    mount = commands.add_parser("mount", help="ground markers -> height, pitch, yaw and offsets")
    mount.add_argument("--camera", type=Path, required=True)
    mount.add_argument("--markers", type=Path, required=True)
    mount.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", help="ground markers -> projection error report")
    verify.add_argument("--camera", type=Path, required=True)
    verify.add_argument("--markers", type=Path, required=True)
    verify.add_argument("--tolerance-m", type=float, default=0.10)
    verify.add_argument("--mark-calibrated", type=Path, metavar="OUTPUT",
                        help="write a copy marked calibrated if every marker is within tolerance")
    args = parser.parse_args(argv)

    try:
        if args.command == "intrinsics":
            columns, rows = (int(value) for value in args.board.lower().split("x"))
            if columns < 2 or rows < 2 or not math.isfinite(args.square_m) or args.square_m <= 0:
                raise ValueError("board needs >= 2x2 inner corners and a positive square size")
            paths = sorted(path for path in args.images.iterdir()
                           if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            objects, images, size, skipped = find_checkerboards(paths, columns, rows, args.square_m)
            for reason in skipped:
                print(f"skipped {reason}")
            base = GroundCamera.load(args.base) if args.base else None
            result = calibrate_intrinsics(objects, images, size, args.model, base)
            print(f"{result.images_used} views, {size[0]}x{size[1]}, RMS reprojection {result.rms_reprojection_px:.3f} px")
            if result.rms_reprojection_px > 1.0:
                print("warning: RMS above 1 px; retake views covering the whole image at varied angles")
            _write(result.camera, args.output)
        elif args.command == "mount":
            result = estimate_mount(GroundCamera.load(args.camera), load_markers(args.markers))
            c = result.camera
            print(f"height={c.height_m:.3f} m pitch_down={c.pitch_down_degrees:.2f} deg "
                  f"yaw_left={c.yaw_left_degrees:.2f} deg forward={c.forward_offset_m:.3f} m "
                  f"left={c.left_offset_m:.3f} m roll={result.roll_degrees:.2f} deg "
                  f"RMS={result.rms_reprojection_px:.2f} px")
            if abs(result.roll_degrees) > MAX_MODELLED_ROLL_DEGREES:
                print(f"warning: roll exceeds {MAX_MODELLED_ROLL_DEGREES} deg; the projection model assumes "
                      "zero roll, so lateral ground positions will be skewed")
            _write(c, args.output)
        else:
            camera = GroundCamera.load(args.camera)
            errors = verify_markers(camera, load_markers(args.markers))
            worst = 0.0
            for item in errors:
                projected = "-" if item.projected_m is None else f"({item.projected_m[0]:.3f}, {item.projected_m[1]:.3f})"
                error = "-" if item.error_m is None else f"{item.error_m:.3f}"
                print(f"pixel=({item.marker.pixel[0]:.1f}, {item.marker.pixel[1]:.1f}) "
                      f"measured=({item.marker.ground_m[0]:.3f}, {item.marker.ground_m[1]:.3f}) "
                      f"projected={projected} error_m={error} {item.status}")
                worst = math.inf if item.error_m is None else max(worst, item.error_m)
            passed = bool(errors) and worst <= args.tolerance_m
            print(f"worst error {worst:.3f} m, tolerance {args.tolerance_m:.3f} m: {'PASS' if passed else 'FAIL'}")
            if args.mark_calibrated:
                if not passed:
                    print("not marking calibrated")
                    return 1
                _write(replace(camera, calibration_status="calibrated",
                               notes=f"{camera.notes} Verified on {len(errors)} ground markers, "
                                     f"worst error {worst:.3f} m.".strip()), args.mark_calibrated)
            return 0 if passed else 1
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
