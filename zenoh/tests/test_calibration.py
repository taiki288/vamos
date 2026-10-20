from __future__ import annotations

from dataclasses import replace
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibrate_camera import Marker, calibrate_intrinsics, estimate_mount, find_checkerboards, main, verify_markers
from projection import GroundCamera, GroundProjector

OPENCV_DISTORTION = (-0.28, 0.09, 0.001, -0.0005, -0.012)
FISHEYE_DISTORTION = (0.05, -0.01, 0.002, -0.0003)
GROUND = [(1.0, 0.0), (1.5, 0.4), (2.0, -0.5), (2.5, 0.2), (3.0, -0.3), (1.2, -0.25), (1.8, 0.6)]


def camera(**kwargs):
    base = GroundCamera(1280, 720, 640.0, 650.0, 636.0, 362.0, 0.42, pitch_down_degrees=14.0,
                        yaw_left_degrees=3.0, forward_offset_m=0.28, left_offset_m=0.02)
    return replace(base, **kwargs)


def markers_for(source: GroundCamera, ground=GROUND):
    pixels = GroundProjector(source).ground_to_pixel(ground)
    return [Marker(pixel, point) for pixel, point in zip(pixels, ground)]


class DistortionTests(unittest.TestCase):
    def test_ground_to_pixel_inverts_projection_for_each_model(self):
        for model, coefficients in (("none", ()), ("opencv", OPENCV_DISTORTION), ("fisheye", FISHEYE_DISTORTION)):
            with self.subTest(model=model):
                projector = GroundProjector(camera(distortion_model=model, distortion_coefficients=coefficients))
                for pixel, point in zip(projector.ground_to_pixel(GROUND), GROUND):
                    self.assertIsNotNone(pixel)
                    projected = projector.project_pixel(pixel)
                    self.assertEqual(projected.status, "ok")
                    self.assertAlmostEqual(projected.point_robot_m[0], point[0], places=4)
                    self.assertAlmostEqual(projected.point_robot_m[1], point[1], places=4)

    def test_ignoring_distortion_moves_projected_points(self):
        distorted = camera(distortion_model="opencv", distortion_coefficients=OPENCV_DISTORTION)
        edge = GroundProjector(distorted).ground_to_pixel([(1.2, 0.6)])[0]
        naive = GroundProjector(camera()).project_pixel(edge).point_robot_m
        self.assertGreater(math.dist(naive, (1.2, 0.6)), 0.03)

    def test_coefficient_counts_are_validated_and_json_round_trips(self):
        with self.assertRaises(ValueError):
            camera(distortion_model="fisheye", distortion_coefficients=(0.1, 0.2))
        with self.assertRaises(ValueError):
            camera(distortion_coefficients=(0.1,))
        source = camera(distortion_model="opencv", distortion_coefficients=OPENCV_DISTORTION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            path.write_text(json.dumps(source.as_json()), encoding="utf-8")
            self.assertEqual(GroundCamera.load(path), source)

    def test_example_without_distortion_fields_still_loads(self):
        example = Path(__file__).resolve().parents[1] / "ground-camera.example.json"
        loaded = GroundCamera.load(example)
        self.assertEqual(loaded.distortion_model, "none")
        self.assertEqual(loaded.calibration_status, "assumed")


class CalibrationTests(unittest.TestCase):
    def test_intrinsics_are_recovered_from_synthetic_checkerboards(self):
        import cv2

        truth = camera(distortion_model="opencv", distortion_coefficients=OPENCV_DISTORTION)
        grid = np.zeros((9 * 6, 3), np.float32)
        grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.03
        grid -= grid.mean(axis=0)
        objects, images = [], []
        for rx, ry, tx, ty, z in ((0.3, 0.1, -0.05, 0.02, 0.45), (-0.25, 0.3, 0.06, -0.03, 0.5),
                                  (0.1, -0.35, 0.0, 0.05, 0.4), (0.4, 0.25, -0.08, -0.06, 0.55),
                                  (-0.2, -0.2, 0.09, 0.07, 0.6), (0.05, 0.45, 0.12, 0.0, 0.5),
                                  (-0.4, 0.05, -0.12, 0.04, 0.48)):
            pixels, _ = cv2.projectPoints(grid, np.array([rx, ry, 0.0]), np.array([tx, ty, z]),
                                          truth.matrix, np.array(OPENCV_DISTORTION))
            objects.append(grid)
            images.append(pixels.reshape(-1, 2).astype(np.float32))
        result = calibrate_intrinsics(objects, images, (1280, 720), "opencv", camera())
        self.assertLess(result.rms_reprojection_px, 0.01)
        self.assertAlmostEqual(result.camera.fx, truth.fx, delta=1.0)
        self.assertAlmostEqual(result.camera.cy, truth.cy, delta=1.0)
        self.assertEqual(result.camera.distortion_model, "opencv")
        self.assertEqual(result.camera.height_m, 0.42)  # Mount values come from the base file.
        self.assertEqual(result.camera.calibration_status, "assumed")

    def test_checkerboard_corners_are_detected_in_image_files(self):
        import cv2

        board = np.kron((np.indices((7, 10)).sum(axis=0) % 2) * 255, np.ones((40, 40))).astype(np.uint8)
        image = np.full((400, 520), 255, np.uint8)
        image[60:340, 60:460] = board
        with tempfile.TemporaryDirectory() as directory:
            good, blank = Path(directory) / "board.png", Path(directory) / "blank.png"
            cv2.imwrite(str(good), image)
            cv2.imwrite(str(blank), np.full((400, 520), 255, np.uint8))
            objects, images, size, skipped = find_checkerboards([good, blank], 9, 6, 0.025)
        self.assertEqual(size, (520, 400))
        self.assertEqual(len(objects), 1)
        self.assertEqual(images[0].shape, (54, 2))
        self.assertAlmostEqual(float(objects[0][:, 0].max()), 8 * 0.025, places=6)
        self.assertEqual(skipped, ["blank.png: board not found"])

    def test_mount_is_recovered_from_ground_markers(self):
        truth = camera(distortion_model="fisheye", distortion_coefficients=FISHEYE_DISTORTION)
        guess = replace(truth, height_m=0.5, pitch_down_degrees=0.0, yaw_left_degrees=0.0,
                        forward_offset_m=0.0, left_offset_m=0.0)
        result = estimate_mount(guess, markers_for(truth))
        mounted = result.camera
        self.assertAlmostEqual(mounted.height_m, truth.height_m, places=3)
        self.assertAlmostEqual(mounted.pitch_down_degrees, truth.pitch_down_degrees, places=2)
        self.assertAlmostEqual(mounted.yaw_left_degrees, truth.yaw_left_degrees, places=2)
        self.assertAlmostEqual(mounted.forward_offset_m, truth.forward_offset_m, places=3)
        self.assertAlmostEqual(mounted.left_offset_m, truth.left_offset_m, places=3)
        self.assertAlmostEqual(result.roll_degrees, 0.0, places=2)
        self.assertLess(result.rms_reprojection_px, 0.01)

    def test_mount_rejects_too_few_or_collinear_markers(self):
        truth = camera()
        with self.assertRaisesRegex(ValueError, "four"):
            estimate_mount(truth, markers_for(truth)[:3])
        with self.assertRaisesRegex(ValueError, "collinear"):
            estimate_mount(truth, markers_for(truth, [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0)]))

    def test_verify_marks_calibrated_only_within_tolerance(self):
        truth = camera()
        errors = verify_markers(truth, markers_for(truth))
        self.assertLess(max(item.error_m for item in errors), 1e-4)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            camera_path, markers_path = directory / "camera.json", directory / "markers.csv"
            camera_path.write_text(json.dumps(truth.as_json()), encoding="utf-8")
            rows = ["u,v,x_m,y_m"] + [f"{m.pixel[0]},{m.pixel[1]},{m.ground_m[0]},{m.ground_m[1]}"
                                      for m in markers_for(truth)]
            markers_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            output = directory / "calibrated.json"
            with redirect_stdout(io.StringIO()):
                code = main(["verify", "--camera", str(camera_path), "--markers", str(markers_path),
                             "--mark-calibrated", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(GroundCamera.load(output).calibration_status, "calibrated")

            shifted = rows[:-1] + [rows[-1].rsplit(",", 2)[0] + ",9.0,0.0"]
            markers_path.write_text("\n".join(shifted) + "\n", encoding="utf-8")
            output.unlink()
            with redirect_stdout(io.StringIO()):
                code = main(["verify", "--camera", str(camera_path), "--markers", str(markers_path),
                             "--mark-calibrated", str(output)])
            self.assertEqual(code, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
