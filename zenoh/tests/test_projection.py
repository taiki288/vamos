from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller import Pose2D, PurePursuitController
from projection import GroundCamera, GroundProjector, path_to_world


def camera(**kwargs):
    return replace(GroundCamera(320, 240, 160, 160, 160, 120, 0.5), **kwargs)


class GroundProjectionTests(unittest.TestCase):
    def test_level_camera_has_known_distance_and_left_right_axes(self):
        projector = GroundProjector(camera())
        # v-cy=80 => forward distance height*fy/80=1 m.
        self.assertEqual(projector.project_pixel((160, 200)).point_robot_m, (1.0, 0.0))
        self.assertEqual(projector.project_pixel((120, 200)).point_robot_m, (1.0, 0.25))
        self.assertEqual(projector.project_pixel((200, 200)).point_robot_m, (1.0, -0.25))

    def test_pitch_down_puts_centre_ray_on_ground(self):
        point = GroundProjector(camera(pitch_down_degrees=30)).project_pixel((160, 120))
        self.assertEqual(point.status, "ok")
        self.assertAlmostEqual(point.point_robot_m[0], 0.5 / math.tan(math.pi / 6))

    def test_mount_yaw_offsets_and_height_scaling(self):
        point = GroundProjector(camera(yaw_left_degrees=30, forward_offset_m=0.1,
                                       left_offset_m=-0.2)).project_pixel((160, 200)).point_robot_m
        self.assertAlmostEqual(point[0], math.cos(math.pi / 6) + 0.1)
        self.assertAlmostEqual(point[1], 0.5 - 0.2)
        doubled = GroundProjector(camera(height_m=1)).project_pixel((120, 200)).point_robot_m
        self.assertEqual(doubled, (2.0, 0.5))

    def test_matches_original_vamos_matrix_projector_for_valid_rays(self):
        import numpy as np

        source = (Path(__file__).resolve().parents[2] /
                  "vamos_ws/src/trajectory_projection/trajectory_projection/projector.py")
        spec = importlib.util.spec_from_file_location("original_projector", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        c = camera(pitch_down_degrees=17, yaw_left_degrees=23,
                   forward_offset_m=0.12, left_offset_m=-0.03)
        p, y = math.radians(17), math.radians(23)
        pitch = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0],
                          [-math.sin(p), 0, math.cos(p)]])
        yaw = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        optical_to_level = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        transform = np.eye(4)
        transform[:3, :3] = yaw @ pitch @ optical_to_level
        transform[:3, 3] = [c.forward_offset_m, c.left_offset_m, c.height_m]
        pixels = [(110, 200), (160, 180), (200, 140), (180, 120)]
        original = module.Projector(K=[[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]])
        expected = original.pixels_to_ground(pixels, transform, z_ground=0)
        projector = GroundProjector(c)
        for pixel, xyz in zip(pixels, expected):
            np.testing.assert_allclose(projector.project_pixel(pixel).point_robot_m, xyz[:2], atol=1e-12)

    def test_horizon_above_horizon_distance_and_behind_robot_are_rejected(self):
        projector = GroundProjector(camera())
        self.assertEqual(projector.project_pixel((160, 120)).status, "not_ground_facing")
        self.assertEqual(projector.project_pixel((160, 119)).status, "not_ground_facing")
        self.assertEqual(projector.project_pixel((160, 120.001)).status, "out_of_range")
        backwards = GroundProjector(camera(yaw_left_degrees=180))
        self.assertEqual(backwards.project_pixel((160, 200)).status, "behind_robot")

    def test_malformed_nonfinite_and_outside_pixels(self):
        projector = GroundProjector(camera())
        for pixel in (None, (1,), (1, 2, 3), (True, 200), ("160", 200), (math.nan, 200), (160, math.inf)):
            with self.subTest(pixel=pixel):
                self.assertEqual(projector.project_pixel(pixel).status, "invalid_pixel")
        for pixel in ((-1, 200), (320, 200), (160, 240)):
            self.assertEqual(projector.project_pixel(pixel).status, "outside_image")

    def test_bad_middle_point_truncates_without_bridging(self):
        pixels = [(160, 220), (165, 200), (160, 100), (170, 180)]
        path = GroundProjector(camera()).project_path(pixels, candidate_index=4)
        self.assertTrue(path.usable)
        self.assertTrue(path.truncated)
        self.assertEqual(path.retained_source_indices, (0, 1))
        self.assertEqual(path.candidate_index, 4)
        self.assertEqual(path.point_statuses, ("ok", "ok", "not_ground_facing", "ok"))
        self.assertEqual(path.stop_reason, "point 2: not_ground_facing")
        self.assertEqual(len(path.points_robot_m), 2)

    def test_invalid_start_and_duplicate_only_paths_are_unusable(self):
        projector = GroundProjector(camera())
        for pixels in ([], [(160, 100), (160, 220), (160, 200)], [(160, 200)] * 3):
            with self.subTest(pixels=pixels):
                self.assertFalse(projector.project_path(pixels).usable)

    def test_large_gap_is_not_connected(self):
        projector = GroundProjector(camera(max_segment_length_m=0.5))
        path = projector.project_path([(160, 220), (160, 200), (160, 140)])
        self.assertTrue(path.usable)
        self.assertEqual(path.stop_reason, "point 2: segment_too_long")
        self.assertEqual(len(path.points_robot_m), 2)

    def test_world_transform_preserves_controller_body_command(self):
        path = GroundProjector(camera()).project_path([(160, 220), (180, 180), (190, 160)])
        local = PurePursuitController().compute(path.points_robot_m, Pose2D(0, 0, 0))
        capture = Pose2D(5, -3, 1.1)
        world = path_to_world(path.points_robot_m, capture)
        anchored = PurePursuitController().compute(world, capture)
        self.assertAlmostEqual(local.command.vx, anchored.command.vx)
        self.assertAlmostEqual(local.command.vyaw, anchored.command.vyaw)
        self.assertEqual(world, path_to_world(path.points_robot_m, capture))

    def test_invalid_camera_and_size_mismatch_fail_before_projection(self):
        for kwargs in ({"height_m": 0}, {"fx": 0}, {"fy": math.inf}, {"image_width": True},
                       {"cx": 320}, {"pitch_down_degrees": 90}, {"height_m": True},
                       {"max_distance_m": -1}, {"calibration_status": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                camera(**kwargs)
        with self.assertRaises(ValueError):
            camera().check_image_size(1280, 720)


class ProjectionDemoTests(unittest.TestCase):
    def test_saved_result_reports_selection_and_rejects_mismatched_size(self):
        from PIL import Image

        script = Path(__file__).resolve().parents[1] / "tools/projection_demo.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (320, 240), "gray").save(root / "image.png")
            (root / "camera.json").write_text(json.dumps(camera().as_json()), encoding="utf-8")
            result = {"inference_id": 1, "frame_sequence": 1, "image_width": 320, "image_height": 240,
                      "coordinate_frame": "image_pixels", "goal_pixels": [190, 160], "prompt": "test",
                      "trajectories": [[[160, 200], [180, 180], [190, 160]]], "generated_texts": ["test"],
                      "duration_seconds": 1, "source_age_seconds": 1, "device": "test"}
            (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
            command = [sys.executable, str(script), "--result", str(root / "result.json"),
                       "--image", str(root / "image.png"), "--camera", str(root / "camera.json"),
                       "--headless", "--output-dir", str(root / "output")]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "output/projection_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["selected_candidate"], 0)
            self.assertEqual(report["selection_strategy"], "medoid")
            self.assertEqual(report["camera"]["calibration_status"], "assumed")
            for name in ("projection.png", "source.png"):
                with Image.open(root / "output" / name) as image:
                    image.verify()
            # An unusable fixed choice is reported, never replaced by another path.
            result["trajectories"].append(result["trajectories"][0])
            result["trajectories"][0] = [[160, 100], [160, 200], [190, 160]]
            (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
            rejected = subprocess.run([*command, "--candidate", "0"], text=True, capture_output=True, timeout=30)
            self.assertEqual(rejected.returncode, 1, rejected.stderr)
            report = json.loads((root / "output/projection_report.json").read_text(encoding="utf-8"))
            self.assertIsNone(report["selected_candidate"])
            self.assertFalse(report["candidates"][0]["usable"])
            medoid = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(medoid.returncode, 0, medoid.stderr)
            Image.new("RGB", (640, 480)).save(root / "image.png")
            failed = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("dimensions do not match", failed.stderr)


if __name__ == "__main__":
    unittest.main()
