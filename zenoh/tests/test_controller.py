from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller import Pose2D, PurePursuitConfig, PurePursuitController, VelocityCommand


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = PurePursuitController()

    def test_straight_path_produces_forward_motion(self):
        result = self.controller.compute([(0, 0), (2, 0)], Pose2D(0, 0, 0))
        self.assertEqual(result.status, "tracking")
        self.assertAlmostEqual(result.command.vx, 0.2)
        self.assertEqual(result.command.vy, 0.0)
        self.assertAlmostEqual(result.command.vyaw, 0.0)

    def test_left_and_right_paths_produce_mirrored_turns(self):
        left = self.controller.compute([(0, 0), (2, 1)], Pose2D(0, 0, 0))
        right = self.controller.compute([(0, 0), (2, -1)], Pose2D(0, 0, 0))
        self.assertGreater(left.command.vyaw, 0)
        self.assertLess(right.command.vyaw, 0)
        self.assertAlmostEqual(left.command.vyaw, -right.command.vyaw)
        self.assertAlmostEqual(left.command.vx, right.command.vx)

    def test_world_rotation_and_translation_do_not_change_body_command(self):
        path = [(0, 0), (1, 0), (2, 1)]
        original = self.controller.compute(path, Pose2D(0.1, -0.1, 0.2))
        angle = 1.1

        def transform(x, y):
            return (10 + math.cos(angle) * x - math.sin(angle) * y,
                    -4 + math.sin(angle) * x + math.cos(angle) * y)

        px, py = transform(0.1, -0.1)
        rotated = self.controller.compute([transform(x, y) for x, y in path], Pose2D(px, py, 0.2 + angle))
        self.assertEqual(rotated.status, original.status)
        self.assertAlmostEqual(rotated.command.vx, original.command.vx)
        self.assertAlmostEqual(rotated.command.vyaw, original.command.vyaw)

    def test_behind_or_sideways_target_rotates_without_translation(self):
        for target in ((-2, 0), (0, 2), (0, -2)):
            with self.subTest(target=target):
                result = self.controller.compute([(0, 0), target], Pose2D(0, 0, 0))
                self.assertEqual(result.status, "rotating")
                self.assertEqual(result.command.vx, 0.0)
                self.assertEqual(result.command.vy, 0.0)
                self.assertGreater(abs(result.command.vyaw), 0)
                self.assertLessEqual(abs(result.command.vyaw), 0.8)

    def test_lookahead_is_interpolated_along_sparse_metric_path(self):
        result = self.controller.compute([(0, 0), (10, 0)], Pose2D(1, 0, 0))
        self.assertEqual(result.target_world, (1.5, 0.0))
        self.assertEqual(result.target_robot, (0.5, 0.0))

    def test_passed_points_are_pruned_geometrically(self):
        result = self.controller.compute([(0, 0), (1, 0), (2, 0), (3, 0)], Pose2D(1.6, 0, 0))
        self.assertAlmostEqual(result.target_world[0], 2.1)
        self.assertGreater(result.command.vx, 0)
        self.assertAlmostEqual(result.command.vyaw, 0)

    def test_turn_limit_reduces_speed_without_distorting_curvature(self):
        config = PurePursuitConfig(max_yaw_rate_rps=0.1)
        result = PurePursuitController(config).compute([(0, 0), (2, 1)], Pose2D(0, 0, 0))
        self.assertGreater(result.command.vx, 0)
        self.assertLess(result.command.vx, config.forward_speed_mps)
        self.assertAlmostEqual(result.command.vyaw, config.max_yaw_rate_rps)

    def test_decelerates_near_endpoint_and_stops_within_tolerance(self):
        path = [(0, 0), (2, 0)]
        far = self.controller.compute(path, Pose2D(1, 0, 0))
        near = self.controller.compute(path, Pose2D(1.8, 0, 0))
        end = self.controller.compute(path, Pose2D(1.97, 0.02, 0))
        self.assertGreater(near.command.vx, 0)
        self.assertLess(near.command.vx, far.command.vx)
        self.assertEqual(end.status, "reached")
        self.assertEqual(end.command, VelocityCommand())

    def test_duplicate_points_and_single_point_path(self):
        duplicated = self.controller.compute([(0, 0), (0, 0), (2, 0), (2, 0)], Pose2D(0, 0, 0))
        single = self.controller.compute([(2, 0)], Pose2D(0, 0, 0))
        self.assertGreater(duplicated.command.vx, 0)
        self.assertGreater(single.command.vx, 0)
        self.assertEqual(self.controller.compute([(2, 0)], Pose2D(2, 0, 0)).status, "reached")

    def test_empty_and_invalid_paths_stop(self):
        for path in ([], None, [(0, 0), (float("nan"), 1)], [(float("inf"), 0)],
                     [(1, 2, 3)], [(True, 1)], [("1", 2)], [(1e308, 0), (-1e308, 0)]):
            with self.subTest(path=path):
                result = self.controller.compute(path, Pose2D(0, 0, 0))
                self.assertIn(result.status, ("no_path", "invalid_path"))
                self.assertEqual(result.command, VelocityCommand())

    def test_invalid_pose_stops(self):
        for pose in (None, Pose2D(float("nan"), 0, 0), Pose2D(0, 0, float("inf")), Pose2D(True, 0, 0)):
            with self.subTest(pose=pose):
                result = self.controller.compute([(0, 0), (2, 0)], pose)
                self.assertEqual(result.status, "invalid_pose")
                self.assertEqual(result.command, VelocityCommand())

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"forward_speed_mps": 0}, {"max_yaw_rate_rps": 4.1},
                       {"forward_speed_mps": 4}, {"lookahead_distance_m": float("nan")},
                       {"rotate_threshold_rad": math.pi / 2}, {"rotate_gain": True},
                       {"slowdown_distance_m": 0.05}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PurePursuitConfig(**kwargs)

    def test_command_serializes_as_go2_velocity_message(self):
        command = self.controller.compute([(0, 0), (2, 0)], Pose2D(0, 0, 0)).command
        decoded = json.loads(json.dumps(command.as_message(), allow_nan=False))
        self.assertEqual(decoded, {"type": "velocity", "vx": 0.2, "vy": 0.0, "vyaw": 0.0})

    def test_closed_loop_kinematics_reaches_straight_and_curved_endpoints(self):
        # Test repeated control against planar motion, not only isolated formulas.
        cases = (
            ([(0, 0), (2, 0)], Pose2D(0, 0.3, 0)),
            ([(0, 0), (1, 0), (2, 0.5), (2.5, 1)], Pose2D(0, 0, 0)),
            ([(0, 0), (-2, 0)], Pose2D(0, 0, 0)),
            ([(0, 0), (2, 0)], Pose2D(2.4, 0, 0)),
        )
        for path, initial in cases:
            with self.subTest(path=path, initial=initial):
                pose = initial
                for _ in range(3000):
                    result = self.controller.compute(path, pose)
                    self.assertIn(result.status, ("tracking", "rotating", "reached"))
                    command = result.command
                    self.assertGreaterEqual(command.vx, 0)
                    self.assertLessEqual(command.vx, 0.2)
                    self.assertLessEqual(abs(command.vyaw), 0.8 + 1e-12)
                    if result.status == "reached":
                        break
                    dt = 0.05
                    pose = Pose2D(pose.x + command.vx * math.cos(pose.yaw) * dt,
                                  pose.y + command.vx * math.sin(pose.yaw) * dt,
                                  pose.yaw + command.vyaw * dt)
                else:
                    self.fail(f"Controller did not converge from {initial}")
                self.assertLessEqual(math.hypot(pose.x - path[-1][0], pose.y - path[-1][1]), 0.1)
                self.assertEqual(result.command, VelocityCommand())


if __name__ == "__main__":
    unittest.main()
