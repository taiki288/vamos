from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projection import GroundCamera, GroundProjector
from selection import path_distance, resample, select_candidate

# Level camera, 0.5 m high: pixel (160, 200) is 1 m ahead, 40 px lateral is 0.25 m.
PROJECTOR = GroundProjector(GroundCamera(320, 240, 160, 160, 160, 120, 0.5))
STRAIGHT = [(160, 239), (160, 200), (160, 170)]
SLIGHT_LEFT = [(160, 239), (155, 200), (150, 170)]
SLIGHT_RIGHT = [(160, 239), (165, 200), (170, 170)]
OUTLIER = [(160, 239), (60, 200), (20, 190)]
ABOVE_HORIZON = [(160, 100), (160, 90)]


class SelectionTests(unittest.TestCase):
    def test_resample_keeps_endpoints_and_spacing(self):
        points = resample([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], 5)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[-1], (1.0, 1.0))
        self.assertEqual(points[2], (1.0, 0.0))
        self.assertAlmostEqual(path_distance([(0, 0), (1, 0)], [(0, 1), (1, 1)]), 1.0)

    def test_medoid_ignores_outlier_and_unusable_candidates(self):
        selection = select_candidate([OUTLIER, ABOVE_HORIZON, SLIGHT_LEFT, STRAIGHT, SLIGHT_RIGHT], PROJECTOR)
        self.assertEqual(selection.index, 3)
        self.assertIn("medoid of 4 usable candidates", selection.reason)
        reports = {report.index: report for report in selection.candidates}
        self.assertFalse(reports[1].usable)
        self.assertEqual(reports[1].rejected, "point 0: not_ground_facing")
        self.assertGreater(reports[0].mean_distance_m, reports[3].mean_distance_m)

    def test_index_strategy_reports_instead_of_replacing_unusable_choice(self):
        selection = select_candidate([STRAIGHT, ABOVE_HORIZON], PROJECTOR, strategy="index", index=1)
        self.assertFalse(selection.usable)
        self.assertIn("candidate 1 unusable", selection.reason)
        self.assertIn("missing", select_candidate([STRAIGHT], PROJECTOR, strategy="index", index=3).reason)
        self.assertEqual(select_candidate([STRAIGHT], PROJECTOR, strategy="index").index, 0)

    def test_single_and_no_usable_candidates(self):
        self.assertEqual(select_candidate([ABOVE_HORIZON, STRAIGHT], PROJECTOR).reason, "only usable candidate")
        self.assertEqual(select_candidate([ABOVE_HORIZON], PROJECTOR).reason, "no usable candidate")
        short = [(160, 239), (160, 238)]
        self.assertEqual(select_candidate([short], PROJECTOR, min_length_m=0.5).candidates[0].rejected,
                         "shorter_than_minimum")

    def test_invalid_arguments(self):
        for kwargs in ({"strategy": "best"}, {"index": -1}, {"min_length_m": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_candidate([STRAIGHT], PROJECTOR, **kwargs)


if __name__ == "__main__":
    unittest.main()
