from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller import VelocityCommand
from simulator import SCENARIOS, Simulation, SimulatorView


class SimulationTests(unittest.TestCase):
    def test_all_scenarios_reach_endpoint_and_remain_stopped(self):
        for key, scenario in SCENARIOS.items():
            with self.subTest(scenario=key):
                sim = Simulation(scenario)
                self.assertTrue(sim.run(), sim.summary())
                self.assertLessEqual(sim.result.distance_to_end_m, 0.1)
                self.assertEqual(sim.result.command, VelocityCommand())
                self.assertTrue(all(math.isfinite(v) for point in sim.trail for v in point))
                final = sim.summary(), list(sim.trail)
                for _ in range(20):
                    sim.step()
                self.assertEqual((sim.summary(), sim.trail), final)

    def test_pause_resume_and_reset_restore_complete_state(self):
        sim = Simulation(SCENARIOS["curve"])
        for _ in range(10):
            sim.step()
        before = sim.summary(), list(sim.trail)
        sim.paused = True
        sim.step()
        self.assertEqual((sim.summary(), sim.trail), before)
        sim.paused = False
        sim.step()
        self.assertGreater(sim.elapsed, before[0]["time_seconds"])
        sim.reset()
        fresh = Simulation(SCENARIOS["curve"])
        self.assertEqual(sim.summary(), fresh.summary())
        self.assertEqual(sim.trail, fresh.trail)
        self.assertFalse(sim.paused)

    def test_behind_goal_turns_in_place_before_moving(self):
        sim = Simulation(SCENARIOS["behind"])
        self.assertEqual(sim.result.status, "rotating")
        sim.step()
        self.assertEqual((sim.pose.x, sim.pose.y), (0, 0))
        self.assertGreater(sim.pose.yaw, 0)

    def test_time_budget_cannot_be_reported_as_reached(self):
        sim = Simulation(SCENARIOS["straight"])
        self.assertFalse(sim.run(0.1))
        self.assertEqual(sim.result.status, "tracking")
        self.assertAlmostEqual(sim.elapsed, 0.1)

    def test_cli_exit_status_distinguishes_timeout(self):
        script = Path(__file__).resolve().parents[1] / "simulator.py"
        for extra_args, expected in (([], 0), (["--max-time", "0.1"], 1)):
            with self.subTest(args=extra_args):
                result = subprocess.run([sys.executable, str(script), "--headless", *extra_args],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertIn('"status":', result.stdout)


class SimulatorViewTests(unittest.TestCase):
    def test_controls_and_render_use_the_live_simulation(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        view = SimulatorView(Simulation(SCENARIOS["curve"]), speed=2)
        try:
            view.tick()
            self.assertEqual(view.simulation.steps, 2)
            view.on_key(SimpleNamespace(key=" "))
            view.tick()
            self.assertEqual(view.simulation.steps, 2)
            self.assertIn("paused", view.info.get_text())
            view.on_key(SimpleNamespace(key="r"))
            self.assertEqual(view.simulation.steps, 0)
            view.on_key(SimpleNamespace(key="right"))
            self.assertEqual(view.simulation.scenario, SCENARIOS["offset"])
            view.speed_slider.set_val(0.5)
            view.tick()
            self.assertEqual(view.simulation.steps, 0)
            view.tick()
            self.assertEqual(view.simulation.steps, 1)
            self.assertTrue(view.simulation.run())
            view.draw()
            view.figure.canvas.draw()
            self.assertIn("reached", view.info.get_text())
            xs, ys = view.trail_line.get_data()
            self.assertEqual((xs[-1], ys[-1]), (view.simulation.pose.x, view.simulation.pose.y))
        finally:
            plt.close(view.figure)


if __name__ == "__main__":
    unittest.main()
