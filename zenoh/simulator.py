"""Visual planar path-following check; no model, network or robot is used.

This ideal kinematic model follows velocity commands immediately. It models no
legs, inertia, slip, obstacles, sensing error or communication/inference delay.
Default paths are hand-authored in metres. Other entry points may supply a
frozen projected path in the same fixed world frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from controller import Pose2D, PurePursuitController


@dataclass(frozen=True)
class Scenario:
    name: str
    path: tuple[tuple[float, float], ...]
    initial_pose: Pose2D


SCENARIOS = {
    "straight": Scenario("Straight", ((0, 0), (3, 0)), Pose2D(0, 0, 0)),
    "curve": Scenario("Curve", ((0, 0), (1, 0), (2, 0.5), (2.5, 1.5), (3.5, 2)),
                      Pose2D(0, 0, 0)),
    "offset": Scenario("Lateral offset", ((0, 0), (3, 0)), Pose2D(0, 0.7, 0)),
    "behind": Scenario("Goal behind", ((0, 0), (-2, 0)), Pose2D(0, 0, 0)),
    "overshoot": Scenario("Past endpoint", ((0, 0), (2, 0)), Pose2D(2.6, 0, 0)),
}


class Simulation:
    """Deterministic, fixed-step Euler integration in a shared world frame."""

    def __init__(self, scenario: Scenario, dt: float = 0.05):
        if isinstance(dt, bool) or not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("dt must be finite and in (0, 0.1] seconds")
        self.dt = dt
        self.controller = PurePursuitController()
        self.reset(scenario)

    def reset(self, scenario: Scenario | None = None):
        if scenario is not None:
            self.scenario = scenario
        self.pose = self.scenario.initial_pose
        self.elapsed = 0.0
        self.steps = 0
        self.paused = False
        self.trail = [(self.pose.x, self.pose.y)]
        self.result = self.controller.compute(self.scenario.path, self.pose)

    @property
    def finished(self) -> bool:
        return self.result.status not in ("tracking", "rotating")

    def step(self):
        if self.paused or self.finished:
            return
        command = self.result.command
        cosine, sine = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        yaw = self.pose.yaw + command.vyaw * self.dt
        self.pose = Pose2D(
            self.pose.x + (command.vx * cosine - command.vy * sine) * self.dt,
            self.pose.y + (command.vx * sine + command.vy * cosine) * self.dt,
            math.atan2(math.sin(yaw), math.cos(yaw)),
        )
        self.steps += 1
        self.elapsed = self.steps * self.dt
        self.trail.append((self.pose.x, self.pose.y))
        self.result = self.controller.compute(self.scenario.path, self.pose)

    def run(self, max_time: float = 120.0) -> bool:
        if not math.isfinite(max_time) or max_time <= 0:
            raise ValueError("max_time must be finite and positive")
        # A step budget also bounds runs paused by a caller.
        for _ in range(math.ceil(max_time / self.dt)):
            if self.finished:
                break
            self.step()
        return self.result.status == "reached"

    def summary(self) -> dict:
        return {"scenario": self.scenario.name, "status": self.result.status,
                "time_seconds": self.elapsed, "steps": self.steps,
                "pose": {"x": self.pose.x, "y": self.pose.y, "yaw": self.pose.yaw},
                "distance_to_end_m": self.result.distance_to_end_m,
                "command": self.result.command.as_message()}


class SimulatorView:
    """Matplotlib window with scenario selection, pause, reset and speed."""

    def __init__(self, simulation: Simulation, speed: float = 2.0, controls: bool = True,
                 show_scenarios: bool = True,
                 description: str = "Synthetic metric paths / ideal motion / no robot connection"):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        self.simulation = simulation
        self.speed = speed
        self._step_credit = 0.0
        self.figure, self.axes = plt.subplots(figsize=(11, 7))
        self.figure.subplots_adjust(left=0.09, right=0.75, bottom=0.21, top=0.88)
        self.figure.suptitle("VAMOS path follower | 2D kinematic simulation", fontsize=15, y=0.99)
        self.figure.text(0.09, 0.935, description,
                         fontsize=9, color="dimgray")
        self.axes.set_xlabel("World x [m]")
        self.axes.set_ylabel("World y [m]")
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, alpha=0.25)
        self.path_line, = self.axes.plot([], [], "--", color="0.45", label="Reference path")
        self.trail_line, = self.axes.plot([], [], color="tab:blue", lw=2, label="Robot trail")
        self.goal, = self.axes.plot([], [], "*", color="tab:red", ms=14, label="Endpoint")
        self.target, = self.axes.plot([], [], "o", color="tab:orange", ms=7, label="Lookahead")
        self.target_line, = self.axes.plot([], [], ":", color="tab:orange", alpha=0.8)
        self.robot = Polygon([(0, 0)] * 3, closed=True, facecolor="tab:blue", edgecolor="navy")
        self.axes.add_patch(self.robot)
        self.axes.legend(loc="upper left", fontsize=8)
        self.info = self.figure.text(0.78, 0.48, "", fontsize=10, va="top", family="monospace")
        shortcuts = "Space: pause/resume    R: reset    Esc: close"
        if show_scenarios:
            shortcuts += "    Left/Right: scenario"
        self.figure.text(0.09, 0.035, shortcuts,
                         fontsize=9)
        if controls:
            from matplotlib.widgets import Button, RadioButtons, Slider

            if show_scenarios:
                active = next((i for i, s in enumerate(SCENARIOS.values())
                               if s == simulation.scenario), 0)
                self.scenario_selector = RadioButtons(self.figure.add_axes((0.78, 0.59, 0.21, 0.28)),
                                                      list(SCENARIOS), active=active)
                self.scenario_selector.on_clicked(self.select_scenario)
            self.pause_button = Button(self.figure.add_axes((0.09, 0.10, 0.16, 0.05)), "Pause / Resume")
            self.pause_button.on_clicked(self.toggle_pause)
            self.reset_button = Button(self.figure.add_axes((0.28, 0.10, 0.12, 0.05)), "Reset")
            self.reset_button.on_clicked(self.reset)
            self.speed_slider = Slider(self.figure.add_axes((0.50, 0.11, 0.23, 0.025)),
                                       "Speed", 0.5, 8.0, valinit=speed, valstep=0.5, valfmt="%1.1fx")
            self.speed_slider.on_changed(self.set_speed)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw(reframe=True)

    def draw(self, reframe: bool = False):
        sim = self.simulation
        points = sim.scenario.path
        self.path_line.set_data(*zip(*points))
        self.trail_line.set_data(*zip(*sim.trail))
        self.goal.set_data([points[-1][0]], [points[-1][1]])
        pose = sim.pose
        cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
        self.robot.set_xy([(pose.x + cosine * x - sine * y, pose.y + sine * x + cosine * y)
                           for x, y in ((0.18, 0), (-0.12, 0.10), (-0.12, -0.10))])
        if sim.result.target_world is not None:
            x, y = sim.result.target_world
            self.target.set_data([x], [y])
            self.target_line.set_data([pose.x, x], [pose.y, y])
        else:
            self.target.set_data([], [])
            self.target_line.set_data([], [])
        command = sim.result.command
        status = "paused" if sim.paused and not sim.finished else sim.result.status
        vx, yaw_rate = (0.0, 0.0) if sim.paused else (command.vx, command.vyaw)
        distance = sim.result.distance_to_end_m
        distance_text = "n/a" if distance is None else f"{distance:.3f} m"
        self.info.set_text(f"{sim.scenario.name}\n\nState: {status}\nTime: {sim.elapsed:5.1f} s\n"
                           f"v: {vx:.3f} m/s\nw: {yaw_rate:+.3f} rad/s\n\n"
                           f"End distance:\n{distance_text}\n\nPlayback: {self.speed:.1f}x")
        self.axes.set_title(sim.scenario.name)
        if reframe:
            all_points = [*points, *sim.trail]
            xs, ys = zip(*all_points)
            self.axes.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
            self.axes.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
        self.figure.canvas.draw_idle()

    def tick(self, _frame=None):
        if not self.simulation.paused and not self.simulation.finished:
            self._step_credit += self.speed
            count = int(self._step_credit)
            self._step_credit -= count
            for _ in range(count):
                self.simulation.step()
        self.draw()

    def toggle_pause(self, _event=None):
        if not self.simulation.finished:
            self.simulation.paused = not self.simulation.paused
        self.draw()

    def reset(self, _event=None):
        self.simulation.reset()
        self._step_credit = 0.0
        self.draw(reframe=True)

    def select_scenario(self, key: str):
        self.simulation.reset(SCENARIOS[key])
        self._step_credit = 0.0
        self.draw(reframe=True)

    def set_speed(self, speed: float):
        self.speed = speed
        self.draw()

    def on_key(self, event):
        if event.key == " ":
            self.toggle_pause()
        elif event.key == "r":
            self.reset()
        elif event.key in ("left", "right") and hasattr(self, "scenario_selector"):
            keys = list(SCENARIOS)
            index = keys.index(self.scenario_selector.value_selected)
            self.scenario_selector.set_active((index + (1 if event.key == "right" else -1)) % len(keys))
        elif event.key == "escape":
            import matplotlib.pyplot as plt

            plt.close(self.figure)

    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # Keep the animation alive for the window's lifetime. The timer only
        # controls playback; integration always uses the fixed simulation dt.
        self.animation = FuncAnimation(self.figure, self.tick,
                                       interval=self.simulation.dt * 1000,
                                       init_func=lambda: (), cache_frame_data=False)
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="curve")
    parser.add_argument("--speed", type=float, default=2.0, help="GUI playback multiplier (0.5 to 8)")
    parser.add_argument("--headless", action="store_true", help="Run to completion without a window")
    parser.add_argument("--max-time", type=float, default=120.0, help="Headless simulated time budget [s]")
    parser.add_argument("--save", type=Path, help="Save the final plot as PNG (requires --headless)")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0.5 <= args.speed <= 8:
        parser.error("--speed must be in [0.5, 8]")
    if not math.isfinite(args.max_time) or args.max_time <= 0:
        parser.error("--max-time must be finite and positive")
    if args.save and not args.headless:
        parser.error("--save requires --headless")
    if args.save and args.save.suffix.lower() != ".png":
        parser.error("--save must name a .png file")
    simulation = Simulation(SCENARIOS[args.scenario])
    if args.headless:
        reached = simulation.run(args.max_time)
        if args.save:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            view = SimulatorView(simulation, speed=args.speed, controls=False)
            args.save.parent.mkdir(parents=True, exist_ok=True)
            view.figure.savefig(args.save, dpi=140)
            plt.close(view.figure)
        print(json.dumps(simulation.summary(), allow_nan=False))
        return 0 if reached else 1
    SimulatorView(simulation, speed=args.speed).show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
