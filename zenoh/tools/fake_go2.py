"""Simulated Go2 node for robot-free tests of go2_executor.py.

Speaks the unitree-go2-zenoh-node contract on `{prefix}/state` and
`{prefix}/command`: posture changes take time and report motion, velocity is
accepted only while standing and expires after 0.25 s, and an ideal planar
model integrates the commanded velocity. The optional window shows the
simulated robot and the executor's current step path. Never run it alongside
a real Go2 node on the same keys.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import json
import logging
import math
from pathlib import Path
import sys
from threading import RLock
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zenoh

from config import ExecutorState, JSON_ENCODING, load_node_config
from node import HERE, has_encoding, stop_on_signals

STANDING = ("ready_stand", "locomotion")
VELOCITY_TIMEOUT_SECONDS = 0.25


class SimulatedGo2:
    def __init__(self, state="down", posture_seconds=1.0, clock=time.monotonic, publish_pose=True):
        self.state, self.motion = state, "quiescent"
        self.publish_pose = publish_pose
        self.posture_seconds, self.clock = posture_seconds, clock
        self.connected = self.accepting = True
        self.transition: tuple[str, float] | None = None
        self.velocity = (0.0, 0.0, 0.0)
        self.velocity_at = -math.inf
        self.x = self.y = self.yaw = 0.0
        self.trail = deque([(0.0, 0.0)], maxlen=20000)
        self.last_step: float | None = None
        self.lock = RLock()

    def receive(self, message: dict, now: float | None = None):
        now = self.clock() if now is None else now
        with self.lock:
            if message.get("type") == "posture" and message.get("posture") in ("stand", "down"):
                target = "ready_stand" if message["posture"] == "stand" else "down"
                if target != self.state:
                    self.velocity, self.transition = (0.0, 0.0, 0.0), (target, now + self.posture_seconds)
            elif message.get("type") == "velocity" and self.state in STANDING and self.transition is None:
                self.velocity = tuple(float(message[name]) for name in ("vx", "vy", "vyaw"))
                self.velocity_at = now
                if self.velocity != (0.0, 0.0, 0.0):
                    self.state = "locomotion"

    def step(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        with self.lock:
            dt = 0.0 if self.last_step is None else min(0.1, max(0.0, now - self.last_step))
            self.last_step = now
            if now - self.velocity_at > VELOCITY_TIMEOUT_SECONDS:
                self.velocity = (0.0, 0.0, 0.0)
            vx, vy, vyaw = self.velocity
            cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (vx * cosine - vy * sine) * dt
            self.y += (vx * sine + vy * cosine) * dt
            self.yaw = math.atan2(math.sin(self.yaw + vyaw * dt), math.cos(self.yaw + vyaw * dt))
            if dt and self.velocity != (0.0, 0.0, 0.0):
                self.trail.append((self.x, self.y))
            if self.transition is not None and now >= self.transition[1]:
                self.state, self.transition = self.transition[0], None
            moving = self.transition is not None or self.velocity != (0.0, 0.0, 0.0)
            self.motion = "moving" if moving else "quiescent"
            robot_state = {"state": self.state, "motion": self.motion, "velocity": [vx, vy, 0.0], "yaw_speed": vyaw}
            if self.publish_pose:  # The proposed odometry fields.
                robot_state.update(position=[self.x, self.y, 0.0], yaw=self.yaw)
            return {"lifecycle": "running", "robot_connected": self.connected,
                    "accepting_commands": self.accepting and self.connected, "simulated": True,
                    "robot_state": robot_state}


class SimulatorView:
    def __init__(self, robot: SimulatedGo2):
        import matplotlib.pyplot as plt

        self.robot, self.executor = robot, None
        self.figure, self.axes = plt.subplots(figsize=(9, 7))
        self.figure.suptitle("Simulated Go2 | real go2_executor.py | no robot", fontsize=13)
        self.axes.set_xlabel("x [m]")
        self.axes.set_ylabel("y [m]")
        self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.grid(True, alpha=0.25)
        self.path, = self.axes.plot([], [], "--", color="gray", label="Executor step path")
        self.goal, = self.axes.plot([], [], "*", color="tab:red", ms=14, label="Goal")
        self.trail, = self.axes.plot([], [], color="tab:blue", label="Simulated robot trail")
        self.heading, = self.axes.plot([], [], color="navy", lw=3)
        self.axes.legend(loc="upper left", fontsize=8)
        self.info = self.figure.text(0.70, 0.86, "", va="top", family="monospace", fontsize=9)

    def receive_executor(self, state: ExecutorState):
        self.executor = state

    def draw(self):
        robot, executor = self.robot, self.executor
        with robot.lock:
            trail, x, y, yaw = list(robot.trail), robot.x, robot.y, robot.yaw
            text = f"Go2: {robot.state}\nmotion: {robot.motion}\nv: {robot.velocity[0]:.2f} m/s\n" \
                   f"w: {robot.velocity[2]:+.2f} rad/s"
        if executor is not None:
            text += f"\n\nexecutor: {executor.motion}\n{executor.message[:28]}"
            if executor.goal_reached:
                text += "\nGOAL REACHED"
        self.path.set_data(*zip(*executor.path_m)) if executor and executor.path_m else self.path.set_data([], [])
        goal = executor.goal_world_m if executor else None
        self.goal.set_data([goal[0]], [goal[1]]) if goal else self.goal.set_data([], [])
        self.trail.set_data(*zip(*trail))
        self.heading.set_data([x, x + 0.25 * math.cos(yaw)], [y, y + 0.25 * math.sin(yaw)])
        points = trail + (executor.path_m if executor else []) + ([goal] if goal else [])
        xs, ys = zip(*points)
        self.axes.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        self.axes.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
        self.info.set_text(text)
        self.figure.canvas.draw_idle()


def serve(robot: SimulatedGo2, keys: dict, session: zenoh.Session, stop, view: SimulatorView | None = None):
    """Publish simulated state at 20 Hz and apply received commands until stopped."""
    def on_command(sample):
        try:
            if not has_encoding(sample, JSON_ENCODING):
                raise ValueError("expected application/json")
            robot.receive(json.loads(sample.payload.to_bytes()))
        except (ValueError, KeyError, TypeError) as error:
            logging.warning("Rejected Go2 command: %s", error)

    def on_executor(sample):
        if view is not None and has_encoding(sample, JSON_ENCODING):
            try:
                view.receive_executor(ExecutorState.model_validate_json(sample.payload.to_bytes()))
            except ValueError:
                pass

    with ExitStack() as resources:
        resources.enter_context(session.declare_subscriber(keys["go2_command"], on_command))
        resources.enter_context(session.declare_subscriber(keys["executor_state"], on_executor))
        state = resources.enter_context(session.declare_publisher(keys["go2_state"], encoding=JSON_ENCODING))
        if view is None:
            while not stop.wait(0.05):
                state.put(json.dumps(robot.step()))
            return
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def tick(_):
            if stop.is_set():
                plt.close(view.figure)
                return
            state.put(json.dumps(robot.step()))
            view.draw()

        view.animation = FuncAnimation(view.figure, tick, interval=50, cache_frame_data=False)
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node-config", type=Path, default=HERE / "inference-node-config.example.json5",
                        help="VAMOS node config, for the Go2 and executor keys")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "peer-zenoh-config.example.json5")
    parser.add_argument("--initial-state", choices=("down", "damping", "ready_stand"), default="down")
    parser.add_argument("--posture-seconds", type=float, default=1.0, help="Duration of stand/down transitions")
    parser.add_argument("--headless", action="store_true", help="No window; print the final pose on exit")
    parser.add_argument("--no-pose", action="store_true", help="Omit position/yaw, like the current Go2 node")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        keys = load_node_config(args.node_config).keys()
        robot = SimulatedGo2(args.initial_state, args.posture_seconds, publish_pose=not args.no_pose)
        with stop_on_signals() as stop, zenoh.open(zenoh.Config.from_file(str(args.zenoh_config))) as session:
            logging.info("SIMULATED Go2 on %s (%s). Ctrl+C to stop.", keys["go2_state"], args.initial_state)
            serve(robot, keys, session, stop, None if args.headless else SimulatorView(robot))
        print(json.dumps({"state": robot.state, "pose": {"x": robot.x, "y": robot.y, "yaw": robot.yaw}}))
    except (OSError, ValueError, zenoh.ZError) as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
