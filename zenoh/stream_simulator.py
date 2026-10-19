"""Receive live Zenoh paths and follow them in a planar static-image demo.

Only local simulated motion is produced. No publisher or Go2 command connection
exists in this process. Static-image capture poses must be explicitly annotated;
unannotated camera streams cannot be used as odometry.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
from threading import RLock
import time

import zenoh

from config import ControlCommand, JSON_ENCODING, NodeState, TrajectoryResult, load_node_config
from controller import Pose2D, PurePursuitController, VelocityCommand
from node import HERE, has_encoding, stop_on_signals
from projection import GroundCamera, GroundProjector, path_to_world


class StreamFollower:
    """Thread-safe latest-path controller, with bounded history and expiry."""

    def __init__(self, camera: GroundCamera, candidate: int = 0, path_timeout: float = 3.0,
                 state_timeout: float = 1.0, clock=time.monotonic, wall_clock=time.time):
        if type(candidate) is not int or candidate < 0:
            raise ValueError("candidate must be a nonnegative integer")
        for value in (path_timeout, state_timeout):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be finite and positive")
        self.projector = GroundProjector(camera)
        self.controller = PurePursuitController()
        self.candidate = candidate
        self.path_timeout, self.state_timeout = path_timeout, state_timeout
        self.clock, self.wall_clock = clock, wall_clock
        self.lock = RLock()
        self.node_state = None
        self.node_at = None
        self.remote_stopped = False
        self.paused = False
        self.closed = False
        self.received = self.accepted = 0
        self.rejected = Counter()
        self.events = deque(maxlen=50)
        self.reset()

    def reset(self):
        with self.lock:
            self.pose = Pose2D(0, 0, 0)
            self.trail = deque([(0.0, 0.0)], maxlen=10000)
            self.elapsed = 0.0
            self.path = ()
            self.capture = None
            self.result_at = None
            self.source_age_at_receipt = None
            self.source_id = None
            self.last_frame_id = -1
            self.cutoff_wall = self.wall_clock()
            self.command = VelocityCommand()
            self.target = None
            self.distance_to_end = None
            self.last_status = "waiting_for_result"
            self.status = "waiting_for_result"
            self.paused = False
            # Remote stop is intentionally preserved until a remote start.

    def _invalidate(self, reason: str):
        if self.status != reason:
            self.events.append({"event": reason, "time_seconds": self.elapsed})
        self.path = ()
        self.command = VelocityCommand()
        self.target = None
        self.distance_to_end = None
        self.status = reason

    def _ready_reason(self):
        if self.closed:
            return "closed"
        if self.paused:
            return "paused"
        if self.remote_stopped:
            return "remote_stop"
        if self.node_at is None or self.clock() - self.node_at > self.state_timeout:
            return "stale_node"
        state = self.node_state
        if (state.lifecycle in ("loading", "paused", "error", "stopping", "stopped")
                or not state.inference.model_loaded or not state.inference.enabled):
            return "node_not_ready"
        if not state.camera.fresh or not state.cli_connected:
            return "stale_inputs"
        return None

    def receive_state(self, state: NodeState):
        with self.lock:
            previous = self.node_state
            self.node_state, self.node_at = state, self.clock()
            if (not state.inference.enabled or state.lifecycle in ("error", "stopping", "stopped")):
                if previous is not None and previous.inference.enabled:
                    self.cutoff_wall = self.wall_clock()
                self._invalidate("node_not_ready")

    def receive_control(self, control: ControlCommand):
        with self.lock:
            if control.type in ("stop", "start"):
                self.remote_stopped = control.type == "stop"
                self.cutoff_wall = self.wall_clock()
                self._invalidate("remote_stop" if self.remote_stopped else "waiting_for_result")

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            self.cutoff_wall = self.wall_clock()
            self._invalidate("paused" if self.paused else "waiting_for_result")

    def reject(self, reason: str, invalidate: bool = True):
        with self.lock:
            self.rejected[reason] += 1
            if invalidate:
                self._invalidate(reason)

    def receive_result(self, result: TrajectoryResult) -> bool:
        with self.lock:
            self.received += 1
            reason = self._ready_reason()
            if reason:
                self.reject(reason)
                return False
            capture = result.capture
            if capture is None:
                self.reject("missing_capture_pose")
                return False
            if self.source_id is not None and capture.source_id != self.source_id:
                self.reject("source_changed_reset_required")
                return False
            if capture.frame_id <= self.last_frame_id:
                self.reject("duplicate_or_reordered", invalidate=False)
                return False
            if capture.captured_at_unix_seconds <= self.cutoff_wall:
                self.reject("before_start_or_reset", invalidate=False)
                return False
            age = self.wall_clock() - capture.captured_at_unix_seconds
            if (age < -0.25 or age > self.path_timeout or not math.isfinite(result.source_age_seconds)
                    or not 0 <= result.source_age_seconds <= self.path_timeout):
                self.reject("stale_or_future_result")
                return False
            self.last_frame_id = capture.frame_id
            self.source_id = capture.source_id
            try:
                self.projector.camera.check_image_size(result.image_width, result.image_height)
                if self.candidate >= len(result.trajectories):
                    raise ValueError("selected candidate is missing")
                projected = self.projector.project_path(result.trajectories[self.candidate], self.candidate)
                if not projected.usable:
                    raise ValueError(projected.stop_reason or "fewer than two distinct points")
                pose = Pose2D(capture.pose.x, capture.pose.y, capture.pose.yaw)
                path = path_to_world(projected.points_robot_m, pose)
            except ValueError as error:
                self.reject("invalid_projection")
                self.events.append({"event": "invalid_projection", "detail": str(error)})
                return False
            self.capture = capture
            self.path = path
            self.result_at = self.clock()
            self.source_age_at_receipt = max(0.0, age, result.source_age_seconds)
            self.accepted += 1
            self.status = "path_updated"
            self.events.append({"event": "path_updated", "frame_id": capture.frame_id,
                                "capture_pose": asdict(pose), "robot_pose": asdict(self.pose),
                                "first_world_point": path[0], "truncated": projected.truncated})
            # Cancel the old command immediately; the next tick computes a new one.
            self.command = VelocityCommand()
            return True

    def tick(self, dt: float = 0.05):
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("dt must be in (0, 0.1]")
        with self.lock:
            reason = self._ready_reason()
            if reason:
                self._invalidate(reason)
                return
            if not self.path:
                return
            age = max(self.wall_clock() - self.capture.captured_at_unix_seconds,
                      self.source_age_at_receipt + self.clock() - self.result_at)
            if age > self.path_timeout or age < -0.25:
                self._invalidate("stale_path")
                return
            result = self.controller.compute(self.path, self.pose)
            command = result.command
            cosine, sine = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
            yaw = self.pose.yaw + command.vyaw * dt
            self.pose = Pose2D(self.pose.x + (command.vx * cosine - command.vy * sine) * dt,
                               self.pose.y + (command.vx * sine + command.vy * cosine) * dt,
                               math.atan2(math.sin(yaw), math.cos(yaw)))
            self.elapsed += dt
            self.trail.append((self.pose.x, self.pose.y))
            result = self.controller.compute(self.path, self.pose)
            self.command, self.status = result.command, result.status
            self.target, self.distance_to_end = result.target_world, result.distance_to_end_m

    def close(self):
        with self.lock:
            self.last_status = self.status
            self.status = "closed"
            self.command = VelocityCommand()
            self.closed = True

    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.status, "last_status": self.last_status, "paused": self.paused,
                    "pose": asdict(self.pose), "simulated_motion_seconds": self.elapsed,
                    "command": self.command.as_message(), "candidate": self.candidate,
                    "received": self.received, "accepted": self.accepted, "rejected": dict(self.rejected),
                    "path_world_m": self.path, "trail_world_m": list(self.trail), "target_world_m": self.target,
                    "distance_to_end_m": self.distance_to_end,
                    "capture": None if self.capture is None else self.capture.model_dump(),
                    "calibration_status": self.projector.camera.calibration_status,
                    "events": list(self.events)}


class StreamView:
    def __init__(self, follower: StreamFollower):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        self.follower = follower
        self.figure, self.axes = plt.subplots(figsize=(11, 7))
        self.figure.subplots_adjust(left=0.09, right=0.72, bottom=0.14, top=0.85)
        self.figure.suptitle("VAMOS live Zenoh paths | 2D simulation", fontsize=15)
        self.figure.text(0.09, 0.92,
                         f"Static-image capture pose fixed / {follower.projector.camera.calibration_status} calibration / no robot commands",
                         fontsize=9)
        self.axes.set_xlabel("World x [m]")
        self.axes.set_ylabel("World y [m]")
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, alpha=0.25)
        self.reference, = self.axes.plot([], [], "--", color="gray", label="Latest reference path")
        self.trail, = self.axes.plot([], [], color="tab:blue", label="Robot trail")
        self.target, = self.axes.plot([], [], "o", color="tab:orange", label="Lookahead")
        self.endpoint, = self.axes.plot([], [], "*", color="tab:red", ms=12, label="Endpoint")
        self.robot = Polygon([(0, 0)] * 3, closed=True, facecolor="tab:blue", edgecolor="navy")
        self.axes.add_patch(self.robot)
        self.axes.legend(loc="upper left", fontsize=8)
        self.info = self.figure.text(0.76, 0.82, "", va="top", family="monospace", fontsize=10)
        self.figure.text(0.09, 0.04, "Space: pause/resume    R: reset pose/source    Esc: close", fontsize=10)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def on_key(self, event):
        if event.key == " ":
            self.follower.toggle_pause()
        elif event.key == "r":
            self.follower.reset()
        elif event.key == "escape":
            import matplotlib.pyplot as plt

            plt.close(self.figure)
        self.draw()

    def draw(self):
        s = self.follower.snapshot()
        path, trail = s["path_world_m"], s["trail_world_m"]
        self.reference.set_data(*zip(*path)) if path else self.reference.set_data([], [])
        self.trail.set_data(*zip(*trail))
        if path:
            self.endpoint.set_data([path[-1][0]], [path[-1][1]])
        else:
            self.endpoint.set_data([], [])
        if s["target_world_m"] is not None:
            x, y = s["target_world_m"]
            self.target.set_data([x], [y])
        else:
            self.target.set_data([], [])
        pose = s["pose"]
        cosine, sine = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        self.robot.set_xy([(pose["x"] + cosine * x - sine * y, pose["y"] + sine * x + cosine * y)
                           for x, y in ((0.18, 0), (-0.12, 0.1), (-0.12, -0.1))])
        points = [*path, *trail]
        xs, ys = zip(*points)
        self.axes.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        self.axes.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        command = s["command"]
        display_status = (f"closed\nLast: {s['last_status']}" if s["status"] == "closed" else s["status"])
        self.info.set_text(f"{display_status}\n\nCandidate: {s['candidate']}\n"
                           f"Received: {s['received']}\nAccepted: {s['accepted']}\n\n"
                           f"v: {command['vx']:.3f} m/s\nw: {command['vyaw']:+.3f} rad/s\n\n"
                           f"Pose [m]\nx: {pose['x']:.2f}\ny: {pose['y']:.2f}\n\n"
                           "Refresh keeps the\ncapture pose fixed.\n\nStatic-image demo,\nnot moving-camera\nnavigation.")
        self.figure.canvas.draw_idle()


def run_session(follower: StreamFollower, keys: dict, zenoh_config, stop, *, headless=False,
                duration=30.0, output_dir: Path | None = None):
    def callback(model, handler, name):
        def receive(sample):
            try:
                if not has_encoding(sample, JSON_ENCODING):
                    raise ValueError("expected application/json")
                handler(model.model_validate_json(sample.payload.to_bytes()))
            except ValueError as error:
                follower.reject(f"invalid_{name}")
                logging.warning("Rejected %s: %s", name, error)
        return receive

    view = None
    try:
        with zenoh.open(zenoh_config) as session, ExitStack() as resources:
            resources.enter_context(session.declare_subscriber(keys["trajectories"],
                callback(TrajectoryResult, follower.receive_result, "result")))
            resources.enter_context(session.declare_subscriber(keys["state"],
                callback(NodeState, follower.receive_state, "state")))
            resources.enter_context(session.declare_subscriber(keys["command"],
                callback(ControlCommand, follower.receive_control, "control")))
            if headless:
                deadline = time.monotonic() + duration
                while not stop.is_set() and time.monotonic() < deadline:
                    follower.tick()
                    stop.wait(0.05)
            else:
                import matplotlib.pyplot as plt
                from matplotlib.animation import FuncAnimation

                view = StreamView(follower)

                def tick(_):
                    if stop.is_set():
                        plt.close(view.figure)
                        return
                    follower.tick()
                    view.draw()

                view.animation = FuncAnimation(view.figure, tick, interval=50,
                                               init_func=lambda: (), cache_frame_data=False)
                plt.show()
    finally:
        follower.close()
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "stream_report.json").write_text(
                json.dumps(follower.snapshot(), indent=2, allow_nan=False), encoding="utf-8")
            if headless:
                import matplotlib

                matplotlib.use("Agg")
                view = StreamView(follower)
            if view:
                view.draw()
                view.figure.savefig(output_dir / "stream.png", dpi=140)
                import matplotlib.pyplot as plt

                plt.close(view.figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, default=HERE / "inference-node-config.example.json5")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "peer-zenoh-config.example.json5")
    parser.add_argument("--camera", type=Path, default=HERE / "ground-camera.example.json")
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--path-timeout", type=float, default=3.0)
    parser.add_argument("--state-timeout", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=30.0, help="Headless wall-clock duration [s]")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_node_config(args.node_config)
        follower = StreamFollower(GroundCamera.load(args.camera), args.candidate,
                                  args.path_timeout, args.state_timeout)
        with stop_on_signals() as stop:
            run_session(follower, config.keys(), zenoh.Config.from_file(str(args.zenoh_config)), stop,
                        headless=args.headless, duration=args.duration, output_dir=args.output_dir)
        summary = follower.snapshot()
        print(json.dumps({k: summary[k] for k in
                          ("status", "last_status", "pose", "received", "accepted", "rejected")}, allow_nan=False))
        return 0 if follower.accepted else 1
    except (ValueError, OSError, RuntimeError, zenoh.ZError) as error:
        logging.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
