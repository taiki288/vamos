from __future__ import annotations

from contextlib import ExitStack
import json
import math
from pathlib import Path
from queue import Empty, Queue
import socket
import sys
import tempfile
from threading import Event, Thread
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pydantic import ValidationError
import zenoh

from config import ControlCommand, Go2NodeState, NodeConfig, TrajectoryResult
from controller import VelocityCommand
from go2_executor import ExecutorConfig, ExecutorRuntime, Go2StepExecutor, load_camera
from node import TransportState
from projection import GroundCamera
from test_stop_go import CONFIG
from test_transport import wait_for_listener

CAMERA = GroundCamera(64, 48, 32, 32, 32, 24, 0.5)
STRAIGHT = [(32, 44), (32, 36), (32, 32)]
ABOVE_HORIZON = [(32, 10), (32, 5)]


class FakeGo2:
    """Observed-state Go2 stand-in: posture changes take time, motion follows velocity."""

    def __init__(self, state="down"):
        self.state, self.motion = state, "quiescent"
        self.connected = self.accepting = True
        self.transition = None  # (target, completes_at)
        self.velocity = VelocityCommand()
        self.velocity_at = -math.inf
        self.postures, self.velocities = [], []

    def command(self, now, output):
        if output.posture is not None:
            self.postures.append(output.posture)
            target = "ready_stand" if output.posture == "stand" else "down"
            self.transition = (target, now + 0.5)
        if output.velocity is not None:
            self.velocities.append(output.velocity)
            if self.state in ("ready_stand", "locomotion"):
                self.velocity, self.velocity_at = output.velocity, now

    def step(self, now):
        if now - self.velocity_at > 0.25:  # Go2 node velocity timeout.
            self.velocity = VelocityCommand()
        if self.transition is not None:
            target, completes_at = self.transition
            self.motion = "moving"
            if now >= completes_at:
                self.state, self.transition = target, None
        else:
            self.motion = "moving" if self.velocity != VelocityCommand() else "quiescent"
        return Go2NodeState.model_validate({
            "robot_connected": self.connected, "accepting_commands": self.accepting,
            "robot_state": {"state": self.state, "motion": self.motion}})


class Go2ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.now, self.wall = 100.0, 1000.0
        self.node_config = NodeConfig.model_validate(CONFIG)
        self.node = TransportState(self.node_config, clock=lambda: self.now, wall_clock=lambda: self.wall)
        self.node.model_loaded("cpu")
        self.config = ExecutorConfig(camera_calibration="unused.json", allow_assumed_calibration=True)
        self.robot = FakeGo2()
        self.executor = self.make_executor()
        self.go2_fresh = True
        self.heartbeat = True
        self.paths = [STRAIGHT]
        self.feedback = []
        for _ in range(4):
            self.pump()

    def make_executor(self, config=None):
        return Go2StepExecutor(config or self.config, self.node_config.inference.stop_go, CAMERA,
                               clock=lambda: self.now)

    def pump(self, infer=True):
        self.now += 0.05
        self.wall += 0.05
        if self.heartbeat:
            self.node.receive_command(ControlCommand(type="heartbeat"))
        self.node.receive_frame(Image.new("RGB", (64, 48)), self.now)  # Raw camera frame.
        self.executor.receive_node_state(self.node.snapshot())
        if self.go2_fresh:
            self.executor.receive_go2_state(self.robot.step(self.now))
        output = self.executor.tick(self.now)
        self.robot.command(self.now, output)
        if output.feedback is not None:
            self.feedback.append(output.feedback)
            self.node.receive_executor(output.feedback)
        if infer:
            request = self.node.begin_inference()
            if request is not None:
                self.node.complete_inference(request, 0.1, lambda: self.executor.receive_result(self.result(request)))
        return output

    def result(self, request):
        return TrajectoryResult(
            node_session_id=self.node.snapshot().node_session_id, generated_at_unix_seconds=self.wall,
            control_epoch=request.epoch, goal_revision=request.goal_revision, cycle_id=request.cycle_id,
            inference_id=request.inference_id, frame_sequence=request.frame.sequence, image_width=64,
            image_height=48, goal_pixels=request.goal.pixels(64, 48), prompt="test", trajectories=self.paths,
            generated_texts=["test"] * len(self.paths), device="cpu", duration_seconds=0.1,
            source_age_seconds=0.0)

    def start(self):
        self.node.receive_command(ControlCommand(type="start"))

    def pump_until(self, predicate, limit=200, **kwargs):
        for _ in range(limit):
            self.pump(**kwargs)
            if predicate():
                return
        self.fail(f"condition not reached: {self.executor.snapshot()} {self.node.snapshot()}")

    def test_cycle_stands_moves_within_bounds_and_completes(self):
        self.assertEqual(self.robot.postures, [])  # Resting at startup needs no command.
        self.assertTrue(self.node.snapshot().start_ready)
        self.start()
        self.pump_until(lambda: self.node.snapshot().cycle.completed >= 1)
        self.assertEqual(self.robot.postures, ["stand"])
        moving = [v for v in self.robot.velocities if v != VelocityCommand()]
        self.assertTrue(moving)
        self.assertTrue(all(0 < v.vx <= self.config.motion.forward_speed_mps for v in moving))
        finished = [e for e in self.executor.events if e["event"] == "step_finished"]
        self.assertLessEqual(finished[0]["distance_m"], self.node_config.inference.stop_go.max_step_distance_m + 0.02)
        self.assertGreater(self.executor.pose.x, 0)
        selection = [e for e in self.executor.events if e["event"] == "selection"][0]
        self.assertEqual(selection["candidate"], 0)
        # The next cycle observes from the new stopped pose; no inference while moving.
        self.pump_until(lambda: self.node.snapshot().cycle.completed >= 2)
        self.assertEqual(self.robot.postures, ["stand"])

    def test_operator_stop_zeroes_then_requests_down_until_rest(self):
        self.start()
        self.pump_until(lambda: self.executor.phase == "moving")
        stop = ControlCommand(type="stop")
        self.executor.receive_control(stop)
        output = self.pump(infer=False)
        self.node.receive_command(stop)
        self.assertEqual(output.velocity, VelocityCommand())
        self.assertEqual(output.posture, "down")
        self.assertEqual(self.executor.phase, "stopping")
        self.pump_until(lambda: self.executor.phase == "idle")
        self.assertEqual(self.robot.state, "down")
        self.assertIsNone(self.executor.last_error)
        self.assertEqual(self.robot.postures, ["stand", "down"])

    def test_heartbeat_loss_pauses_node_and_rests_robot(self):
        self.start()
        self.pump_until(lambda: self.executor.phase == "moving")
        self.heartbeat = False
        self.pump_until(lambda: self.executor.phase == "idle" and self.robot.state == "down")
        self.assertIn("heartbeat", self.node.snapshot().status_message)
        self.assertEqual(self.robot.postures[-1], "down")

    def test_stale_go2_state_stops_motion_without_unverifiable_posture_commands(self):
        self.start()
        self.pump_until(lambda: self.executor.phase == "moving")
        self.go2_fresh = False
        self.pump_until(lambda: self.executor.phase in ("error", "stopping", "idle"))
        self.pump_until(lambda: self.executor.phase == "idle")
        self.assertEqual(self.robot.postures, ["stand"])  # No down without fresh state.
        self.assertIn("Go2", self.executor.last_error)
        self.assertFalse(self.node.snapshot().inference.enabled)
        self.assertTrue(any(f.motion == "error" for f in self.feedback))

    def test_unusable_paths_fail_without_moving(self):
        self.paths = [ABOVE_HORIZON]
        self.start()
        self.pump_until(lambda: self.executor.last_error is not None)
        self.assertIn("no executable path", self.executor.last_error)
        self.assertFalse(any(v != VelocityCommand() for v in self.robot.velocities))
        self.assertEqual(self.robot.postures, ["stand", "down"])

    def test_standing_robot_is_sent_down_at_startup(self):
        self.robot = FakeGo2("ready_stand")
        self.executor = self.make_executor()
        output = self.pump()
        self.assertEqual(output.posture, "down")
        self.pump_until(lambda: self.executor.phase == "idle" and self.robot.state == "down")

    def test_shutdown_rests_robot_and_reports_unconfirmed_rest(self):
        self.start()
        self.pump_until(lambda: self.executor.phase == "holding")
        self.executor.request_shutdown()
        self.pump_until(lambda: self.executor.shutdown_complete)
        self.assertIsNone(self.executor.shutdown_error)
        self.assertEqual(self.robot.state, "down")

        executor = self.make_executor()
        executor.request_shutdown()
        executor.tick(self.now)  # No Go2 state was ever received.
        self.assertTrue(executor.shutdown_complete)
        self.assertIn("unavailable", executor.shutdown_error)

    def test_start_is_blocked_while_go2_state_is_missing(self):
        self.go2_fresh = False
        for _ in range(15):
            self.pump()
        self.assertFalse(self.node.snapshot().start_ready)
        with self.assertRaisesRegex(ValueError, "executor"):
            self.start()


class ExecutorConfigTests(unittest.TestCase):
    def test_assumed_calibration_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "camera.json").write_text(json.dumps(CAMERA.as_json()), encoding="utf-8")
            config = ExecutorConfig(camera_calibration="camera.json")
            with self.assertRaisesRegex(ValueError, "not verified"):
                load_camera(config, directory / "executor.json5")
            allowed = config.model_copy(update={"allow_assumed_calibration": True})
            self.assertEqual(load_camera(allowed, directory / "executor.json5"), CAMERA)

    def test_invalid_rates_and_speeds_are_rejected(self):
        for changes in ({"control_frequency_hz": 4.0}, {"motion": {"forward_speed_mps": 2.0}},
                        {"shutdown_timeout_seconds": 5.0}, {"selection": "best"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ExecutorConfig.model_validate({"camera_calibration": "c.json", **changes})

    def test_example_config_is_valid(self):
        example = Path(__file__).resolve().parents[1] / "go2-executor-config.example.json5"
        from go2_executor import load_executor_config

        config = load_executor_config(example)
        self.assertFalse(config.allow_assumed_calibration)


class ExecutorNetworkTests(unittest.TestCase):
    def test_runtime_publishes_go2_contract_messages(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
        node_config = NodeConfig.model_validate(CONFIG)
        config = ExecutorConfig(camera_calibration="unused.json", allow_assumed_calibration=True,
                                shutdown_timeout_seconds=11.0)
        executor = Go2StepExecutor(config, node_config.inference.stop_go, CAMERA)
        stop, failures, commands = Event(), Queue(), Queue()

        def run():
            try:
                ExecutorRuntime(executor, node_config).run(zenoh.Config.from_json5(json.dumps({
                    "mode": "peer", "listen": {"endpoints": [endpoint]},
                    "scouting": {"multicast": {"enabled": False}}})), stop)
            except Exception as error:
                failures.put(error)

        worker = Thread(target=run, daemon=True)
        worker.start()
        try:
            wait_for_listener(endpoint)
            client = zenoh.Config.from_json5(json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]},
                                                        "scouting": {"multicast": {"enabled": False}}}))
            with zenoh.open(client) as session, ExitStack() as resources:
                resources.enter_context(session.declare_subscriber(
                    "unitree/go2/command", lambda s: commands.put((str(s.encoding), json.loads(s.payload.to_bytes())))))
                state = resources.enter_context(session.declare_publisher("unitree/go2/state",
                                                                          encoding="application/json"))
                standing = {"robot_connected": True, "accepting_commands": True,
                            "robot_state": {"state": "ready_stand", "motion": "quiescent", "extra": 1}, "lifecycle": "running"}
                deadline = time.monotonic() + 5
                received = None
                while received is None and time.monotonic() < deadline:
                    state.put(json.dumps(standing))
                    try:
                        received = commands.get(timeout=0.05)
                    except Empty:
                        pass
                self.assertIsNotNone(received, "no Go2 command for a standing robot at startup")
                encoding, payloads = received[0], [received[1]]
                while True:
                    try:
                        payloads.append(commands.get(timeout=0.2)[1])
                    except Empty:
                        break
                self.assertTrue(encoding.startswith("application/json"))
                self.assertIn({"type": "velocity", "vx": 0.0, "vy": 0.0, "vyaw": 0.0}, payloads)
                self.assertIn({"type": "posture", "posture": "down"}, payloads)
                resting = {**standing, "robot_state": {"state": "down", "motion": "quiescent"}}
                stop.set()
                deadline = time.monotonic() + 5
                while worker.is_alive() and time.monotonic() < deadline:
                    state.put(json.dumps(resting))
                    time.sleep(0.05)
        finally:
            stop.set()
            worker.join(timeout=15)
        self.assertFalse(worker.is_alive())
        self.assertTrue(failures.empty(), list(failures.queue))
        self.assertTrue(executor.shutdown_complete)


if __name__ == "__main__":
    unittest.main()
