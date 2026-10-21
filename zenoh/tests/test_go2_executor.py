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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from PIL import Image
from pydantic import ValidationError
import zenoh

from config import ControlCommand, GoalConfig, Go2NodeState, NodeConfig, TrajectoryResult
from fake_go2 import SimulatedGo2
from controller import VelocityCommand
from go2_executor import ExecutorConfig, ExecutorRuntime, Go2StepExecutor, load_camera
from node import TransportState
from projection import GroundCamera
from test_stop_go import CONFIG
from test_transport import wait_for_listener

CAMERA = GroundCamera(64, 48, 32, 32, 32, 24, 0.5)
STRAIGHT = [(32, 44), (32, 36), (32, 32)]
ABOVE_HORIZON = [(32, 10), (32, 5)]


class FakeGo2(SimulatedGo2):
    """The shared simulated Go2, fed directly from executor outputs and recording them."""

    def __init__(self, state="down"):
        super().__init__(state, posture_seconds=0.5)
        self.postures, self.velocities = [], []

    def command(self, now, output):
        if output.posture is not None:
            self.postures.append(output.posture)
            self.receive({"type": "posture", "posture": output.posture}, now)
        if output.velocity is not None:
            self.velocities.append(output.velocity)
            self.receive(output.velocity.as_message(), now)

    def observe(self, now):
        return Go2NodeState.model_validate(self.step(now))


class ExecutorHarness:
    """Real node state machine + executor + simulated Go2, on a manual clock."""

    def setUp(self, stop_go=None):
        self.now, self.wall = 100.0, 1000.0
        self.node_config = NodeConfig.model_validate({**CONFIG, "inference": {**CONFIG["inference"], "stop_go": {
            **CONFIG["inference"]["stop_go"], **(stop_go or {})}}})
        self.node = TransportState(self.node_config, clock=lambda: self.now, wall_clock=lambda: self.wall)
        self.node.model_loaded("cpu")
        self.config = ExecutorConfig(camera_calibration="unused.json", allow_assumed_calibration=True)
        self.robot = FakeGo2()
        self.executor = self.make_executor()
        self.go2_fresh = True
        self.heartbeat = True
        self.paths = [STRAIGHT]
        self.feedback = []
        self.requests = []
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
            self.executor.receive_go2_state(self.robot.observe(self.now))
        output = self.executor.tick(self.now)
        self.robot.command(self.now, output)
        if output.feedback is not None:
            self.feedback.append(output.feedback)
            self.node.receive_executor(output.feedback)
        if infer:
            request = self.node.begin_inference()
            if request is not None:
                self.requests.append((request, len(self.robot.velocities)))
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


class Go2ExecutorTests(ExecutorHarness, unittest.TestCase):
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

    def test_turn_limit_ends_a_rotating_step(self):
        self.setUp({"max_step_seconds": 1.0, "max_step_yaw_rad": 0.1})
        self.paths = [[(2, 44), (2, 36), (2, 32)]]  # Far left: rotate in place.
        self.start()
        self.pump_until(lambda: any(e["event"] == "step_finished" for e in self.executor.events))
        finished = [e for e in self.executor.events if e["event"] == "step_finished"][0]
        self.assertEqual(finished["reason"], "max_step_yaw")
        self.assertLessEqual(finished["yaw_rad"], 0.1 + self.config.motion.max_yaw_rate_rps * 0.05 + 1e-9)
        self.assertGreater(self.executor.pose.yaw, 0)  # Counter-clockwise toward the left path.
        self.assertTrue(all(v.vx == 0 for v in self.robot.velocities))

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

    def test_postures_wait_for_open_intake_and_stand_is_never_sent_after_failure(self):
        self.robot.accepting = False  # Intake stays closed: the stand request cannot be delivered.
        self.start()
        self.pump_until(lambda: self.executor.last_error is not None, limit=300)
        self.assertIn("standing state", self.executor.last_error)
        self.assertEqual(self.robot.postures, [])  # No posture was sent into the closed intake.
        self.robot.accepting = True
        for _ in range(20):
            self.pump()
        self.assertNotIn("stand", self.robot.postures)

    def test_intake_window_after_stand_is_waited_out(self):
        self.robot.awaiting_seconds = 0.4  # Longer than several control periods.
        self.start()
        self.pump_until(lambda: self.node.snapshot().cycle.completed >= 1)
        self.assertEqual(self.robot.postures, ["stand"])
        self.assertIsNone(self.executor.last_error)

    def test_start_is_blocked_while_go2_state_is_missing(self):
        self.go2_fresh = False
        for _ in range(15):
            self.pump()
        self.assertFalse(self.node.snapshot().start_ready)
        with self.assertRaisesRegex(ValueError, "executor"):
            self.start()


class GoalTests(ExecutorHarness, unittest.TestCase):
    """World-fixed goals: anchoring, per-cycle projection, turning and arrival."""

    def set_goal(self, **goal):
        self.node.receive_command(ControlCommand(type="goal", goal=GoalConfig(**goal)))
        for _ in range(3):
            self.pump()

    def test_metric_goal_is_reprojected_each_cycle_and_reached(self):
        self.set_goal(forward_m=1.0, left_m=0.0)
        self.start()
        self.pump_until(lambda: self.node.snapshot().goal_reached, limit=600)
        state = self.node.snapshot()
        self.assertFalse(state.inference.enabled)
        self.assertIn("Goal reached", state.status_message)
        anchored = [e for e in self.executor.events if e["event"] == "goal_anchored"]
        self.assertEqual(len(anchored), 1)
        self.assertAlmostEqual(anchored[0]["world"][0], 1.0)
        rows = [request.goal.y for request, _ in self.requests]
        self.assertGreater(len(rows), 2)
        self.assertEqual(rows, sorted(rows))  # The fixed goal moves down the image as the robot nears it.
        self.assertLess(math.dist((self.executor.pose.x, self.executor.pose.y), (1.0, 0.0)),
                        self.config.goal.arrival_tolerance_m + 0.2)
        self.pump_until(lambda: self.executor.phase == "idle" and self.robot.state == "down")

    def test_goal_behind_turns_in_place_before_the_first_observation(self):
        self.set_goal(forward_m=-1.0, left_m=0.5)
        self.start()
        self.pump_until(lambda: self.requests, limit=600)
        before_first_step = self.robot.velocities[:self.requests[0][1]]
        self.assertTrue(any(v.vyaw > 0 for v in before_first_step))
        self.assertTrue(all(v.vx == 0 for v in before_first_step))
        self.assertAlmostEqual(self.executor.pose.yaw, math.atan2(0.5, -1.0), delta=0.1)
        goal = self.requests[0][0].goal
        self.assertTrue(0 <= goal.x < 1 and 0 <= goal.y < 1)

    def test_image_goal_above_the_horizon_fails_without_moving(self):
        self.set_goal(x=0.5, y=0.1)
        self.start()
        self.pump_until(lambda: self.executor.last_error is not None)
        self.assertIn("not on visible ground", self.executor.last_error)
        self.assertFalse(any(v != VelocityCommand() for v in self.robot.velocities))
        self.assertFalse(self.node.snapshot().inference.enabled)

    def test_go2_pose_source_uses_reported_odometry_or_refuses_without_it(self):
        config = self.config.model_copy(update={"pose_source": "go2"})
        self.executor = self.make_executor(config)
        self.set_goal(forward_m=1.0, left_m=0.0)
        self.start()
        self.pump_until(lambda: self.node.snapshot().cycle.completed >= 1)
        self.assertAlmostEqual(self.executor.pose.x, self.robot.x, places=6)
        self.node.receive_command(ControlCommand(type="stop"))
        self.pump_until(lambda: self.executor.phase == "idle" and self.robot.state == "down")

        self.robot.publish_pose = False
        self.executor = self.make_executor(config)
        for _ in range(3):
            self.pump()
        self.start()
        self.pump_until(lambda: self.executor.last_error is not None)
        self.assertIn("no position/yaw", self.executor.last_error)


# robot_state as published by unitree-go2-zenoh-node with the odometry change
# (generated from that node's own controller), pinned as a contract fixture.
GO2_NODE_STATE = (
    '{"lifecycle":"running","robot_connected":true,"robot_state":{"state":"ready_stand","reason":null,'
    '"motion":"quiescent","state_machine_code":1,"state_machine_name":"balance_standing","mode":1,'
    '"mode_name":"balance_stand","velocity":[0.0,0.0,0.0],"yaw_speed":0.0,"position":[1.0,-2.0,0.3],'
    '"yaw":1.5,"stamp_sec":12,"stamp_nanosec":34},"accepting_commands":true,"requested_posture":null,'
    '"requested_velocity":null,"last_posture_action":null,"last_sdk_diagnostic":null,"last_node_error":null}')


class Go2ContractTests(unittest.TestCase):
    def test_go2_node_odometry_drives_the_go2_pose_source(self):
        state = Go2NodeState.model_validate_json(GO2_NODE_STATE)
        self.assertEqual((state.robot_state.position, state.robot_state.yaw), ((1.0, -2.0, 0.3), 1.5))
        executor = Go2StepExecutor(ExecutorConfig(camera_calibration="unused.json", allow_assumed_calibration=True,
                                                  pose_source="go2"),
                                   NodeConfig.model_validate(CONFIG).inference.stop_go, CAMERA, clock=lambda: 10.0)
        executor.receive_go2_state(state)
        executor.tick(10.0)
        self.assertEqual((executor.pose.x, executor.pose.y, executor.pose.yaw), (1.0, -2.0, 1.5))
        unknown = json.loads(GO2_NODE_STATE)
        unknown["robot_state"].update(state="unknown", motion="unknown", position=None, yaw=None)
        self.assertIsNone(Go2NodeState.model_validate(unknown).robot_state.position)


class ExecutorConfigTests(unittest.TestCase):
    def test_assumed_calibration_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "camera.json").write_text(json.dumps(CAMERA.as_json()), encoding="utf-8")
            config = ExecutorConfig(camera_calibration=str(directory / "camera.json"))
            with self.assertRaisesRegex(ValueError, "not verified"):
                load_camera(config)
            allowed = config.model_copy(update={"allow_assumed_calibration": True})
            self.assertEqual(load_camera(allowed), CAMERA)

    def test_invalid_rates_and_speeds_are_rejected(self):
        for changes in ({"control_frequency_hz": 4.0}, {"motion": {"forward_speed_mps": 2.0}},
                        {"shutdown_timeout_seconds": 5.0}, {"selection": "best"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ExecutorConfig.model_validate({"camera_calibration": "c.json", **changes})


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
