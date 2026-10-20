from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from queue import Queue
import socket
import sys
from threading import Event, Thread
import time
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import zenoh

from cli import CliRuntime
from config import CaptureContext, CapturePose, ControlCommand, ExecutorState, NodeConfig, NodeState, TrajectoryResult
from controller import VelocityCommand
from inference import InferenceOutput
from node import NodeRuntime, TransportState
from stop_go import StopGoFollower
from stream_simulator import run_session
from test_stream_simulator import CAMERA
from test_transport import BASE_CONFIG, image_bytes, wait_for_listener


CONFIG = {**BASE_CONFIG, "inference": {"execution_mode": "stop_go", "device": "cpu",
    "frequency_hz": 20.0, "goal": {"x": 0.5, "y": 0.65},
    "stop_go": {"max_step_seconds": 0.3, "stop_settle_seconds": 0.1}}}


class StopGoTests(unittest.TestCase):
    def setUp(self):
        self.now, self.wall, self.frame_id = 100.0, 1000.0, 0
        self.config = NodeConfig.model_validate(CONFIG)
        self.state = TransportState(self.config, clock=lambda: self.now, wall_clock=lambda: self.wall)
        self.state.model_loaded("cpu")
        self.follower = StopGoFollower(CAMERA, self.config.inference.stop_go,
                                      clock=lambda: self.now, wall_clock=lambda: self.wall)
        for _ in range(4):
            self.pump()

    def pump(self, frame=True):
        self.now += 0.05
        self.wall += 0.05
        self.state.receive_command(ControlCommand(type="heartbeat"))
        if frame:
            self.frame_id += 1
            capture = CaptureContext(source_id="static-test", frame_id=self.frame_id,
                                     captured_at_unix_seconds=self.wall)
            self.state.receive_frame(Image.new("RGB", (64, 48)), self.now, capture)
        self.follower.receive_state(self.state.snapshot())
        self.follower.tick()
        self.state.receive_executor(self.follower.feedback())

    def start(self):
        command = ControlCommand(type="start")
        self.state.receive_command(command)
        self.follower.receive_control(command)

    def next_request(self):
        for _ in range(20):
            self.pump()
            request = self.state.begin_inference()
            if request is not None:
                return request
        self.fail(str(self.state.snapshot()))

    def result(self, request):
        return TrajectoryResult(node_session_id=self.state.snapshot().node_session_id,
            generated_at_unix_seconds=self.wall,
            control_epoch=request.epoch, goal_revision=request.goal_revision, cycle_id=request.cycle_id,
            inference_id=request.inference_id, frame_sequence=request.frame.sequence,
            image_width=64, image_height=48, goal_pixels=request.goal.pixels(64, 48), prompt="test",
            trajectories=[[(32, 44), (32, 36), (32, 32)]], generated_texts=["test"], device="cpu",
            duration_seconds=0.1, source_age_seconds=self.now-request.frame.received_at, capture=request.frame.capture)

    def publish(self, request):
        for _ in range(3):
            before = self.follower.pose
            self.pump()
            self.assertEqual(self.follower.pose, before)
            self.assertEqual(self.follower.command, VelocityCommand())
        result = self.result(request)
        self.assertTrue(self.state.complete_inference(request, 0.1, lambda: self.follower.receive_result(result)))
        return result

    def test_two_cycles_wait_for_motion_and_use_new_stop_pose(self):
        self.start()
        first = self.next_request()
        self.publish(first)
        for _ in range(30):
            if self.state.snapshot().cycle.completed:
                break
            self.assertIsNone(self.state.begin_inference(), "Inference began before movement/stop completed")
            self.pump()
        self.assertEqual(self.state.snapshot().cycle.completed, 1)
        pose = self.follower.pose
        self.assertGreater(pose.x, 0)
        self.assertLessEqual(self.follower._step_distance, 0.2 + 0.01)
        second = self.next_request()
        self.assertGreater(second.frame.sequence, first.frame.sequence)
        self.assertGreater(second.cycle_id, first.cycle_id)
        self.assertGreater(second.frame.capture.captured_at_unix_seconds,
                           self.state.snapshot().cycle.capture_after_unix_seconds)
        self.publish(second)
        self.pump()
        started = [event for event in self.follower.events if event["event"] == "step_started"]
        self.assertEqual(started[-1]["pose"], asdict(pose))
        self.assertEqual(started[-1]["anchor_source"], "simulated_stop_pose")
        self.assertGreater(self.follower.pose.x, pose.x)

    def test_pre_stop_and_unannotated_images_cannot_start_inference(self):
        self.start()
        for _ in range(5):
            self.pump()
            if self.state.snapshot().cycle.phase == "waiting_image":
                break
        self.assertEqual(self.state.snapshot().cycle.phase, "waiting_image")
        self.assertIsNone(self.state.begin_inference())  # Latest frame preceded stop confirmation.
        old = CaptureContext(source_id="static-test", frame_id=999, captured_at_unix_seconds=self.wall - 1)
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now, old)
        self.assertIsNone(self.state.begin_inference())
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.assertIsNone(self.state.begin_inference())
        self.pump()
        self.assertIsNotNone(self.state.begin_inference())

    def test_stop_during_inference_discards_result_and_requires_restart(self):
        self.start()
        request = self.next_request()
        command = ControlCommand(type="stop")
        self.state.receive_command(command)
        self.follower.receive_control(command)
        publish = Mock()
        self.assertFalse(self.state.complete_inference(request, 0.1, publish))
        publish.assert_not_called()
        for _ in range(8):
            self.pump()
            self.assertIsNone(self.state.begin_inference())
        self.assertFalse(self.follower.receive_result(self.result(request)))
        self.assertEqual(self.follower.command, VelocityCommand())

    def test_stop_during_motion_is_immediate_and_old_feedback_cannot_advance_cycle(self):
        self.start()
        result = self.publish(self.next_request())
        self.pump()
        self.assertGreater(self.follower.command.vx, 0)
        old_feedback = self.follower.feedback()
        self.state.receive_command(ControlCommand(type="stop"))
        self.follower.receive_control(ControlCommand(type="stop"))
        self.assertEqual(self.follower.command, VelocityCommand())
        pose = self.follower.pose
        self.state.receive_executor(old_feedback.model_copy(update={"motion": "stopped", "inference_id": result.inference_id}))
        for _ in range(8):
            self.pump()
        self.assertEqual(self.follower.pose, pose)
        self.assertEqual(self.state.snapshot().cycle.completed, 0)
        self.assertFalse(self.follower.receive_result(result))
        self.assertTrue(self.state.snapshot().cycle.executor_fresh)

    def test_wrong_result_and_missing_completion_cannot_advance_cycle(self):
        self.start()
        request = self.next_request()
        self.publish(request)
        self.pump()
        feedback = self.follower.feedback().model_copy(update={"motion": "stopped", "inference_id": 999})
        self.state.receive_executor(feedback)
        self.assertEqual(self.state.snapshot().cycle.phase, "moving")
        self.assertEqual(self.state.snapshot().cycle.completed, 0)
        self.assertFalse(self.follower.receive_result(self.result(request)))
        self.now += 1.1
        self.wall += 1.1
        self.state.receive_command(ControlCommand(type="heartbeat"))
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.assertFalse(self.state.snapshot().inference.enabled)
        self.assertIn("Executor", self.state.snapshot().status_message)

    def test_motion_during_inference_or_replaced_executor_invalidates_result(self):
        for change in ({"motion": "moving"}, {"executor_session_id": "another"}, {"pose": CapturePose(x=0.1)}):
            with self.subTest(change=change):
                self.setUp()
                self.start()
                request = self.next_request()
                self.pump()
                feedback = self.follower.feedback().model_copy(update=change)
                self.state.receive_executor(feedback)
                self.assertFalse(self.state.snapshot().inference.enabled)
                self.assertFalse(self.state.complete_inference(request, 0.1, Mock()))

    def test_start_requires_executor_and_view_exposes_cycle_phase(self):
        self.now += 2
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.state.receive_command(ControlCommand(type="heartbeat"))
        with self.assertRaisesRegex(ValueError, "simulator"):
            self.start()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from stream_simulator import StreamView
        view = StreamView(self.follower)
        try:
            view.draw()
            self.assertIn("Cycle:", view.info.get_text())
            self.assertIn("View does not move", view.info.get_text())
        finally:
            plt.close(view.figure)

    def test_distance_and_turn_limits_stop_before_next_observation(self):
        for setting, path, measured, bound, tick_error in (
            ({"max_step_distance_m": 0.025}, [(32, 44), (32, 36), (32, 32)], "_step_distance", 0.025, 0.01),
            ({"max_step_yaw_rad": 0.1}, [(2, 44), (2, 36), (2, 32)], "_step_yaw", 0.1, 0.04),
        ):
            with self.subTest(setting=setting):
                self.setUp()
                self.config = NodeConfig.model_validate({**CONFIG, "inference": {**CONFIG["inference"],
                    "stop_go": {**setting, "max_step_seconds": 1.0, "stop_settle_seconds": 0.2}}})
                self.state = TransportState(self.config, clock=lambda: self.now, wall_clock=lambda: self.wall)
                self.state.model_loaded("cpu")
                self.follower = StopGoFollower(CAMERA, self.config.inference.stop_go,
                    clock=lambda: self.now, wall_clock=lambda: self.wall)
                for _ in range(5):
                    self.pump()
                self.start()
                request = self.next_request()
                result = self.result(request).model_copy(update={"trajectories": [path]})
                self.assertTrue(self.state.complete_inference(request, 0.0, lambda: self.follower.receive_result(result)))
                for _ in range(20):
                    self.pump()
                    if self.follower._step_finished:
                        break
                self.assertTrue(self.follower._step_finished)
                self.assertGreaterEqual(getattr(self.follower, measured), bound)
                self.assertLessEqual(getattr(self.follower, measured), bound + tick_error + 1e-9)
                self.assertEqual(self.follower.feedback().motion, "settling")
                self.assertEqual(self.follower.command, VelocityCommand())
                self.assertEqual(self.state.snapshot().cycle.completed, 0)
                self.assertIsNone(self.state.begin_inference())

    def test_delayed_result_delivery_shortens_remaining_execution_time(self):
        self.start()
        request = self.next_request()
        result = self.result(request)
        for _ in range(57):
            self.pump()
        self.assertTrue(self.state.complete_inference(request, 0.1, lambda: self.follower.receive_result(result)))
        for _ in range(12):
            self.pump()
            if self.state.snapshot().cycle.completed:
                break
        self.assertEqual(self.state.snapshot().cycle.completed, 1)
        self.assertGreater(self.follower.pose.x, 0)
        self.assertLess(self.follower.pose.x, 0.04)
        self.assertIsNone(self.follower._error)

    def test_twelve_second_inference_uses_separate_stopped_budget(self):
        self.start()
        request = self.next_request()
        pose = self.follower.pose
        for _ in range(240):
            self.pump()
            self.assertEqual(self.follower.pose, pose)
            self.assertIsNone(self.state.begin_inference())
        result = self.result(request).model_copy(update={"duration_seconds": 12.0})
        self.assertGreater(result.source_age_seconds, 5)
        self.assertTrue(self.state.complete_inference(request, 12.0, lambda: self.follower.receive_result(result)))
        for _ in range(20):
            self.pump()
            if self.state.snapshot().cycle.completed:
                break
        self.assertEqual(self.state.snapshot().cycle.completed, 1)
        self.assertGreater(self.follower.pose.x, pose.x)
        self.assertEqual(self.follower.expiry_origin_unix_seconds, result.generated_at_unix_seconds)
        self.assertEqual(self.follower.capture, request.frame.capture)

    def test_stopped_inference_still_has_a_timeout(self):
        config = NodeConfig.model_validate({**CONFIG, "inference": {**CONFIG["inference"],
            "stop_go": {"inference_timeout_seconds": 0.4}}})
        self.state.config = config
        self.follower.settings = config.inference.stop_go
        self.start()
        request = self.next_request()
        for _ in range(12):
            self.pump()
        self.assertFalse(self.state.snapshot().inference.enabled)
        self.assertFalse(self.state.complete_inference(request, 0.6, Mock()))
        self.assertEqual(self.follower.command, VelocityCommand())

    def test_missing_or_stale_generation_time_cannot_start_motion(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.setUp()
                self.start()
                request = self.next_request()
                result = self.result(request)
                if missing:
                    result = result.model_copy(update={"generated_at_unix_seconds": None})
                else:
                    for _ in range(65):
                        self.pump()
                self.assertTrue(self.state.complete_inference(request, 0.1, lambda: self.follower.receive_result(result)))
                self.pump()
                self.assertFalse(self.state.snapshot().inference.enabled)
                self.assertEqual(self.follower.command, VelocityCommand())


class StopGoNetworkTests(unittest.TestCase):
    def test_real_tcp_cycles_never_infer_while_moving_and_stop_is_confirmed(self):
        self.run_cycles(0.12, 3)

    def test_twelve_second_inference_continues_for_two_real_tcp_cycles(self):
        self.run_cycles(12.0, 2)

    def run_cycles(self, inference_delay, cycles):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
        server = zenoh.Config.from_json5(json.dumps({"mode": "peer", "listen": {"endpoints": [endpoint]},
                                                   "scouting": {"multicast": {"enabled": False}}}))
        client = json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]},
                             "scouting": {"multicast": {"enabled": False}}})
        config = NodeConfig.model_validate(CONFIG)
        follower = StopGoFollower(CAMERA, config.inference.stop_go)
        failures, robot_commands = Queue(), Queue()
        stop_node, stop_follower, stop_camera = Event(), Event(), Event()
        phases = set()

        class Engine:
            device = "cpu"
            def infer(self, image, goal=None):
                before = follower.pose
                if follower.command != VelocityCommand():
                    failures.put("Inference started while moving")
                time.sleep(inference_delay)
                if follower.pose != before:
                    failures.put("Pose changed during inference")
                return InferenceOutput([[(32, 44), (32, 36), (32, 32)]], ["test"], "test", "cpu")

        node = NodeRuntime(config, engine_loader=lambda _: Engine())
        def guarded(action):
            try:
                action()
            except Exception as error:
                failures.put(error)
        node_thread = Thread(target=lambda: guarded(lambda: node.run(server, stop_node)), daemon=True)
        follow_thread = Thread(target=lambda: guarded(lambda: run_session(follower, config.keys(),
            zenoh.Config.from_json5(client), stop_follower, headless=True,
            duration=max(15, cycles * (inference_delay + 2) + 15))), daemon=True)
        node_thread.start()
        sender = None
        try:
            wait_for_listener(endpoint)
            follow_thread.start()
            with zenoh.open(zenoh.Config.from_json5(client)) as session, ExitStack() as resources:
                resources.enter_context(session.declare_subscriber("unitree/go2/command", lambda s: robot_commands.put(s)))
                resources.enter_context(session.declare_subscriber("vamos/state",
                    lambda s: phases.add(NodeState.model_validate_json(s.payload.to_bytes()).cycle.phase)))
                camera = resources.enter_context(session.declare_publisher("camera/front", encoding="image/jpeg"))
                def send_frames():
                    index = 0
                    while not stop_camera.is_set():
                        index += 1
                        context = CaptureContext(source_id="tcp-cycle", frame_id=index,
                                                 captured_at_unix_seconds=time.time())
                        camera.put(image_bytes(), attachment=context.model_dump_json().encode())
                        stop_camera.wait(0.05)
                sender = Thread(target=send_frames, daemon=True)
                sender.start()
                resources.callback(lambda: (stop_camera.set(), sender.join(2)))
                operator = CliRuntime(config, command_timeout=max(5, inference_delay + 3), output=lambda _: None)
                with operator.connect(zenoh.Config.from_json5(client)):
                    self.assertIsNotNone(operator.cache.wait(lambda s: s.camera.fresh and s.inference.model_loaded
                        and s.cycle.executor_fresh and s.cycle.executor_motion == "stopped", 5))
                    operator.start()
                    state = operator.cache.wait(lambda s: s.cycle.completed >= cycles,
                                                max(10, cycles * (inference_delay + 2) + 5))
                    self.assertIsNotNone(state, node.state.snapshot())
                    stopped = operator.stop()
                    self.assertEqual(stopped.cycle.executor_motion, "stopped")
                    pose = follower.pose
                    time.sleep(0.2)
                    self.assertEqual(follower.pose, pose)
                    self.assertFalse(node.state.snapshot().inference.active)
                    self.assertTrue(robot_commands.empty())
                    # Brief waiting phases may fall between periodic snapshots;
                    # their capture barriers are checked with the deterministic clock above.
                    self.assertTrue({"inferencing", "moving"} <= phases, phases)
        finally:
            stop_camera.set()
            if sender:
                sender.join(2)
            stop_follower.set()
            stop_node.set()
            node_thread.join(3)
            if follow_thread.ident is not None:
                follow_thread.join(3)
        self.assertFalse(node_thread.is_alive())
        self.assertFalse(follow_thread.is_alive())
        self.assertTrue(failures.empty(), list(failures.queue))


if __name__ == "__main__":
    unittest.main()
