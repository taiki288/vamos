"""Node side of the stop/observe/infer/move protocol, and one end-to-end TCP run."""

from __future__ import annotations

from contextlib import ExitStack
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from PIL import Image
import zenoh

from cli import CliRuntime
from config import ControlCommand, NodeConfig, PlanarPose
from fake_go2 import SimulatedGo2, serve
from go2_executor import ExecutorConfig, ExecutorRuntime, Go2StepExecutor
from inference import InferenceOutput
from node import NodeRuntime, TransportState
from projection import GroundCamera
from stop_go_support import executor_feedback, report_stopped
from test_transport import BASE_CONFIG, image_bytes, wait_for_listener

CONFIG = {**BASE_CONFIG, "inference": {"device": "cpu", "frequency_hz": 20.0, "goal": {"x": 0.5, "y": 0.65},
                                       "stop_go": {"max_step_seconds": 0.3, "stop_settle_seconds": 0.1}}}


class StopGoNodeTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.config = NodeConfig.model_validate(CONFIG)
        self.state = TransportState(self.config, clock=lambda: self.now)
        self.state.model_loaded("cpu")
        self.inputs()
        report_stopped(self.state)

    def inputs(self, dt=0.0):
        self.now += dt
        self.state.receive_command(ControlCommand(type="heartbeat"))
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)

    def start(self):
        self.state.receive_command(ControlCommand(type="start"))

    def observe(self):
        """Stopped feedback, then a frame after the observation delay."""
        report_stopped(self.state)
        self.assertEqual(self.state.snapshot().cycle.phase, "waiting_image")
        self.inputs(self.config.inference.stop_go.observation_delay_seconds)
        report_stopped(self.state)
        return self.state.begin_inference()

    def test_start_ready_reflects_start_preconditions(self):
        ready = self.state.snapshot()
        self.assertTrue(ready.start_ready)
        self.assertEqual(ready.status_message, "Ready; send start.")
        self.start()
        self.assertFalse(self.state.snapshot().start_ready)
        self.state.receive_command(ControlCommand(type="stop"))
        self.now += 5.0  # Image, heartbeat and executor feedback all expire.
        self.assertFalse(self.state.snapshot().start_ready)
        self.state.receive_command(ControlCommand(type="heartbeat"))
        self.assertIn("fresh image", self.state.snapshot().status_message)

    def test_start_requires_fresh_stopped_executor(self):
        self.inputs(2.0)
        with self.assertRaisesRegex(ValueError, "executor"):
            self.start()
        self.state.receive_executor(executor_feedback(self.state.snapshot(), motion="settling"))
        with self.assertRaisesRegex(ValueError, "executor"):
            self.start()
        report_stopped(self.state)
        self.start()
        self.assertEqual(self.state.snapshot().cycle.phase, "waiting_stop")

    def test_only_frames_received_after_stop_and_delay_are_observed(self):
        self.start()
        report_stopped(self.state)
        self.assertEqual(self.state.snapshot().cycle.phase, "waiting_image")
        self.assertIsNone(self.state.begin_inference())  # Latest frame preceded stopped feedback.
        delay = self.config.inference.stop_go.observation_delay_seconds
        self.inputs(delay / 2)
        report_stopped(self.state)
        self.assertIsNone(self.state.begin_inference())  # Inside the camera latency window.
        self.inputs(delay)
        report_stopped(self.state)
        request = self.state.begin_inference()
        self.assertIsNotNone(request)
        self.assertEqual(request.cycle_id, self.state.snapshot().cycle.cycle_id)
        self.assertIsNone(self.state.begin_inference())  # No overlap.

    def test_cycle_completes_only_with_matching_stopped_feedback(self):
        self.start()
        request = self.observe()
        self.assertTrue(self.state.complete_inference(request, 0.1, Mock()))
        moving = self.state.snapshot()
        self.assertEqual(moving.cycle.phase, "moving")
        for wrong in (executor_feedback(moving, inference_id=999),
                      executor_feedback(moving, motion="moving", inference_id=request.inference_id)):
            self.state.receive_executor(wrong)
            self.assertEqual(self.state.snapshot().cycle.completed, 0)
        stale = executor_feedback(moving, inference_id=request.inference_id).model_copy(
            update={"control_epoch": moving.control_epoch - 1})
        self.state.receive_executor(stale)
        self.assertEqual(self.state.snapshot().cycle.completed, 0)
        report_stopped(self.state)
        after = self.state.snapshot()
        self.assertEqual((after.cycle.completed, after.cycle.phase), (1, "waiting_stop"))
        self.assertGreater(after.cycle.cycle_id, moving.cycle.cycle_id)
        self.assertIsNone(self.state.begin_inference())  # A new stop and frame are required.
        self.assertGreater(self.observe().frame.sequence, request.frame.sequence)

    def test_stop_during_inference_discards_result_and_requires_restart(self):
        self.start()
        request = self.observe()
        self.state.receive_command(ControlCommand(type="stop"))
        publish = Mock()
        self.assertFalse(self.state.complete_inference(request, 0.1, publish))
        publish.assert_not_called()
        self.inputs(0.5)
        report_stopped(self.state)
        self.assertIsNone(self.state.begin_inference())
        self.assertFalse(self.state.snapshot().inference.enabled)

    def test_motion_or_replaced_executor_during_inference_pauses(self):
        for change in ({"motion": "moving"}, {"executor_session_id": "another"}, {"pose": PlanarPose(x=0.1)}):
            with self.subTest(change=change):
                self.setUp()
                self.start()
                request = self.observe()
                feedback = executor_feedback(self.state.snapshot()).model_copy(update=change)
                self.state.receive_executor(feedback)
                self.assertFalse(self.state.snapshot().inference.enabled)
                self.assertFalse(self.state.complete_inference(request, 0.1, Mock()))

    def test_long_stopped_inference_is_accepted_within_its_budget(self):
        self.start()
        request = self.observe()
        for _ in range(24):  # 12 s with fresh inputs and stopped feedback.
            self.inputs(0.5)
            report_stopped(self.state)
        publish = Mock()
        self.assertTrue(self.state.complete_inference(request, 12.0, publish))
        publish.assert_called_once()

    def test_phase_timeouts_pause(self):
        for settings, phase_setup in (({"inference_timeout_seconds": 0.4}, "inferencing"),
                                      ({"stop_confirm_timeout_seconds": 0.4}, "waiting_stop")):
            with self.subTest(phase=phase_setup):
                self.config = NodeConfig.model_validate({**CONFIG, "inference": {
                    **CONFIG["inference"], "stop_go": {**CONFIG["inference"]["stop_go"], **settings}}})
                self.state = TransportState(self.config, clock=lambda: self.now)
                self.state.model_loaded("cpu")
                self.inputs()
                report_stopped(self.state)
                self.start()
                if phase_setup == "inferencing":
                    self.observe()
                    report = report_stopped
                else:
                    report = lambda state: state.receive_executor(executor_feedback(state.snapshot(), "settling"))
                for _ in range(6):
                    self.inputs(0.1)
                    report(self.state)
                self.assertFalse(self.state.snapshot().inference.enabled)
                self.assertIn("timed out", self.state.snapshot().status_message)


class StopGoNetworkTests(unittest.TestCase):
    def test_node_executor_and_simulated_go2_cycle_over_tcp_and_rest_on_stop(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
        server = zenoh.Config.from_json5(json.dumps({"mode": "peer", "listen": {"endpoints": [endpoint]},
                                                   "scouting": {"multicast": {"enabled": False}}}))
        client = json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]},
                             "scouting": {"multicast": {"enabled": False}}})
        config = NodeConfig.model_validate(CONFIG)
        robot = SimulatedGo2(posture_seconds=0.3)
        executor = Go2StepExecutor(ExecutorConfig(camera_calibration="unused.json", allow_assumed_calibration=True,
                                                  shutdown_timeout_seconds=11.0),
                                   config.inference.stop_go, GroundCamera(64, 48, 32, 32, 32, 24, 0.5))
        failures = Queue()
        stops = {name: Event() for name in ("node", "executor", "go2", "camera")}

        class Engine:
            device = "cpu"

            def infer(self, image, goal=None):
                if robot.velocity != (0.0, 0.0, 0.0) or robot.state not in ("ready_stand", "locomotion"):
                    failures.put(f"inference while {robot.state} with velocity {robot.velocity}")
                time.sleep(0.1)
                return InferenceOutput([[(32, 44), (32, 36), (32, 32)]], ["test"], "test", "cpu")

        def guarded(action):
            try:
                action()
            except Exception as error:
                failures.put(error)

        node = NodeRuntime(config, engine_loader=lambda _: Engine())
        threads = [Thread(target=guarded, args=(lambda: node.run(server, stops["node"]),), daemon=True)]
        threads[0].start()
        try:
            wait_for_listener(endpoint)
            with zenoh.open(zenoh.Config.from_json5(client)) as session, ExitStack() as resources:
                for target in (lambda: serve(robot, config.keys(), session, stops["go2"]),
                               lambda: ExecutorRuntime(executor, config).run(zenoh.Config.from_json5(client),
                                                                             stops["executor"])):
                    threads.append(Thread(target=guarded, args=(target,), daemon=True))
                    threads[-1].start()
                camera = resources.enter_context(session.declare_publisher("camera/front", encoding="image/jpeg"))

                def send_frames():
                    while not stops["camera"].wait(0.05):
                        camera.put(image_bytes())

                threads.append(Thread(target=send_frames, daemon=True))
                threads[-1].start()
                operator = CliRuntime(config, command_timeout=8, output=lambda _: None)
                with operator.connect(zenoh.Config.from_json5(client)):
                    self.assertIsNotNone(operator.cache.wait(lambda s: s.start_ready, 8))
                    operator.start()
                    self.assertIsNotNone(operator.cache.wait(lambda s: s.cycle.completed >= 2, 15),
                                         executor.snapshot())
                    self.assertGreater(robot.x, 0)
                    stopped = operator.stop()
                    self.assertEqual(stopped.cycle.executor_motion, "stopped")
                self.assertEqual(robot.state, "down")  # Rest confirmed before stop returned.
                stops["executor"].set()
                threads[2].join(5)
                self.assertTrue(executor.shutdown_complete)
                self.assertIsNone(executor.shutdown_error)
                stops["camera"].set()
                stops["go2"].set()
        finally:
            for event in stops.values():
                event.set()
            for thread in threads:
                thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(failures.empty(), list(failures.queue))


if __name__ == "__main__":
    unittest.main()
