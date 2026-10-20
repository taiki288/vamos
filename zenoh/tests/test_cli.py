from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from queue import Empty, Queue
import socket
import sys
from threading import Event, Thread
import time
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pydantic import ValidationError
import zenoh

from cli import CliRuntime, StateCache
from config import ControlCommand, GoalConfig, NodeConfig, TrajectoryResult
from inference import InferenceOutput, VAMOSEngine
from node import NodeRuntime, TransportState
from stream_simulator import StreamFollower
from test_transport import BASE_CONFIG, image_bytes, wait_for_listener
from test_stream_simulator import CAMERA


CONFIG = {**BASE_CONFIG, "inference": {"goal": {"x": 0.5, "y": 0.6}, "device": "cpu", "frequency_hz": 20.0}}


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.config = NodeConfig.model_validate(CONFIG)
        self.state = TransportState(self.config, clock=lambda: self.now)
        self.state.model_loaded("cpu")
        self.refresh()

    def refresh(self):
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.state.receive_command(ControlCommand(type="heartbeat"))

    def test_manual_start_is_default_and_stop_remains_unconditional(self):
        self.assertIsNone(self.state.begin_inference())
        self.state.receive_command(ControlCommand(type="start"))
        self.assertIsNotNone(self.state.begin_inference())
        self.now += 100
        self.state.receive_command(ControlCommand(type="stop", request_id="stop", node_session_id="old-node"))
        self.assertTrue(self.state.snapshot().last_control_ack.accepted)
        self.assertFalse(self.state.snapshot().inference.enabled)

    def test_stale_inputs_latch_pause_until_explicit_start(self):
        self.state.receive_command(ControlCommand(type="start"))
        self.now += 2
        self.assertFalse(self.state.snapshot().inference.enabled)
        self.refresh()
        self.assertIsNone(self.state.begin_inference())
        self.state.receive_command(ControlCommand(type="start"))
        self.assertIsNotNone(self.state.begin_inference())

    def test_stop_drain_goal_change_and_resume_cannot_publish_old_result(self):
        self.state.receive_command(ControlCommand(type="start"))
        pending = self.state.begin_inference()
        self.state.receive_command(ControlCommand(type="stop"))
        goal = GoalConfig(x=0.7, y=0.8, instruction="New goal")
        with self.assertRaises(ValueError):
            self.state.receive_command(ControlCommand(type="goal", goal=goal, request_id="too-early"))
        with self.assertRaises(ValueError):
            self.state.receive_command(ControlCommand(type="start"))
        publish = Mock()
        self.assertFalse(self.state.complete_inference(pending, 0.1, publish))
        publish.assert_not_called()
        self.state.receive_command(ControlCommand(type="goal", goal=goal))
        self.state.receive_command(ControlCommand(type="start"))
        next_request = self.state.begin_inference()
        self.assertEqual(next_request.goal, goal)
        self.assertEqual(next_request.goal_revision, 1)
        self.assertNotEqual(next_request.epoch, pending.epoch)
        self.assertNotEqual(pending.goal, goal)

    def test_correlated_ack_is_not_overwritten_by_heartbeat_and_retry_is_idempotent(self):
        command = ControlCommand(type="goal", goal=GoalConfig(x=0.2, y=0.6), request_id="one")
        self.state.receive_command(command)
        self.state.receive_command(ControlCommand(type="heartbeat"))
        self.state.receive_command(command)
        state = self.state.snapshot()
        self.assertEqual(state.goal_revision, 1)
        self.assertEqual(state.last_control_ack.request_id, "one")
        self.assertTrue(state.last_control_ack.accepted)
        with self.assertRaises(ValueError):
            self.state.receive_command(ControlCommand(type="stop", request_id="one"))
        self.assertFalse(self.state.snapshot().last_control_ack.accepted)

    def test_wrong_session_revision_and_running_goal_are_rejected_with_ack(self):
        for command in (
            ControlCommand(type="start", node_session_id="old", request_id="session"),
            ControlCommand(type="goal", goal=GoalConfig(x=0.2, y=0.6), expected_goal_revision=9, request_id="revision"),
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.state.receive_command(command)
            self.assertFalse(self.state.snapshot().last_control_ack.accepted)
        self.state.receive_command(ControlCommand(type="start"))
        with self.assertRaises(ValueError):
            self.state.receive_command(ControlCommand(type="goal", goal=GoalConfig(x=0.2, y=0.6)))

    def test_goal_override_reaches_model_prompt_without_mutating_load_config(self):
        generate = Mock(return_value=([[(1, 2), (20, 30)]], ["valid"], None))
        engine = VAMOSEngine(None, None, "cpu", generate, self.config.inference)
        goal = GoalConfig(x=0.25, y=0.75, instruction="Turn left")
        output = engine.infer(Image.new("RGB", (64, 48)), goal=goal)
        self.assertIn("<loc0256>", output.prompt)
        self.assertIn("<loc0768>", output.prompt)
        self.assertIn("Turn left", output.prompt)
        self.assertEqual(self.config.inference.goal, GoalConfig(x=0.5, y=0.6))

    def test_malformed_command_fields_are_rejected(self):
        for data in ({"type": "goal"}, {"type": "start", "goal": {"x": 0.5, "y": 0.5}},
                     {"type": "goal", "goal": {"x": 1.1, "y": 0.5}},
                     {"type": "heartbeat", "request_id": "bad"},
                     {"type": "stop", "expected_goal_revision": 1}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ControlCommand.model_validate(data)

    def test_old_epoch_result_and_reordered_state_cannot_restore_following(self):
        self.state.receive_command(ControlCommand(type="start"))
        old_state = self.state.snapshot()
        follower = StreamFollower(CAMERA, clock=lambda: self.now, wall_clock=lambda: 1000)
        follower.receive_state(old_state)
        self.state.receive_command(ControlCommand(type="stop"))
        follower.receive_state(self.state.snapshot())
        follower.receive_state(old_state)
        self.assertEqual(follower.node_state.control_epoch, self.state.snapshot().control_epoch)
        self.state.receive_command(ControlCommand(type="start"))
        follower.receive_state(self.state.snapshot())
        old_result = TrajectoryResult(inference_id=1, frame_sequence=1, image_width=64, image_height=48,
            goal_pixels=(32, 32), prompt="old", trajectories=[[(32, 44), (32, 32)]], generated_texts=["old"],
            duration_seconds=0.1, source_age_seconds=0.1, device="cpu", node_session_id=old_state.node_session_id,
            control_epoch=old_state.control_epoch, goal_revision=old_state.goal_revision)
        self.assertFalse(follower.receive_result(old_result))
        self.assertEqual(follower.rejected["obsolete_control_result"], 1)

    def test_state_cache_does_not_return_stale_observations(self):
        cache = StateCache(1, clock=lambda: self.now)
        cache.store(self.state.snapshot())
        self.assertIsNotNone(cache.snapshot())
        self.now += 2
        self.assertIsNone(cache.snapshot())

    def test_unacknowledged_stop_is_reported_as_timeout(self):
        cli = CliRuntime(self.config, heartbeat_interval=0.05, state_timeout=0.2, command_timeout=0.3,
                         output=lambda _: None)
        cli.cache.store(self.state.snapshot())
        cli._publisher = Mock()
        with self.assertRaises(TimeoutError):
            cli.stop()
        self.assertTrue(cli._publisher.put.called)

    def test_source_failure_does_not_block_stop_to_connected_node(self):
        processes = Mock()
        processes.check_required.side_effect = RuntimeError("source exited")
        cli = CliRuntime(self.config, processes=processes, output=lambda _: None)
        cli.cache.store(self.state.snapshot())
        cli._publisher = Mock()

        def publish(payload):
            self.state.receive_command(ControlCommand.model_validate_json(payload))
            cli.cache.store(self.state.snapshot())

        cli._publisher.put.side_effect = publish
        with self.assertRaisesRegex(RuntimeError, "source exited"):
            cli.state()
        self.assertFalse(cli.stop().inference.enabled)

    def test_start_after_input_loss_waits_for_pending_inference(self):
        self.state.receive_command(ControlCommand(type="start"))
        pending = self.state.begin_inference()
        self.now += 2
        self.assertFalse(self.state.snapshot().inference.enabled)
        self.refresh()
        waiting = Event()
        cli = CliRuntime(self.config, command_timeout=3, output=lambda _: waiting.set())
        cli.cache.store(self.state.snapshot())

        def publish(payload):
            self.state.receive_command(ControlCommand.model_validate_json(payload))
            cli.cache.store(self.state.snapshot())

        cli._publisher = Mock()
        cli._publisher.put.side_effect = publish
        finished = Queue()

        def resume():
            try:
                finished.put(cli.start())
            except Exception as error:
                finished.put(error)

        waiter = Thread(target=resume, daemon=True)
        waiter.start()
        try:
            self.assertTrue(waiting.wait(1))
            self.assertTrue(finished.empty())
            cli._publisher.put.assert_not_called()
            self.assertFalse(self.state.complete_inference(pending, 2, Mock()))
            cli.cache.store(self.state.snapshot())
            update = finished.get(timeout=2)
            self.assertNotIsInstance(update, Exception)
            self.assertTrue(update.inference.enabled)
        finally:
            waiter.join(3)
        self.assertFalse(waiter.is_alive())


class CliNetworkTests(unittest.TestCase):
    def test_real_cli_start_stop_change_goal_recover_and_exit(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
        config = NodeConfig.model_validate(CONFIG)
        server = zenoh.Config.from_json5(json.dumps({"mode": "peer", "listen": {"endpoints": [endpoint]},
                                                    "scouting": {"multicast": {"enabled": False}}}))
        client_json = json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]},
                                 "scouting": {"multicast": {"enabled": False}}})
        results, commands, failures = Queue(), Queue(), Queue()
        stop_node, stop_camera = Event(), Event()
        allow_images = Event()
        allow_images.set()

        class Engine:
            device = "cpu"

            def infer(self, image, goal=None):
                return InferenceOutput([[(32, 44), goal.pixels(*image.size)]], ["test"], goal.instruction, "cpu")

        node = NodeRuntime(config, engine_loader=lambda _: Engine())

        def node_run():
            try:
                node.run(server, stop_node)
            except Exception as error:
                failures.put(error)

        thread = Thread(target=node_run, daemon=True)
        thread.start()
        sender = None
        try:
            wait_for_listener(endpoint)
            with zenoh.open(zenoh.Config.from_json5(client_json)) as session, ExitStack() as resources:
                resources.enter_context(session.declare_subscriber("vamos/trajectories", lambda s: results.put(
                    TrajectoryResult.model_validate_json(s.payload.to_bytes()))))
                resources.enter_context(session.declare_subscriber("unitree/go2/command", lambda s: commands.put(s.payload.to_bytes())))
                camera = resources.enter_context(session.declare_publisher("camera/front", encoding="image/jpeg"))

                def send_images():
                    while not stop_camera.is_set():
                        if allow_images.is_set():
                            camera.put(image_bytes())
                        stop_camera.wait(0.03)

                sender = Thread(target=send_images, daemon=True)
                sender.start()
                resources.callback(lambda: (stop_camera.set(), sender.join(2)))
                output = []
                cli = CliRuntime(config, command_timeout=5, heartbeat_interval=0.05, output=output.append)
                with cli.connect(zenoh.Config.from_json5(client_json)):
                    self.assertFalse(cli.state().inference.enabled)
                    self.assertIsNotNone(cli.cache.wait(lambda s: s.camera.fresh and s.inference.model_loaded, 3))
                    cli.execute("start")
                    first = results.get(timeout=3)
                    cli.execute("goal 0.25 0.75")
                    self.assertFalse(cli.state().inference.enabled)
                    cli.execute("ins 日本語の指示")
                    cli.execute("start")
                    deadline = time.monotonic() + 3
                    latest = None
                    while time.monotonic() < deadline:
                        try:
                            result = results.get(timeout=0.1)
                        except Empty:
                            continue
                        if result.goal_revision == 2:
                            latest = result
                            break
                    self.assertIsNotNone(latest)
                    self.assertEqual(latest.goal_pixels, (16, 36))
                    self.assertEqual(latest.prompt, "日本語の指示")
                    self.assertNotEqual(latest.control_epoch, first.control_epoch)
                    # Heartbeat loss requires explicit start even after recovery.
                    cli._stop_heartbeat.set()
                    self.assertIsNotNone(cli.cache.wait(lambda s: not s.inference.enabled, 3))
                    session.put("vamos/command", '{"type":"heartbeat"}', encoding="application/json")
                    self.assertIsNotNone(cli.cache.wait(lambda s: s.cli_connected, 2))
                    self.assertFalse(cli.state().inference.enabled)
                    self.assertFalse(cli.execute("quit"))
                    with self.assertRaises(ValueError):
                        cli.execute("goal 2 0.5")
                self.assertFalse(node.state.snapshot().inference.enabled)
                self.assertFalse(node.state.snapshot().inference.active)
                self.assertTrue(commands.empty())
        finally:
            stop_camera.set()
            if sender:
                sender.join(2)
            stop_node.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures.empty(), list(failures.queue))


if __name__ == "__main__":
    unittest.main()
