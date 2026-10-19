from __future__ import annotations

from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
from queue import Queue
import socket
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import zenoh

from config import CaptureContext, ControlCommand, NodeConfig, TrajectoryResult
from controller import VelocityCommand
from inference import InferenceOutput
from node import NodeRuntime, TransportState
from projection import GroundCamera
from stream_simulator import StreamFollower, StreamView, run_session
from test_transport import BASE_CONFIG, image_bytes, sample, wait_for_listener


CAMERA = GroundCamera(64, 48, 32, 32, 32, 24, 0.5)
CONFIG = {**BASE_CONFIG, "inference": {"goal": {"x": 0.5, "y": 0.65},
                                     "device": "cpu", "frequency_hz": 20.0}}


class StreamFollowerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.wall = 1000.0
        self.follower = StreamFollower(CAMERA, clock=lambda: self.now, wall_clock=lambda: self.wall)
        config = NodeConfig.model_validate(CONFIG)
        self.transport = TransportState(config, clock=lambda: self.now)
        self.transport.model_loaded("cpu")
        self.refresh_state()
        self.wall += 0.01

    def refresh_state(self):
        self.transport.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.transport.receive_command(ControlCommand(type="heartbeat"))
        self.follower.receive_state(self.transport.snapshot())

    def result(self, frame_id=1, **changes):
        context = CaptureContext(source_id="fixture", frame_id=frame_id,
                                 captured_at_unix_seconds=self.wall)
        data = dict(inference_id=frame_id, frame_sequence=frame_id, image_width=64, image_height=48,
                    goal_pixels=(32, 32), prompt="test", trajectories=[[(32, 44), (32, 36), (32, 32)]],
                    generated_texts=["test"], duration_seconds=0.1, source_age_seconds=0.1,
                    device="cpu", capture=context)
        return TrajectoryResult(**{**data, **changes})

    def test_updates_stay_anchored_to_capture_pose_while_robot_moves(self):
        self.assertTrue(self.follower.receive_result(self.result()))
        original_path = self.follower.path
        for _ in range(10):
            self.now += 0.05
            self.wall += 0.05
            self.refresh_state()
            self.follower.tick()
        self.assertGreater(self.follower.pose.x, 0)
        self.assertTrue(self.follower.receive_result(self.result(2)))
        self.assertEqual(self.follower.path, original_path)
        self.assertNotEqual(self.follower.events[-1]["robot_pose"], self.follower.events[-1]["capture_pose"])

    def test_expiry_uses_source_age_and_duplicates_do_not_refresh_it(self):
        result = self.result(source_age_seconds=2.8)
        self.assertTrue(self.follower.receive_result(result))
        self.now += 0.15
        self.wall += 0.15
        self.assertFalse(self.follower.receive_result(result))
        self.assertEqual(self.follower.rejected["duplicate_or_reordered"], 1)
        self.now += 0.1
        self.wall += 0.1
        self.refresh_state()
        self.follower.tick()
        self.assertEqual(self.follower.status, "stale_path")
        self.assertEqual(self.follower.command, VelocityCommand())

    def test_node_loss_and_remote_stop_halt_motion(self):
        self.follower.receive_result(self.result())
        self.follower.tick()
        self.assertGreater(self.follower.command.vx, 0)
        self.now += 1.1
        self.follower.tick()
        self.assertEqual(self.follower.status, "stale_node")
        self.assertEqual(self.follower.command, VelocityCommand())
        self.refresh_state()
        self.follower.receive_result(self.result(2))
        self.follower.receive_control(ControlCommand(type="stop"))
        self.follower.receive_control(ControlCommand(type="heartbeat"))
        self.assertFalse(self.follower.receive_result(self.result(3)))
        self.assertEqual(self.follower.command, VelocityCommand())
        old = self.result(4)
        self.follower.receive_control(ControlCommand(type="start"))
        self.assertFalse(self.follower.receive_result(old))
        self.wall += 0.1
        self.assertTrue(self.follower.receive_result(self.result(5)))

    def test_pause_resume_requires_new_frame(self):
        old = self.result()
        self.follower.receive_result(old)
        self.follower.toggle_pause()
        pose = self.follower.pose
        self.follower.tick()
        self.assertEqual(self.follower.pose, pose)
        self.follower.toggle_pause()
        self.assertFalse(self.follower.receive_result(old))
        self.wall += 0.1
        self.assertTrue(self.follower.receive_result(self.result(2)))

    def test_missing_pose_bad_projection_and_source_change_stop(self):
        self.assertFalse(self.follower.receive_result(self.result(capture=None)))
        self.assertEqual(self.follower.status, "missing_capture_pose")
        self.assertTrue(self.follower.receive_result(self.result(2)))
        self.assertFalse(self.follower.receive_result(self.result(3, trajectories=[[(32, 10), (32, 12)]])))
        self.assertEqual(self.follower.status, "invalid_projection")
        self.assertFalse(self.follower.path)
        changed = self.result(4).model_copy(update={"capture": self.result(4).capture.model_copy(update={"source_id": "other"})})
        self.assertFalse(self.follower.receive_result(changed))
        self.assertEqual(self.follower.status, "source_changed_reset_required")

    def test_future_old_and_wrong_sized_results_are_rejected(self):
        for stamp in (self.wall + 1, self.wall - 5):
            result = self.result().model_copy(update={"capture": self.result().capture.model_copy(
                update={"captured_at_unix_seconds": stamp})})
            self.assertFalse(self.follower.receive_result(result))
        self.assertFalse(self.follower.receive_result(self.result(image_width=128)))
        self.assertEqual(self.follower.command, VelocityCommand())

    def test_attachment_is_forwarded_with_exact_source_frame(self):
        runtime = NodeRuntime(NodeConfig.model_validate(CONFIG), clock=lambda: self.now)
        context = self.result().capture
        packet = sample(image_bytes(), "image/jpeg")
        packet.attachment = SimpleNamespace(to_bytes=lambda: context.model_dump_json().encode())
        runtime._on_camera(packet)
        self.assertEqual(runtime.state.latest_frame().capture, context)
        runtime.state.model_loaded("cpu")
        runtime.state.receive_command(ControlCommand(type="heartbeat"))
        request = runtime.state.begin_inference()
        runtime.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.assertEqual(request.frame.capture, context)
        packet.attachment = SimpleNamespace(to_bytes=lambda: b'{"kind":"static_image_demo","frame_id":"bad"}')
        runtime._on_camera(packet)
        self.assertEqual(runtime.state.snapshot().rejected_samples["camera"], 1)

    def test_unrelated_camera_attachment_preserves_image_only_inference(self):
        runtime = NodeRuntime(NodeConfig.model_validate(CONFIG), clock=lambda: self.now)
        packet = sample(image_bytes(), "image/jpeg")
        for attachment in (b"camera timestamp", b'{"timestamp":123}', b"\xff\xff"):
            packet.attachment = SimpleNamespace(to_bytes=lambda: attachment)
            runtime._on_camera(packet)
            self.assertIsNone(runtime.state.latest_frame().capture)
        self.assertEqual(runtime.state.snapshot().camera.received_frames, 3)
        self.assertEqual(runtime.state.snapshot().rejected_samples["camera"], 0)

    def test_view_displays_fresh_path_and_clears_expired_path(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.follower.receive_result(self.result())
        view = StreamView(self.follower)
        try:
            self.follower.tick()
            view.draw()
            view.figure.canvas.draw()
            self.assertEqual(len(view.reference.get_xdata()), 3)
            self.now += 4
            self.refresh_state()
            self.follower.tick()
            view.draw()
            self.assertEqual(len(view.reference.get_xdata()), 0)
            self.assertIn("stale_path", view.info.get_text())
        finally:
            plt.close(view.figure)


class StreamNetworkTests(unittest.TestCase):
    def test_annotated_image_to_inference_to_follower_over_real_tcp(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
        peer_config = zenoh.Config.from_json5(json.dumps({"mode": "peer", "listen": {"endpoints": [endpoint]},
                                                        "scouting": {"multicast": {"enabled": False}}}))
        client_json = json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]},
                                 "scouting": {"multicast": {"enabled": False}}})
        failures, commands = Queue(), Queue()
        config = NodeConfig.model_validate(CONFIG)

        class Engine:
            device = "cpu"

            def infer(self, image):
                return InferenceOutput([[(32, 44), (32, 36), (32, 32)]], ["test"], "test", "cpu")

        node = NodeRuntime(config, engine_loader=lambda _: Engine())
        stop_node, stop_follower = Event(), Event()
        follower = StreamFollower(CAMERA)

        def node_run():
            try:
                node.run(peer_config, stop_node)
            except Exception as error:
                failures.put(error)

        def follower_run():
            try:
                wait_for_listener(endpoint)
                run_session(follower, config.keys(), zenoh.Config.from_json5(client_json),
                            stop_follower, headless=True, duration=8)
            except Exception as error:
                failures.put(error)

        node_thread = Thread(target=node_run, daemon=True)
        follow_thread = Thread(target=follower_run, daemon=True)
        node_thread.start()
        follow_thread.start()
        try:
            wait_for_listener(endpoint)
            with zenoh.open(zenoh.Config.from_json5(client_json)) as session, ExitStack() as resources:
                resources.enter_context(session.declare_subscriber("unitree/go2/command", lambda s: commands.put(s.payload.to_bytes())))
                camera_pub = resources.enter_context(session.declare_publisher("camera/front", encoding="image/jpeg"))
                control_pub = resources.enter_context(session.declare_publisher("vamos/command", encoding="application/json"))
                deadline = time.monotonic() + 5
                index = 0
                while time.monotonic() < deadline:
                    index += 1
                    context = CaptureContext(source_id="tcp-test", frame_id=index,
                                             captured_at_unix_seconds=time.time())
                    camera_pub.put(image_bytes(), attachment=context.model_dump_json().encode())
                    control_pub.put('{"type":"heartbeat"}')
                    if follower.snapshot()["accepted"] >= 2 and follower.snapshot()["pose"]["x"] > 0:
                        break
                    time.sleep(0.05)
                self.assertGreaterEqual(follower.snapshot()["accepted"], 2, follower.snapshot())
                self.assertGreater(follower.snapshot()["pose"]["x"], 0)
                context = follower.snapshot()["capture"]
                self.assertEqual(context["source_id"], "tcp-test")
                control_pub.put('{"type":"stop"}')
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and follower.snapshot()["status"] not in ("remote_stop", "node_not_ready"):
                    time.sleep(0.02)
                self.assertEqual(follower.snapshot()["command"]["vx"], 0)
                self.assertTrue(commands.empty())
        finally:
            stop_follower.set()
            stop_node.set()
            follow_thread.join(3)
            node_thread.join(3)
        self.assertFalse(node_thread.is_alive())
        self.assertFalse(follow_thread.is_alive())
        self.assertTrue(failures.empty(), list(failures.queue))


if __name__ == "__main__":
    unittest.main()
