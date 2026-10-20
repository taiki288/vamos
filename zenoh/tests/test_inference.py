from __future__ import annotations

from contextlib import ExitStack
from io import BytesIO
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

from config import ControlCommand, InferenceConfig, NodeConfig, NodeState, TrajectoryResult
from inference import InferenceOutput, VAMOSEngine, build_prompt, render_preview
from node import NodeRuntime, TransportState
from test_transport import BASE_CONFIG, image_bytes, wait_for_listener


INFERENCE = {"goal": {"x": 0.434, "y": 0.514}, "device": "cpu", "frequency_hz": 20.0, "auto_start": True}


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.config = NodeConfig.model_validate({**BASE_CONFIG, "inference": INFERENCE})
        self.state = TransportState(self.config, clock=lambda: self.now)
        self.state.model_loaded("cpu")
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.state.receive_command(ControlCommand(type="heartbeat"))

    def test_prompt_and_goal_match_demo_tokens(self):
        prompt = build_prompt(self.config.inference)
        self.assertEqual(prompt, "Navigate to x=<loc0444>, y=<loc0526>.")
        edge = InferenceConfig.model_validate({"goal": {"x": 1.0, "y": 0.0}})
        self.assertIn("<loc1023>", build_prompt(edge))
        self.assertLess(edge.goal.pixels(64, 48)[0], 64)

    def test_invalid_generation_configs_fail_before_loading_model(self):
        for changes in ({"temperature": 0.0, "num_samples": 10}, {"max_new_tokens": 9},
                        {"goal": {"x": 1.1, "y": 0.5}}, {"frequency_hz": 1e-320},
                        {"disable_compile": "yes"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                InferenceConfig.model_validate({**INFERENCE, **changes})

    def test_no_overlap_no_reprocessing_and_newest_frame_wins(self):
        request = self.state.begin_inference()
        self.assertIsNotNone(request)
        self.assertIsNone(self.state.begin_inference())
        self.state.complete_inference(request, 0.1, lambda: None)
        self.assertIsNone(self.state.begin_inference())
        self.state.receive_frame(Image.new("RGB", (80, 60)), self.now)
        self.state.receive_frame(Image.new("RGB", (100, 75)), self.now)
        latest = self.state.begin_inference()
        self.assertEqual(latest.frame.sequence, 3)
        self.assertEqual(latest.frame.image.size, (100, 75))

    def test_stop_start_cannot_publish_old_request(self):
        request = self.state.begin_inference()
        self.state.receive_command(ControlCommand(type="stop"))
        self.state.receive_command(ControlCommand(type="heartbeat"))
        self.assertFalse(self.state.snapshot().inference.enabled)
        publish = Mock()
        self.assertFalse(self.state.complete_inference(request, 0.1, publish))
        self.state.receive_command(ControlCommand(type="start"))
        publish.assert_not_called()
        self.assertEqual(self.state.snapshot().inference.discarded, 1)
        self.assertIsNotNone(self.state.begin_inference())

    def test_stale_camera_or_heartbeat_discards_result(self):
        request = self.state.begin_inference()
        self.now += 3.0
        publish = Mock()
        self.assertFalse(self.state.complete_inference(request, 3.0, publish))
        publish.assert_not_called()
        self.assertIsNone(self.state.begin_inference())

    def test_fresh_inputs_do_not_make_an_old_source_frame_current(self):
        request = self.state.begin_inference()
        self.now += 6.0
        self.state.receive_frame(Image.new("RGB", (64, 48)), self.now)
        self.state.receive_command(ControlCommand(type="heartbeat"))
        publish = Mock()
        self.assertFalse(self.state.complete_inference(request, 6.0, publish))
        publish.assert_not_called()

    def test_failure_pauses_and_shutdown_invalidates_result(self):
        request = self.state.begin_inference()
        self.state.fail_inference(RuntimeError("test failure"), request)
        self.assertEqual(self.state.snapshot().inference.last_error, "test failure")
        self.assertIsNone(self.state.begin_inference())
        self.state.receive_command(ControlCommand(type="start"))
        request = self.state.begin_inference()
        self.state.request_shutdown()
        self.assertFalse(self.state.complete_inference(request, 0.1, Mock()))

    def test_model_load_failure_is_reported_instead_of_loading(self):
        state = TransportState(self.config, clock=lambda: self.now)
        state.fail_inference(RuntimeError("load failed"))
        status = state.snapshot()
        self.assertFalse(status.start_ready)
        self.assertIn("model failed to load", status.status_message)

    def test_invalid_paths_are_filtered_by_adapter(self):
        generate = Mock(return_value=([[[1, 2], [20, 30]], [[float("nan"), 0]], [[64, 2]]],
                                      ["valid", "nan", "outside"], None))
        engine = VAMOSEngine(None, None, "cpu", generate, self.config.inference)
        output = engine.infer(Image.new("RGB", (64, 48)))
        self.assertEqual(output.trajectories, [[[1, 2], [20, 30]]])
        self.assertEqual(output.generated_texts, ["valid"])
        self.assertIs(generate.call_args.kwargs["disable_compile"], True)

    def test_explicit_compilation_preference_reaches_the_generator(self):
        generate = Mock(return_value=([[[1, 2], [20, 30]]], ["valid"], None))
        config = self.config.inference.model_copy(update={"disable_compile": False})
        engine = VAMOSEngine(None, None, "cpu", generate, config)
        engine.infer(Image.new("RGB", (64, 48)))
        self.assertIs(generate.call_args.kwargs["disable_compile"], False)

    def test_preview_uses_source_dimensions_and_result_json_round_trips(self):
        result = TrajectoryResult(
            inference_id=1, frame_sequence=2, image_width=64, image_height=48,
            goal_pixels=(30, 30), prompt="test", trajectories=[[(5, 40), (30, 30)]],
            generated_texts=["test"], duration_seconds=0.1, source_age_seconds=0.2, device="cpu")
        with Image.open(BytesIO(render_preview(Image.new("RGB", (64, 48)), result))) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (64, 48))
        self.assertEqual(TrajectoryResult.model_validate_json(result.model_dump_json()), result)


class InferenceNetworkTests(unittest.TestCase):
    def test_state_is_live_during_model_loading_and_inference_and_results_are_published(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
        config = NodeConfig.model_validate({**BASE_CONFIG, "inference": INFERENCE})
        load_release, infer_release, infer_started = Event(), Event(), Event()
        failures, states, results, previews, commands = Queue(), Queue(), Queue(), Queue(), Queue()

        class FakeEngine:
            device = "cpu"

            def infer(self, image, goal=None):
                infer_started.set()
                if not infer_release.wait(5):
                    raise RuntimeError("Test inference was not released")
                return InferenceOutput([[(5, 40), (30, 30)]], ["test"], "test", "cpu")

        def load(_config):
            if not load_release.wait(5):
                raise RuntimeError("Test model load was not released")
            return FakeEngine()

        node = NodeRuntime(config, engine_loader=load)
        stop = Event()

        def run():
            try:
                node.run(zenoh.Config.from_json5(json.dumps({
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
                resources.enter_context(session.declare_subscriber("vamos/state", lambda s: states.put(NodeState.model_validate_json(s.payload.to_bytes()))))
                resources.enter_context(session.declare_subscriber("vamos/trajectories", lambda s: results.put(TrajectoryResult.model_validate_json(s.payload.to_bytes()))))
                resources.enter_context(session.declare_subscriber("vamos/preview", lambda s: previews.put((s.payload.to_bytes(), s.attachment.to_bytes()))))
                resources.enter_context(session.declare_subscriber("unitree/go2/command", lambda s: commands.put(s.payload.to_bytes())))
                camera = resources.enter_context(session.declare_publisher("camera/front", encoding="image/jpeg"))
                control = resources.enter_context(session.declare_publisher("vamos/command", encoding="application/json"))

                def await_state(predicate):
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        camera.put(image_bytes())
                        control.put('{"type":"heartbeat"}')
                        try:
                            state = states.get(timeout=0.05)
                        except Empty:
                            continue
                        if predicate(state):
                            return state
                    self.fail("State publication stalled")

                loading = await_state(lambda s: s.lifecycle == "loading" and s.camera.fresh)
                self.assertFalse(loading.inference.model_loaded)
                load_release.set()
                active = await_state(lambda s: s.inference.active)
                self.assertTrue(active.inference.model_loaded)
                self.assertTrue(infer_started.is_set())
                # Multiple periodic states must arrive while the fake model is blocked.
                await_state(lambda s: s.inference.active and s.camera.received_frames > active.camera.received_frames)
                infer_release.set()
                complete = await_state(lambda s: s.inference.completed > 0)
                result = results.get(timeout=3)
                jpeg, attachment = previews.get(timeout=3)
                self.assertEqual(int(attachment), result.inference_id)
                self.assertEqual(result.image_width, 64)
                self.assertLess(result.frame_sequence, complete.camera.received_frames)
                with Image.open(BytesIO(jpeg)) as image:
                    self.assertEqual(image.size, (64, 48))
                self.assertTrue(commands.empty())
        finally:
            load_release.set()
            infer_release.set()
            stop.set()
            worker.join(timeout=6)
        self.assertFalse(worker.is_alive())
        self.assertTrue(failures.empty(), list(failures.queue))


if __name__ == "__main__":
    unittest.main()
