from __future__ import annotations

import copy
from io import BytesIO
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pydantic import ValidationError

from config import DEMO_CONFIG_PATH, EXAMPLE_CONFIG_PATH, ControlCommand, Go2NodeState, NodeConfig, load_config
from node import NodeRuntime


BASE_CONFIG = {
    "zenoh_key_prefix": "vamos",
    "node_state_publish_frequency_hz": 10.0,
    "cli_heartbeat_timeout_seconds": 1.0,
    "camera": {"key": "camera/front", "frame_freshness_seconds": 2.5},
    "go2": {"zenoh_key_prefix": "unitree/go2", "node_state_timeout_seconds": 0.5},
}
GO2 = {
    "robot_connected": True,
    "robot_state": {"state": "down", "motion": "quiescent"},
    "accepting_commands": True,
}


def image_bytes(fmt="JPEG"):
    output = BytesIO()
    Image.new("RGB", (64, 48), "green").save(output, format=fmt)
    return output.getvalue()


def sample(payload, encoding):
    return SimpleNamespace(encoding=encoding, payload=SimpleNamespace(to_bytes=lambda: payload))


def wait_for_listener(endpoint: str, timeout: float = 5.0):
    """Wait for an actual TCP listener, not a scheduler-dependent fixed sleep."""
    host, port = endpoint.removeprefix("tcp/").rsplit(":", 1)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=0.1):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Listener did not become ready: {endpoint}")
            time.sleep(0.01)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.config = NodeConfig.model_validate(BASE_CONFIG)
        self.runtime = NodeRuntime(self.config, clock=lambda: self.now)

    def test_settings_files_load_and_keys_match_existing_nodes(self):
        for path in (EXAMPLE_CONFIG_PATH, DEMO_CONFIG_PATH):
            with self.subTest(path=path.name):
                settings = load_config(path)
                keys = settings.node_config().keys()
                self.assertEqual(keys["camera"], "camera/front")
                self.assertEqual(keys["go2_state"], "unitree/go2/state")
                self.assertEqual(keys["state"], "vamos/state")
                self.assertTrue(Path(settings.executor.camera_calibration).is_absolute())
                settings.zenoh.session("node"), settings.zenoh.session("client")
        self.assertFalse(load_config(EXAMPLE_CONFIG_PATH).executor.allow_assumed_calibration)
        self.assertTrue(load_config(DEMO_CONFIG_PATH).executor.allow_assumed_calibration)

    def test_settings_file_is_validated_as_a_whole(self):
        import json
        import tempfile

        base = {"zenoh": {"node": {"mode": "peer"}, "client": {"mode": "client"}},
                "node": {"inference": {"goal": {"x": 0.5, "y": 0.5}}}, "executor": {"camera_calibration": "c.json"}}
        for name, change in (("unknown section", {"extra": {}}),
                             ("bad zenoh", {"zenoh": {"node": {"mode": "nonsense"}, "client": {"mode": "client"}}}),
                             ("heartbeat", {"cli": {"heartbeat_interval_seconds": 2.0}}),
                             ("key collision", {"keys": {"camera": "vamos/state"}})):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "vamos.json5"
                path.write_text(json.dumps({**base, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_goal_is_either_an_image_point_or_a_metric_offset(self):
        from config import GoalConfig

        self.assertTrue(GoalConfig(x=0.5, y=0.6).is_image)
        self.assertEqual(GoalConfig(forward_m=2, left_m=-1).describe(), "2 m forward, -1 m left")
        for data in ({}, {"x": 0.5}, {"x": 0.5, "y": 0.5, "forward_m": 1, "left_m": 0}, {"forward_m": 1}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                GoalConfig.model_validate(data)
        with self.assertRaises(ValueError):
            GoalConfig(forward_m=1, left_m=0).tokens()

    def test_invalid_config_cannot_create_wildcard_or_colliding_channels(self):
        for change in ("wildcard", "collision", "nan", "infinite_period", "unknown"):
            with self.subTest(change=change):
                data = copy.deepcopy(BASE_CONFIG)
                if change == "wildcard":
                    data["camera"]["key"] = "camera/*"
                elif change == "collision":
                    data["camera"]["key"] = "vamos/state"
                elif change == "nan":
                    data["node_state_publish_frequency_hz"] = float("nan")
                elif change == "infinite_period":
                    data["node_state_publish_frequency_hz"] = 1e-320
                else:
                    data["typo"] = True
                with self.assertRaises(ValidationError):
                    NodeConfig.model_validate(data)

    def test_latest_frame_replaces_previous_and_returns_owned_copy(self):
        self.runtime._on_camera(sample(image_bytes(), "image/jpeg"))
        frame = self.runtime.state.latest_frame()
        self.assertEqual(frame.image.size, (64, 48))
        frame.image.putpixel((0, 0), (255, 0, 0))
        self.assertNotEqual(self.runtime.state.latest_frame().image.getpixel((0, 0)), (255, 0, 0))
        self.runtime._on_camera(sample(image_bytes(), "image/jpeg;test"))
        self.assertEqual(self.runtime.state.latest_frame().sequence, 2)

    def test_invalid_images_do_not_replace_or_refresh_valid_image(self):
        self.runtime._on_camera(sample(image_bytes(), "image/jpeg"))
        self.now += 3.0
        for payload, encoding in ((b"broken", "image/jpeg"), (image_bytes("PNG"), "image/jpeg"),
                                  (image_bytes(), "application/octet-stream")):
            self.runtime._on_camera(sample(payload, encoding))
        state = self.runtime.state.snapshot()
        self.assertEqual(state.camera.received_frames, 1)
        self.assertFalse(state.camera.fresh)
        self.assertEqual(state.rejected_samples["camera"], 3)
        self.assertIsNone(self.runtime.state.latest_frame())

    def test_stale_go2_state_is_not_presented_as_current(self):
        state = Go2NodeState.model_validate({**GO2, "new_external_field": 42})
        self.runtime.state.receive_go2(state)
        self.assertTrue(self.runtime.state.snapshot().go2.robot_connected)
        self.now += 0.6
        stale = self.runtime.state.snapshot().go2
        self.assertFalse(stale.fresh)
        self.assertFalse(stale.robot_connected)
        self.assertFalse(stale.accepting_commands)
        self.assertIsNone(stale.robot_state)

    def test_heartbeat_expires_and_stop_is_accepted_without_heartbeat(self):
        self.runtime.state.receive_command(ControlCommand(type="stop"))
        self.assertEqual(self.runtime.state.snapshot().last_command, "stop")
        self.runtime.state.receive_command(ControlCommand(type="heartbeat"))
        self.assertTrue(self.runtime.state.snapshot().cli_connected)
        self.now += 1.1
        self.assertFalse(self.runtime.state.snapshot().cli_connected)

    def test_unsupported_commands_and_old_go2_schema_are_rejected(self):
        self.runtime._on_control(sample(b'{"type":"start"}', "application/json"))
        self.runtime._on_control(sample(b'{"type":"heartbeat","extra":1}', "application/json"))
        self.runtime._on_go2_state(sample(b'{"connected":true,"robot":{},"accepting_commands":true}', "application/json"))
        self.runtime._on_go2_state(sample(b'{"robot_connected":"true","robot_state":{"state":"down","motion":"quiescent"},"accepting_commands":true}', "application/json"))
        state = self.runtime.state.snapshot()
        self.assertEqual(state.rejected_samples, {"camera": 0, "go2": 2, "command": 2})
        self.assertFalse(state.start_ready)
        self.assertEqual(state.go2.received_states, 0)


if __name__ == "__main__":
    unittest.main()
