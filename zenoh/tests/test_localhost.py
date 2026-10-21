"""Exercise real TCP Zenoh sessions in separate processes, without hardware."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from queue import Empty, Queue
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zenoh

from config import NodeState
from test_transport import BASE_CONFIG, GO2, image_bytes


class LocalhostTests(unittest.TestCase):
    def test_cross_process_round_trip_rejection_and_staleness(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        endpoint = f"tcp/127.0.0.1:{port}"
        root = Path(__file__).resolve().parents[2]
        process = None
        with tempfile.TemporaryDirectory(prefix="vamos-zenoh-") as directory:
            work = Path(directory)
            config = {**BASE_CONFIG, "node_state_publish_frequency_hz": 20.0,
                      "cli_heartbeat_timeout_seconds": 0.3,
                      "camera": {"key": "camera/front", "frame_freshness_seconds": 0.3},
                      "go2": {"zenoh_key_prefix": "unitree/go2", "node_state_timeout_seconds": 0.3}}
            (work / "node.json").write_text(json.dumps(config), encoding="utf-8")
            (work / "zenoh.json").write_text(json.dumps({
                "mode": "peer", "listen": {"endpoints": [endpoint]},
                "scouting": {"multicast": {"enabled": False}},
            }), encoding="utf-8")
            child = """
import sys
from pathlib import Path
from threading import Event, Thread
sys.path.insert(0, sys.argv[1])
import zenoh
from config import NodeConfig
from node import NodeRuntime
stop = Event()
def wait_for_parent():
    sys.stdin.readline()
    stop.set()
Thread(target=wait_for_parent, daemon=True).start()
NodeRuntime(NodeConfig.model_validate_json(Path(sys.argv[2]).read_text())).run(zenoh.Config.from_file(sys.argv[3]), stop)
"""
            try:
                process = subprocess.Popen(
                    [sys.executable, "-B", "-u", "-c", child, str(root / "zenoh"),
                     str(work / "node.json"), str(work / "zenoh.json")],
                    cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                # Wait for the listener rather than sleeping for a fixed startup delay.
                deadline = time.monotonic() + 8
                while True:
                    if process.poll() is not None:
                        self.fail(f"Node exited during startup: {process.stdout.read()}")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            self.fail("Node listener did not start")
                        time.sleep(0.05)

                states = Queue()
                robot_commands = Queue()
                network = zenoh.Config.from_json5(json.dumps({
                    "mode": "client", "connect": {"endpoints": [endpoint]},
                    "scouting": {"multicast": {"enabled": False}},
                }))
                with zenoh.open(network) as session, ExitStack() as resources:
                    resources.enter_context(session.declare_subscriber(
                        "vamos/state", lambda s: states.put(NodeState.model_validate_json(s.payload.to_bytes()))))
                    resources.enter_context(session.declare_subscriber(
                        "unitree/go2/command", lambda s: robot_commands.put(s.payload.to_bytes())))
                    camera = resources.enter_context(session.declare_publisher("camera/front", encoding="image/jpeg"))
                    go2 = resources.enter_context(session.declare_publisher("unitree/go2/state", encoding="application/json"))
                    control = resources.enter_context(session.declare_publisher("vamos/command", encoding="application/json"))

                    def await_state(predicate, send=None):
                        end = time.monotonic() + 5
                        while time.monotonic() < end:
                            if send:
                                send()
                            try:
                                state = states.get(timeout=0.05)
                            except Empty:
                                continue
                            if predicate(state):
                                return state
                        self.fail("Timed out waiting for matching VAMOS state")

                    jpeg = image_bytes()

                    def send_inputs():
                        camera.put(jpeg)
                        go2.put(json.dumps(GO2))
                        control.put('{"type":"heartbeat"}')

                    fresh = await_state(lambda s: s.camera.fresh and s.go2.fresh and s.cli_connected, send_inputs)
                    self.assertEqual((fresh.camera.width, fresh.camera.height), (64, 48))
                    self.assertGreater(fresh.camera.received_frames, 0)
                    self.assertFalse(fresh.start_ready)
                    control.put('{"type":"stop"}')
                    await_state(lambda s: s.last_command == "stop")
                    control.put('{"type":"start"}')
                    camera.put(b"not a JPEG")
                    await_state(lambda s: s.rejected_samples["command"] > 0 and s.rejected_samples["camera"] > 0)
                    stale = await_state(lambda s: not s.camera.fresh and not s.go2.fresh and not s.cli_connected)
                    self.assertIsNone(stale.go2.robot_state)
                    self.assertFalse(stale.go2.robot_connected)
                    self.assertTrue(robot_commands.empty(), "Transport stage must not send robot commands")
                    process.stdin.write("\n")
                    process.stdin.flush()
                    process.wait(timeout=5)
                    self.assertEqual(process.returncode, 0, process.stdout.read())
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    process.stdin.close()
                    process.stdout.close()


if __name__ == "__main__":
    unittest.main()
