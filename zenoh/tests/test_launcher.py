from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
import zenoh

import cli
from cli import CliRuntime
from config import NodeConfig
from launcher import HERE, ManagedProcesses, check_listener_available, listener_addresses
from test_transport import BASE_CONFIG


class LauncherTests(unittest.TestCase):
    def test_cli_modes_launch_only_the_requested_services(self):
        for flags, expected in (([], ["node"]), (["--demo"], ["node", "source", "simulator"]), (["--demo", "--viewer", "--simulator"],
                ["node", "source", "viewer", "simulator"]), (["--attach"], [])):
            with self.subTest(flags=flags), patch.object(sys, "argv", ["cli.py", *flags]), \
                    patch.object(cli, "ManagedProcesses") as manager, \
                    patch.object(cli, "CliRuntime") as runtime, \
                    patch.object(cli, "check_listener_available"):
                owned = manager.return_value.__enter__.return_value
                self.assertEqual(cli.main(), 0)
                self.assertEqual([call.args[0] for call in owned.start.call_args_list], expected)
                for call in owned.start.call_args_list:
                    if call.args[0] == "node":
                        self.assertIn(HERE / "inference-node-config.example.json5", call.args)
                    if call.args[0] == "source":
                        self.assertIn("--fake-go2-state", call.args)
                        self.assertNotIn("--heartbeat", call.args)
                    if call.args[0] == "simulator":
                        self.assertEqual("--headless" in call.args, "--simulator" not in flags)
                runtime.return_value.run.assert_called_once()
                manager.return_value.__exit__.assert_called_once()

    def test_existing_listener_is_reported_without_starting_or_stopping_it(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            path = Path(directory) / "server.json"
            path.write_text(json.dumps({"listen": {"endpoints": [f"tcp/127.0.0.1:{port}"]}}))
            with patch.object(sys, "argv", ["cli.py", "--node-zenoh-config", str(path)]), \
                    patch.object(cli, "CliRuntime") as runtime, self.assertLogs(level="ERROR") as logs:
                self.assertEqual(cli.main(), 1)
            runtime.return_value.run.assert_not_called()
            self.assertIn("--attach", " ".join(logs.output))
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass  # The external listener is still alive.

    def test_local_startup_rejects_transport_only_configuration(self):
        with patch.object(sys, "argv", ["cli.py", "--node-config", str(HERE / "node-config.example.json5")]), \
                self.assertLogs(level="ERROR"):
            self.assertEqual(cli.main(), 1)

    def test_owned_node_and_source_stop_even_after_operator_error(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as available:
            work = Path(directory)
            available.bind(("127.0.0.1", 0))
            endpoint = f"tcp/127.0.0.1:{available.getsockname()[1]}"
            available.close()
            server = work / "server.json"
            client = work / "client.json"
            server.write_text(json.dumps({"mode": "peer", "listen": {"endpoints": [endpoint]},
                                         "scouting": {"multicast": {"enabled": False}}}))
            client.write_text(json.dumps({"mode": "client", "connect": {"endpoints": [endpoint]},
                                         "scouting": {"multicast": {"enabled": False}}}))
            manager = ManagedProcesses(work / "logs", shutdown_timeout=10, output=lambda _: None)
            descendants = []
            with self.assertRaisesRegex(RuntimeError, "operator error"):
                with manager:
                    addresses = listener_addresses(server)
                    check_listener_available(addresses)
                    manager.start("node", HERE / "node.py", "--zenoh-config", server)
                    manager.wait_for_listener(addresses)
                    manager.start("source", HERE / "diagnostic_peer.py", "--zenoh-config", client, "--fake-go2-state")
                    operator = CliRuntime(NodeConfig.model_validate(BASE_CONFIG), output=lambda _: None)
                    with operator.connect(zenoh.Config.from_file(str(client))):
                        state = operator.cache.wait(lambda s: s.camera.fresh and s.go2.fresh, 8)
                        self.assertIsNotNone(state)
                        self.assertTrue(state.cli_connected)
                        for child in manager.children.values():
                            descendants.extend(psutil.Process(child.process.pid).children(recursive=True))
                        raise RuntimeError("operator error")
            for child in manager.children.values():
                self.assertEqual(child.process.returncode, 0, child.log.read_text())
            _, alive = psutil.wait_procs(descendants, timeout=3)
            self.assertEqual(alive, [])

    def test_failed_node_is_detected_and_unresponsive_child_tree_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            bad = work / "bad.py"
            bad.write_text("raise SystemExit(7)\n")
            with ManagedProcesses(work, output=lambda _: None) as manager:
                manager.start("node", bad)
                with self.assertRaisesRegex(RuntimeError, "code 7"):
                    manager.wait_for_listener([("127.0.0.1", 1)], timeout=5)
            ready = work / "ready"
            stuck = work / "stuck.py"
            stuck.write_text("import sys, time\nfrom pathlib import Path\n"
                             "Path(sys.argv[1]).touch()\nwhile True: time.sleep(0.1)\n")
            output = []
            manager = ManagedProcesses(work, shutdown_timeout=0.05, output=output.append)
            descendants = []
            with manager:
                process = manager.start("stuck", stuck, ready)
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                descendants = psutil.Process(process.pid).children(recursive=True)
            self.assertIsNotNone(process.poll())
            _, alive = psutil.wait_procs(descendants, timeout=3)
            self.assertEqual(alive, [])
            self.assertTrue(any("terminating" in line for line in output))


if __name__ == "__main__":
    unittest.main()
