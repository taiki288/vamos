"""Own the local processes launched by the operator CLI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

import json5
import psutil


STOP_FILE_ENV = "VAMOS_MANAGED_STOP_FILE"
HERE = Path(__file__).resolve().parent


def listener_addresses(path: Path) -> list[tuple[str, int]]:
    settings = json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False)
    addresses = []
    for endpoint in settings.get("listen", {}).get("endpoints", []):
        if not endpoint.startswith("tcp/"):
            continue
        parsed = urlsplit("tcp://" + endpoint[4:])
        if not parsed.hostname or not parsed.port or parsed.query or parsed.fragment:
            raise ValueError(f"Unsupported local listener: {endpoint}")
        addresses.append((parsed.hostname, parsed.port))
    if not addresses:
        raise ValueError("Managed startup needs a TCP listen endpoint; use --attach for an existing node")
    return addresses


def check_listener_available(addresses):
    for host, port in addresses:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            if os.name == "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind((host, port))
            except OSError as error:
                raise RuntimeError(f"Cannot start local node on {host}:{port}. "
                                   "Stop the existing node or use --attach.") from error


@dataclass
class Child:
    process: subprocess.Popen
    log: Path
    stop_file: Path


class ManagedProcesses:
    def __init__(self, log_root: Path, *, shutdown_timeout=35.0, output=print):
        self.log_root = log_root.resolve()
        self.shutdown_timeout, self.output = shutdown_timeout, output
        self.directory = None
        self.children: dict[str, Child] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def start(self, name: str, script: Path, *arguments):
        if name in self.children:
            raise ValueError(f"Process already managed: {name}")
        if self.directory is None:
            self.log_root.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="session-", dir=self.log_root))
            self.output(f"Session logs: {self.directory}")
        log, stop_file = self.directory / f"{name}.log", self.directory / f"{name}.stop"
        env = {**os.environ, "PYTHONUTF8": "1", STOP_FILE_ENV: str(stop_file)}
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(script.resolve()), *map(str, arguments)],
                cwd=HERE.parent, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.children[name] = Child(process, log, stop_file)
        self.output(f"Started {name} (PID {process.pid}).")
        return process

    def wait_for_listener(self, addresses, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            child = self.children["node"]
            if child.process.poll() is not None:
                raise RuntimeError(f"Node exited with code {child.process.returncode}; see {child.log}")
            for host, port in addresses:
                host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
                try:
                    with socket.create_connection((host, port), timeout=0.1):
                        return
                except OSError:
                    pass
            time.sleep(0.05)
        raise TimeoutError(f"Node listener did not start; see {self.children['node'].log}")

    def status(self):
        if not self.children:
            self.output("Attached to an existing node; no local processes are managed.")
        for name, child in self.children.items():
            code = child.process.poll()
            self.output(f"{name}: {'running' if code is None else f'exited ({code})'} "
                        f"PID={child.process.pid} log={child.log}")

    def check_required(self):
        for name in ("node", "camera", "go2", "executor"):
            child = self.children.get(name)
            if child is not None and child.process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {child.process.returncode}; see {child.log}")

    def logs(self, name="node"):
        if name not in self.children:
            raise ValueError(f"No managed process {name!r}; use processes to list them")
        self.output("\n".join(self.children[name].log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]))

    def close(self):
        deadline = time.monotonic() + self.shutdown_timeout
        # The executor rests the (simulated) robot first, which needs Go2 state.
        executor = self.children.get("executor")
        if executor is not None and executor.process.poll() is None:
            executor.stop_file.touch()
            try:
                executor.process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass  # Terminated with the others below.
        # Request every other owned process to stop before waiting for the slow model.
        # Cooperative shutdown also reaches Windows venv redirector children.
        for child in self.children.values():
            if child.process.poll() is None:
                child.stop_file.touch()
        for name, child in reversed(list(self.children.items())):
            try:
                child.process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self.output(f"{name} did not stop in time; terminating its owned process tree.")
                try:
                    parent = psutil.Process(child.process.pid)
                    descendants = parent.children(recursive=True)
                    for process in reversed(descendants):
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    parent.kill()
                    psutil.wait_procs([parent, *descendants], timeout=5)
                except psutil.NoSuchProcess:
                    pass
                child.process.wait(5)
