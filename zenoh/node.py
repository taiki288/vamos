"""Receive camera/Go2 observations and publish VAMOS transport state."""

from __future__ import annotations

import argparse
import logging
import math
import signal
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, RLock
from typing import Callable

import zenoh
from PIL import Image

from config import (
    CameraState, ControlCommand, Go2NodeState, Go2State, JPEG_ENCODING,
    JSON_ENCODING, NodeConfig, NodeState, load_node_config,
)

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReceivedFrame:
    sequence: int
    received_at: float
    image: Image.Image


class TransportState:
    """A bounded latest-observation store shared by Zenoh callbacks."""

    def __init__(self, config: NodeConfig, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.clock = clock
        self._lock = RLock()
        self._frame: ReceivedFrame | None = None
        self._go2: Go2NodeState | None = None
        self._go2_at: float | None = None
        self._go2_count = 0
        self._heartbeat_at: float | None = None
        self._command_count = 0
        self._last_command: str | None = None
        self._rejected = {"camera": 0, "go2": 0, "command": 0}
        self._stopped = False

    def receive_frame(self, image: Image.Image, received_at: float) -> None:
        with self._lock:
            sequence = 1 if self._frame is None else self._frame.sequence + 1
            self._frame = ReceivedFrame(sequence, received_at, image)

    def latest_frame(self) -> ReceivedFrame | None:
        """Return an owned copy only while the latest image is still fresh."""
        with self._lock:
            frame = self._frame
            if frame is None or self.clock() - frame.received_at > self.config.camera.frame_freshness_seconds:
                return None
            return ReceivedFrame(frame.sequence, frame.received_at, frame.image.copy())

    def receive_go2(self, state: Go2NodeState) -> None:
        with self._lock:
            self._go2 = state
            self._go2_at = self.clock()
            self._go2_count += 1

    def receive_command(self, command: ControlCommand) -> None:
        with self._lock:
            self._command_count += 1
            self._last_command = command.type
            if command.type == "heartbeat":
                self._heartbeat_at = self.clock()

    def reject(self, stream: str) -> None:
        with self._lock:
            self._rejected[stream] += 1

    def mark_stopped(self) -> None:
        with self._lock:
            self._stopped = True

    def snapshot(self) -> NodeState:
        with self._lock:
            now = self.clock()
            frame = self._frame
            camera_age = None if frame is None else max(0.0, now - frame.received_at)
            camera_fresh = camera_age is not None and camera_age <= self.config.camera.frame_freshness_seconds
            go2_age = None if self._go2_at is None else max(0.0, now - self._go2_at)
            go2_fresh = go2_age is not None and go2_age <= self.config.go2.node_state_timeout_seconds
            go2 = self._go2 if go2_fresh else None
            return NodeState(
                lifecycle="stopped" if self._stopped else "idle",
                cli_connected=(self._heartbeat_at is not None and
                               now - self._heartbeat_at <= self.config.cli_heartbeat_timeout_seconds),
                received_commands=self._command_count,
                last_command=self._last_command,
                rejected_samples=dict(self._rejected),
                camera=CameraState(
                    received_frames=0 if frame is None else frame.sequence,
                    fresh=camera_fresh, age_seconds=camera_age,
                    width=None if frame is None else frame.image.width,
                    height=None if frame is None else frame.image.height,
                ),
                go2=Go2State(
                    received_states=self._go2_count, fresh=go2_fresh, age_seconds=go2_age,
                    robot_connected=go2.robot_connected if go2 else False,
                    accepting_commands=go2.accepting_commands if go2 else False,
                    robot_state=go2.robot_state if go2 else None,
                ),
            )


def has_encoding(sample, expected: str) -> bool:
    return str(sample.encoding).split(";", 1)[0] == expected


class NodeRuntime:
    def __init__(self, config: NodeConfig, *, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.state = TransportState(config, clock)
        self._decode_lock = Lock()

    def _on_camera(self, sample) -> None:
        if not has_encoding(sample, JPEG_ENCODING):
            self.state.reject("camera")
            return
        if not self._decode_lock.acquire(blocking=False):
            return  # Skip overlapping frames rather than building a backlog.
        received_at = self.state.clock()
        try:
            with Image.open(BytesIO(sample.payload.to_bytes())) as source:
                if source.format != "JPEG":
                    raise ValueError("camera payload is not a JPEG")
                image = source.convert("RGB")
            self.state.receive_frame(image, received_at)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            self.state.reject("camera")
            LOGGER.warning("Rejected camera image: %s", error)
        finally:
            self._decode_lock.release()

    def _on_go2_state(self, sample) -> None:
        try:
            if not has_encoding(sample, JSON_ENCODING):
                raise ValueError("Go2 state must use application/json")
            self.state.receive_go2(Go2NodeState.model_validate_json(sample.payload.to_bytes()))
        except ValueError as error:
            self.state.reject("go2")
            LOGGER.warning("Rejected Go2 state: %s", error)

    def _on_control(self, sample) -> None:
        try:
            if not has_encoding(sample, JSON_ENCODING):
                raise ValueError("control command must use application/json")
            self.state.receive_command(ControlCommand.model_validate_json(sample.payload.to_bytes()))
        except ValueError as error:
            self.state.reject("command")
            LOGGER.warning("Rejected control command: %s", error)

    def run(self, zenoh_config: zenoh.Config, stop: Event) -> None:
        keys = self.config.keys()
        with zenoh.open(zenoh_config) as session, ExitStack() as resources:
            publisher = resources.enter_context(session.declare_publisher(
                keys["state"], encoding=JSON_ENCODING,
                congestion_control=zenoh.CongestionControl.DROP,
                reliability=zenoh.Reliability.BEST_EFFORT,
            ))
            resources.enter_context(session.declare_subscriber(keys["camera"], self._on_camera))
            resources.enter_context(session.declare_subscriber(keys["go2_state"], self._on_go2_state))
            resources.enter_context(session.declare_subscriber(keys["command"], self._on_control))
            LOGGER.info("Transport ready: camera=%s Go2=%s state=%s", keys["camera"], keys["go2_state"], keys["state"])
            period = 1.0 / self.config.node_state_publish_frequency_hz
            deadline = time.monotonic()
            try:
                while not stop.is_set():
                    publisher.put(self.state.snapshot().model_dump_json())
                    now = time.monotonic()
                    deadline += (max(0, math.floor((now - deadline) / period)) + 1) * period
                    stop.wait(max(0.0, deadline - time.monotonic()))
            finally:
                self.state.mark_stopped()
                publisher.put(self.state.snapshot().model_dump_json())


@contextmanager
def stop_on_signals():
    requested = Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: requested.set())
        yield requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, default=HERE / "node-config.example.json5")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "zenoh-config.example.json5")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_node_config(args.node_config)
        with stop_on_signals() as stop:
            NodeRuntime(config).run(zenoh.Config.from_file(str(args.zenoh_config)), stop)
    except (OSError, ValueError, zenoh.ZError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
