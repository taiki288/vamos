"""Receive camera/Go2 observations and optionally run VAMOS image-path inference."""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Callable

import zenoh
from PIL import Image

from config import (
    CameraState, CaptureContext, ControlCommand, Go2NodeState, Go2State, JPEG_ENCODING,
    JSON_ENCODING, InferenceState, NodeConfig, NodeState, TrajectoryResult, load_node_config,
)

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReceivedFrame:
    sequence: int
    received_at: float
    image: Image.Image
    capture: CaptureContext | None = None


@dataclass(frozen=True)
class InferenceRequest:
    inference_id: int
    epoch: int
    frame: ReceivedFrame


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
        self._shutting_down = False
        self._inference_enabled = config.inference is not None
        self._model_loaded = False
        self._device: str | None = None
        self._active = False
        self._epoch = 0
        self._request_id = 0
        self._last_input: tuple[int, int] | None = None
        self._completed = 0
        self._discarded = 0
        self._last_inference_frame: int | None = None
        self._last_duration: float | None = None
        self._inference_error: str | None = None

    def receive_frame(self, image: Image.Image, received_at: float, capture: CaptureContext | None = None) -> None:
        with self._lock:
            sequence = 1 if self._frame is None else self._frame.sequence + 1
            self._frame = ReceivedFrame(sequence, received_at, image, capture)

    def latest_frame(self) -> ReceivedFrame | None:
        """Return an owned copy only while the latest image is still fresh."""
        with self._lock:
            frame = self._frame
            if frame is None or self.clock() - frame.received_at > self.config.camera.frame_freshness_seconds:
                return None
            return ReceivedFrame(frame.sequence, frame.received_at, frame.image.copy(), frame.capture)

    def receive_go2(self, state: Go2NodeState) -> None:
        with self._lock:
            self._go2 = state
            self._go2_at = self.clock()
            self._go2_count += 1

    def receive_command(self, command: ControlCommand) -> None:
        with self._lock:
            if command.type == "start":
                if self.config.inference is None or not self._model_loaded or self._shutting_down:
                    raise ValueError("inference is not available; navigation start is not implemented")
                if not self._inference_enabled:
                    self._epoch += 1
                self._inference_enabled = True
                self._inference_error = None
            elif command.type == "stop":
                self._inference_enabled = False
                self._epoch += 1
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
            self._inference_enabled = False
            self._epoch += 1

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            self._inference_enabled = False
            self._epoch += 1

    def model_loaded(self, device: str) -> None:
        with self._lock:
            self._model_loaded = True
            self._device = device

    def begin_inference(self) -> InferenceRequest | None:
        with self._lock:
            if (self.config.inference is None or not self._model_loaded or not self._inference_enabled
                    or self._active or self._shutting_down or self._stopped):
                return None
            status = self.snapshot()
            if not status.camera.fresh or not status.cli_connected:
                return None
            frame = self._frame
            identity = (self._epoch, frame.sequence)
            if identity == self._last_input:
                return None
            self._last_input = identity
            self._request_id += 1
            self._active = True
            return InferenceRequest(self._request_id, self._epoch,
                                    ReceivedFrame(frame.sequence, frame.received_at, frame.image.copy(), frame.capture))

    def complete_inference(self, request: InferenceRequest, duration: float, publish: Callable[[], None]) -> bool:
        # Serialize the final acceptance/publication with stop commands. A stopped
        # request can never publish later just because a new start was received.
        with self._lock:
            self._active = False
            self._last_duration = duration
            self._last_inference_frame = request.frame.sequence
            status = self.snapshot()
            age = self.clock() - request.frame.received_at
            if (self._shutting_down or self._stopped or not self._inference_enabled
                    or request.epoch != self._epoch or not status.camera.fresh or not status.cli_connected
                    or age > self.config.inference.max_frame_age_seconds):
                self._discarded += 1
                return False
            publish()
            self._completed += 1
            return True

    def fail_inference(self, error: Exception, request: InferenceRequest | None = None) -> None:
        with self._lock:
            self._active = False
            if request is not None and request.epoch != self._epoch:
                self._discarded += 1
                return
            self._inference_error = str(error)
            self._inference_enabled = False

    def snapshot(self) -> NodeState:
        with self._lock:
            now = self.clock()
            frame = self._frame
            camera_age = None if frame is None else max(0.0, now - frame.received_at)
            camera_fresh = camera_age is not None and camera_age <= self.config.camera.frame_freshness_seconds
            go2_age = None if self._go2_at is None else max(0.0, now - self._go2_at)
            go2_fresh = go2_age is not None and go2_age <= self.config.go2.node_state_timeout_seconds
            go2 = self._go2 if go2_fresh else None
            lifecycle = "idle"
            if self.config.inference is not None:
                lifecycle = ("error" if self._inference_error else "loading" if not self._model_loaded
                             else "paused" if not self._inference_enabled else "running" if self._active else "idle")
            if self._shutting_down:
                lifecycle = "stopping"
            if self._stopped:
                lifecycle = "stopped"
            extra = {} if self.config.inference is None else {
                "stage": "inference",
                "status_message": "Image-path inference only; no robot commands are published.",
            }
            return NodeState(
                **extra,
                lifecycle=lifecycle,
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
                inference=InferenceState(
                    model_loaded=self._model_loaded, enabled=self._inference_enabled,
                    active=self._active, device=self._device, completed=self._completed,
                    discarded=self._discarded, last_frame_sequence=self._last_inference_frame,
                    last_duration_seconds=self._last_duration, last_error=self._inference_error,
                ),
            )


def has_encoding(sample, expected: str) -> bool:
    return str(sample.encoding).split(";", 1)[0] == expected


class NodeRuntime:
    def __init__(self, config: NodeConfig, *, clock: Callable[[], float] = time.monotonic, engine_loader=None):
        self.config = config
        self.state = TransportState(config, clock)
        self._decode_lock = Lock()
        self._engine_loader = engine_loader

    def _run_inference(self, stop: Event, paths_publisher, preview_publisher) -> None:
        from inference import VAMOSEngine, render_preview

        config = self.config.inference
        try:
            engine = (self._engine_loader or VAMOSEngine.load)(config)
            self.state.model_loaded(engine.device)
            LOGGER.info("VAMOS ready on %s; waiting for fresh image and heartbeat", engine.device)
        except Exception as error:
            self.state.fail_inference(error)
            LOGGER.exception("VAMOS model loading failed")
            return
        period = 1.0 / config.frequency_hz
        deadline = time.monotonic()
        while not stop.is_set():
            request = self.state.begin_inference()
            if request is not None:
                started = time.perf_counter()
                try:
                    output = engine.infer(request.frame.image)
                    duration = time.perf_counter() - started
                    result = TrajectoryResult(
                        inference_id=request.inference_id, frame_sequence=request.frame.sequence,
                        image_width=request.frame.image.width, image_height=request.frame.image.height,
                        goal_pixels=config.goal.pixels(request.frame.image.width, request.frame.image.height),
                        prompt=output.prompt, trajectories=output.trajectories,
                        generated_texts=output.generated_texts, device=output.device,
                        duration_seconds=duration,
                        source_age_seconds=max(0.0, self.state.clock() - request.frame.received_at),
                        capture=request.frame.capture,
                    )
                    preview = render_preview(request.frame.image, result)

                    def publish():
                        preview_publisher.put(preview, attachment=str(result.inference_id).encode("ascii"))
                        paths_publisher.put(result.model_dump_json())

                    if self.state.complete_inference(request, duration, publish):
                        LOGGER.info("Published inference=%d frame=%d paths=%d %.3fs %s",
                                    result.inference_id, result.frame_sequence, len(result.trajectories), duration, output.device)
                    else:
                        LOGGER.info("Discarded inference=%d %.3fs source_age=%.3fs after stop, stale input, or shutdown",
                                    result.inference_id, duration, result.source_age_seconds)
                except Exception as error:
                    self.state.fail_inference(error, request)
                    LOGGER.exception("VAMOS inference failed; paused until start or restart")
            now = time.monotonic()
            deadline += (max(0, math.floor((now - deadline) / period)) + 1) * period
            stop.wait(max(0.0, deadline - time.monotonic()))

    def _on_camera(self, sample) -> None:
        if not has_encoding(sample, JPEG_ENCODING):
            self.state.reject("camera")
            return
        if not self._decode_lock.acquire(blocking=False):
            return  # Skip overlapping frames rather than building a backlog.
        received_at = self.state.clock()
        try:
            attachment = getattr(sample, "attachment", None)
            capture = None
            if attachment is not None:
                try:
                    metadata = json.loads(attachment.to_bytes())
                except (ValueError, UnicodeDecodeError):
                    metadata = None
                # Attachments from independently versioned camera nodes may use
                # other formats. Only this explicitly tagged demo schema is ours.
                if isinstance(metadata, dict) and metadata.get("kind") == "static_image_demo":
                    capture = CaptureContext.model_validate(metadata)
            with Image.open(BytesIO(sample.payload.to_bytes())) as source:
                if source.format != "JPEG":
                    raise ValueError("camera payload is not a JPEG")
                image = source.convert("RGB")
            self.state.receive_frame(image, received_at, capture)
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
            inference_stop = Event()
            worker = None
            if self.config.inference is not None:
                paths = resources.enter_context(session.declare_publisher(
                    keys["trajectories"], encoding=JSON_ENCODING,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                ))
                preview = resources.enter_context(session.declare_publisher(
                    keys["preview"], encoding=JPEG_ENCODING,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                ))
                worker = Thread(target=self._run_inference, args=(inference_stop, paths, preview),
                                name="vamos-inference", daemon=True)
                worker.start()
            period = 1.0 / self.config.node_state_publish_frequency_hz
            deadline = time.monotonic()
            try:
                while not stop.is_set():
                    publisher.put(self.state.snapshot().model_dump_json())
                    now = time.monotonic()
                    deadline += (max(0, math.floor((now - deadline) / period)) + 1) * period
                    stop.wait(max(0.0, deadline - time.monotonic()))
            finally:
                self.state.request_shutdown()
                inference_stop.set()
                if worker is not None:
                    worker.join(self.config.inference.shutdown_timeout_seconds)
                    if worker.is_alive():
                        raise RuntimeError("VAMOS inference did not finish before the shutdown timeout")
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
    except (OSError, ValueError, RuntimeError, zenoh.ZError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
