"""Receive camera/Go2 observations and run VAMOS inference in stop-go cycles."""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from contextlib import ExitStack, contextmanager
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Callable
from uuid import uuid4

import zenoh
from PIL import Image

from config import (
    CameraState, ControlAck, ControlCommand, CycleState, ExecutorState, GoalConfig, Go2NodeState, Go2State, JPEG_ENCODING,
    JSON_ENCODING, InferenceState, NodeConfig, NodeState, TrajectoryResult, load_node_config,
)

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReceivedFrame:
    sequence: int
    received_at: float
    image: Image.Image


@dataclass(frozen=True)
class InferenceRequest:
    inference_id: int
    epoch: int
    frame: ReceivedFrame
    goal: GoalConfig
    goal_revision: int
    cycle_id: int | None = None


class TransportState:
    """A bounded latest-observation store shared by Zenoh callbacks."""

    def __init__(self, config: NodeConfig, clock: Callable[[], float] = time.monotonic, wall_clock=time.time):
        self.config = config
        self.clock = clock
        self.wall_clock = wall_clock
        self._lock = RLock()
        self._frame: ReceivedFrame | None = None
        self._go2: Go2NodeState | None = None
        self._go2_at: float | None = None
        self._go2_count = 0
        self._heartbeat_at: float | None = None
        self._command_count = 0
        self._last_command: str | None = None
        self._rejected = {"camera": 0, "go2": 0, "command": 0}
        if self.stop_go:
            self._rejected["executor"] = 0
        self._stopped = False
        self._shutting_down = False
        self._inference_enabled = False
        self._session_id = str(uuid4())
        self._goal = config.inference.goal if config.inference else None
        self._goal_revision = 0
        self._last_ack = None
        self._acks = OrderedDict()
        self._inputs_seen = False
        self._pause_reason = None
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
        self._cycle_phase = "paused"
        self._cycle_id = self._cycle_completed = 0
        self._cycle_inference_id = None
        self._cycle_at = clock()
        self._stop_confirmed_at = None
        self._min_cycle_frame = 0
        self._executor = None
        self._executor_at = None
        self._observation_pose = None

    @property
    def stop_go(self):
        """Every inference configuration runs stop/observe/infer/move cycles."""
        return self.config.inference is not None

    def _set_cycle_phase(self, phase):
        self._cycle_phase, self._cycle_at = phase, self.clock()

    def _pause(self, reason):
        self._inference_enabled = False
        self._epoch += 1
        self._pause_reason = reason
        self._set_cycle_phase("paused")

    def _executor_fresh(self, now):
        state = self._executor
        return bool(self.stop_go and state is not None and self._executor_at is not None
                    and now - self._executor_at <= self.config.inference.stop_go.executor_timeout_seconds
                    and state.node_session_id == self._session_id and state.control_epoch == self._epoch
                    and state.cycle_id == self._cycle_id)

    def receive_executor(self, state: ExecutorState):
        with self._lock:
            if (not self.stop_go or state.node_session_id != self._session_id
                    or state.control_epoch != self._epoch or state.cycle_id != self._cycle_id):
                return
            if (self._inference_enabled and self._executor is not None
                    and self._executor.executor_session_id != state.executor_session_id):
                self._pause("Executor changed during a cycle; start required.")
                return
            self._executor, self._executor_at = state, self.clock()
            if not self._inference_enabled:
                return
            if state.motion == "error":
                self._pause(f"Executor stopped the cycle: {state.message}")
            elif self._cycle_phase in ("waiting_image", "inferencing") and (
                    state.motion != "stopped" or state.pose != self._observation_pose):
                self._pause("Executor moved during observation/inference; start required.")
            elif self._cycle_phase == "waiting_stop" and state.motion == "stopped":
                self._min_cycle_frame = 0 if self._frame is None else self._frame.sequence
                self._stop_confirmed_at = self.clock()
                self._observation_pose = state.pose
                self._set_cycle_phase("waiting_image")
            elif (self._cycle_phase == "moving" and state.motion == "stopped"
                    and state.inference_id == self._cycle_inference_id):
                self._cycle_completed += 1
                self._cycle_id += 1
                self._cycle_inference_id = None
                self._set_cycle_phase("waiting_stop")

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
            payload = command.model_dump_json()
            if command.request_id in self._acks:
                previous_payload, ack = self._acks[command.request_id]
                if previous_payload == payload:
                    self._last_ack = ack
                    return  # Reliable delivery/retry cannot execute the action twice.
                self._last_ack = ControlAck(request_id=command.request_id, type=command.type,
                                            accepted=False, message="request ID reused with different content")
                raise ValueError(self._last_ack.message)
            try:
                self._apply_command(command)
            except ValueError as error:
                self._record_ack(command, payload, False, str(error))
                raise
            self._record_ack(command, payload, True, "accepted")
            self._command_count += 1
            self._last_command = command.type

    def _record_ack(self, command, payload, accepted, message):
        if command.request_id is None:
            return
        ack = ControlAck(request_id=command.request_id, type=command.type, accepted=accepted, message=message)
        self._last_ack = ack
        self._acks[command.request_id] = (payload, ack)
        if len(self._acks) > 64:
            self._acks.popitem(last=False)

    def _apply_command(self, command):
        if command.type == "heartbeat":
            self._heartbeat_at = self.clock()
            return
        if command.type == "stop":
            self._pause("Stopped by operator; start required.")
            return  # Stop remains unconditional, including loading/disconnection.
        if command.node_session_id is not None and command.node_session_id != self._session_id:
            raise ValueError("node restarted; refresh state before sending commands")
        if self.config.inference is None or self._shutting_down or self._stopped:
            raise ValueError("inference is unavailable")
        if command.expected_goal_revision is not None and command.expected_goal_revision != self._goal_revision:
            raise ValueError("goal changed; refresh state before sending commands")
        if self._active:
            raise ValueError("waiting for the previous inference to finish")
        if command.type == "goal":
            if self._inference_enabled:
                raise ValueError("stop inference before changing the goal")
            self._goal = command.goal
            self._goal_revision += 1
            self._epoch += 1
            return
        blocker = self._start_blocker(self.snapshot())
        if blocker:
            raise ValueError(blocker)
        if not self._inference_enabled:
            self._epoch += 1
        self._inference_enabled = True
        self._inputs_seen = True
        self._inference_error = None
        self._pause_reason = None
        if self.stop_go:
            self._cycle_id += 1
            self._cycle_inference_id = None
            self._set_cycle_phase("waiting_stop")

    def _start_blocker(self, status: NodeState) -> str | None:
        """Return why start would be rejected for an otherwise valid request."""
        if not self._model_loaded:
            return ("model failed to load; check the node log and restart it" if self._inference_error
                    else "model is still loading")
        if not status.camera.fresh or not status.cli_connected:
            return "start requires a fresh image and heartbeat"
        if self.stop_go and (not status.cycle.executor_fresh or status.cycle.executor_motion != "stopped"):
            return "start requires a connected, stopped stop-go executor"
        return None

    def _expire_inputs(self, now):
        if self._inference_enabled and self._inputs_seen:
            camera_stale = self._frame is None or now - self._frame.received_at > self.config.camera.frame_freshness_seconds
            cli_stale = self._heartbeat_at is None or now - self._heartbeat_at > self.config.cli_heartbeat_timeout_seconds
            if camera_stale or cli_stale:
                self._pause("Image or heartbeat expired; start required after recovery.")
            elif self.stop_go:
                settings = self.config.inference.stop_go
                if not self._executor_fresh(now):
                    # Allow the new cycle/epoch to be echoed once after a transition.
                    if now - self._cycle_at > settings.executor_timeout_seconds:
                        self._pause("Executor feedback expired; start required after recovery.")
                timeout = {"waiting_stop": settings.stop_confirm_timeout_seconds,
                           "waiting_image": settings.image_wait_timeout_seconds,
                           "inferencing": settings.inference_timeout_seconds,
                           "moving": settings.movement_timeout_seconds}.get(self._cycle_phase)
                if timeout is not None and now - self._cycle_at > timeout:
                    self._pause(f"Cycle phase {self._cycle_phase} timed out after {timeout:g}s; start required.")

    def reject(self, stream: str) -> None:
        with self._lock:
            self._rejected[stream] += 1

    def mark_stopped(self) -> None:
        with self._lock:
            self._stopped = True
            self._pause("Node stopped.")

    def request_shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            self._pause("Node shutting down.")

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
            if not self._inference_enabled or not status.camera.fresh or not status.cli_connected:
                return None
            self._inputs_seen = True
            frame = self._frame
            if self._cycle_phase != "waiting_image" or not self._executor_fresh(self.clock()):
                return None
            # The frame must arrive after stopped feedback plus the observation
            # delay, so camera latency cannot pass off a pre-stop image.
            delay = self.config.inference.stop_go.observation_delay_seconds
            if frame.sequence <= self._min_cycle_frame or frame.received_at < self._stop_confirmed_at + delay:
                return None
            identity = (self._epoch, frame.sequence)
            if identity == self._last_input:
                return None
            self._last_input = identity
            self._request_id += 1
            self._active = True
            self._set_cycle_phase("inferencing")
            return InferenceRequest(self._request_id, self._epoch,
                                    ReceivedFrame(frame.sequence, frame.received_at, frame.image.copy()),
                                    self._goal, self._goal_revision, self._cycle_id)

    def complete_inference(self, request: InferenceRequest, duration: float, publish: Callable[[], None]) -> bool:
        # Serialize the final acceptance/publication with stop commands. A stopped
        # request can never publish later just because a new start was received.
        with self._lock:
            self._active = False
            self._last_duration = duration
            self._last_inference_frame = request.frame.sequence
            status = self.snapshot()
            # The executor holds still from observation to result, so the frame
            # stays valid for the whole stopped inference budget.
            age = self.clock() - request.frame.received_at
            if (self._shutting_down or self._stopped or not self._inference_enabled
                    or request.epoch != self._epoch or not status.camera.fresh or not status.cli_connected
                    or age > self.config.inference.stop_go.inference_timeout_seconds):
                self._discarded += 1
                if self._inference_enabled:
                    self._pause("Inference result expired; start required.")
                return False
            publish()
            self._completed += 1
            self._cycle_inference_id = request.inference_id
            self._set_cycle_phase("moving")
            return True

    def fail_inference(self, error: Exception, request: InferenceRequest | None = None) -> None:
        with self._lock:
            self._active = False
            if request is not None and request.epoch != self._epoch:
                self._discarded += 1
                return
            self._inference_error = str(error)
            self._pause("Inference failed; start required.")

    def snapshot(self) -> NodeState:
        with self._lock:
            now = self.clock()
            self._expire_inputs(now)
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
            extra = {} if self.config.inference is None else {"stage": "inference"}
            state = NodeState(
                **extra,
                cycle=CycleState(phase=self._cycle_phase,
                    cycle_id=self._cycle_id, completed=self._cycle_completed, inference_id=self._cycle_inference_id,
                    executor_fresh=self._executor_fresh(now),
                    executor_motion=self._executor.motion if self._executor_fresh(now) else None),
                node_session_id=self._session_id, control_epoch=self._epoch,
                goal=self._goal, goal_revision=self._goal_revision, last_control_ack=self._last_ack,
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
            if self.config.inference is None:
                return state
            blocker = self._start_blocker(state)
            if self._shutting_down or self._stopped or self._active:
                blocker = blocker or "inference is busy or shutting down"
            ready = not self._inference_enabled and blocker is None
            if self._inference_enabled:
                message = "Running stop/observe/infer/move cycles."
            else:
                message = "Ready; send start." if ready else f"Not ready: {blocker}."
                if self._pause_reason:
                    message = f"{self._pause_reason} {message}"
            return state.model_copy(update={"start_ready": ready, "status_message": message})


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
                    output = engine.infer(request.frame.image, goal=request.goal)
                    duration = time.perf_counter() - started
                    result = TrajectoryResult(
                        generated_at_unix_seconds=self.state.wall_clock(),
                        cycle_id=request.cycle_id,
                        inference_id=request.inference_id, frame_sequence=request.frame.sequence,
                        image_width=request.frame.image.width, image_height=request.frame.image.height,
                        goal_pixels=request.goal.pixels(request.frame.image.width, request.frame.image.height),
                        node_session_id=self.state.snapshot().node_session_id,
                        control_epoch=request.epoch, goal_revision=request.goal_revision,
                        prompt=output.prompt, trajectories=output.trajectories,
                        generated_texts=output.generated_texts, device=output.device,
                        duration_seconds=duration,
                        source_age_seconds=max(0.0, self.state.clock() - request.frame.received_at),
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

    def _on_executor(self, sample):
        try:
            if not has_encoding(sample, JSON_ENCODING):
                raise ValueError("executor state must use application/json")
            self.state.receive_executor(ExecutorState.model_validate_json(sample.payload.to_bytes()))
        except ValueError as error:
            self.state.reject("executor")
            LOGGER.warning("Rejected executor feedback: %s", error)

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
            if self.state.stop_go:
                resources.enter_context(session.declare_subscriber(keys["executor_state"], self._on_executor))
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
    watcher_done = Event()
    watcher = None
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: requested.set())
        managed_stop_file = os.environ.get("VAMOS_MANAGED_STOP_FILE")
        if managed_stop_file:
            def watch_stop_file():
                while not watcher_done.is_set():
                    if Path(managed_stop_file).exists():
                        requested.set()
                        return
                    watcher_done.wait(0.1)

            watcher = Thread(target=watch_stop_file, name="vamos-managed-stop", daemon=True)
            watcher.start()
        yield requested
    finally:
        watcher_done.set()
        if watcher is not None:
            watcher.join(1)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def tolerate_unencodable_output():
    """Keep redirected Windows (e.g. cp932) logs from failing on emoji/Unicode.

    vamos_demo prints emoji while loading the model; with stdout redirected to
    a file, a non-UTF-8 console code page otherwise aborts model loading.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def main() -> int:
    tolerate_unencodable_output()
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
