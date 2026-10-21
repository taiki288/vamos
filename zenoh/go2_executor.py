"""Execute VAMOS stop-go steps on a Go2 through unitree-go2-zenoh-node.

This replaces the 2D simulator in the stop/observe/infer/move protocol. It
subscribes to the VAMOS node state, results and operator commands and to the
Go2 node state, and it is the only VAMOS process that publishes
`{go2 prefix}/command`:

* posture: `stand` once when a started cycle finds the robot resting, `down`
  once on every stop path (operator stop, node pause or loss, heartbeat loss,
  errors, shutdown), following the NaVILA node's safety contract
* velocity: repeated at the control rate while the robot stands for a running
  cycle (zero except during one bounded step); one zero when a stop begins

"Stopped" feedback requires Go2 to report a standing, quiescent state for the
settling interval while zero velocity is commanded.

The goal is anchored once per goal revision as a point on the ground in the
odometry frame. Before each observation the executor projects it into the
camera image and reports that point with "stopped"; a goal out of view first
triggers a turn in place, and a goal within `goal.arrival_tolerance_m` (or in
the camera's near blind zone) is reported as reached, which ends the run.

Pose comes from Go2 (`pose_source: "go2"`, the proposed `robot_state.position`
and `yaw` fields) or from dead reckoning of the executor's own commands
(`"dead_reckoning"`), which drifts over long runs.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from threading import Event, RLock
import time
from typing import Annotated, Callable, Literal
from uuid import uuid4

import json5
from pydantic import Field, StrictBool, StrictStr, model_validator
import zenoh

from config import (
    ControlCommand, ExecutorState, Go2NodeState, JSON_ENCODING, NodeConfig, NodeState,
    PlanarPose, PositiveSeconds, Settings, StopGoConfig, TrajectoryResult, load_node_config,
)
from controller import Pose2D, PurePursuitConfig, PurePursuitController, VelocityCommand
from projection import GroundCamera, GroundProjector, path_to_world
from selection import select_candidate

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
GO2_RESTING_STATES = frozenset(("damping", "down"))
GO2_STANDING_STATES = frozenset(("ready_stand", "locomotion"))
Phase = Literal["idle", "standing", "holding", "turning", "moving", "settling", "stopping", "error"]
ACTIVE_PHASES = frozenset(("standing", "holding", "turning", "moving", "settling", "error"))
MIN_TURN_RATE_RPS = 0.15
MAX_TURNS_WITHOUT_VIEW = 2


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class Go2Timing(Settings):
    state_timeout_seconds: PositiveSeconds = 0.5
    stand_timeout_seconds: PositiveSeconds = 10.0
    down_timeout_seconds: PositiveSeconds = 10.0


class ExecutorMotion(Settings):
    # Deliberately low caps for first hardware trials; the Go2 contract allows more.
    forward_speed_mps: Annotated[float, Field(strict=True, gt=0, le=1.0, allow_inf_nan=False)] = 0.3
    max_yaw_rate_rps: Annotated[float, Field(strict=True, gt=0, le=1.5, allow_inf_nan=False)] = 0.6
    lookahead_distance_m: PositiveSeconds = 0.4
    goal_tolerance_m: PositiveSeconds = 0.05
    slowdown_distance_m: PositiveSeconds = 0.2
    rotate_threshold_rad: PositiveSeconds = 0.6

    def pure_pursuit(self) -> PurePursuitConfig:
        return PurePursuitConfig(**self.model_dump())


class GoalSettings(Settings):
    arrival_tolerance_m: PositiveSeconds = 0.5
    turn_tolerance_rad: PositiveSeconds = 0.05


class ExecutorConfig(Settings):
    camera_calibration: Annotated[StrictStr, Field(min_length=1)]
    pose_source: Literal["dead_reckoning", "go2"] = "dead_reckoning"
    allow_assumed_calibration: StrictBool = False
    selection: Literal["medoid", "index"] = "medoid"
    candidate_index: Annotated[int, Field(strict=True, ge=0)] = 0
    min_path_length_m: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] = 0.05
    control_frequency_hz: Annotated[float, Field(strict=True, ge=10, le=100, allow_inf_nan=False)] = 20.0
    node_state_timeout_seconds: PositiveSeconds = 1.0
    result_validity_seconds: PositiveSeconds = 3.0
    shutdown_timeout_seconds: PositiveSeconds = 15.0
    go2: Go2Timing = Field(default_factory=Go2Timing)
    motion: ExecutorMotion = Field(default_factory=ExecutorMotion)
    goal: GoalSettings = Field(default_factory=GoalSettings)

    @model_validator(mode="after")
    def valid_controller(self):
        self.motion.pure_pursuit()
        if self.shutdown_timeout_seconds <= self.go2.down_timeout_seconds:
            raise ValueError("shutdown timeout must exceed the Go2 down timeout")
        return self


def load_executor_config(path: Path) -> ExecutorConfig:
    return ExecutorConfig.model_validate(json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False))


def load_camera(config: ExecutorConfig, config_path: Path) -> GroundCamera:
    path = Path(config.camera_calibration)
    camera = GroundCamera.load(path if path.is_absolute() else config_path.parent / path)
    if camera.calibration_status != "calibrated" and not config.allow_assumed_calibration:
        raise ValueError("camera calibration is not verified (calibration_status != calibrated); run "
                         "calibrate_camera.py verify --mark-calibrated, or set allow_assumed_calibration "
                         "for robot-free tests only")
    return camera


@dataclass(frozen=True)
class ExecutorOutput:
    velocity: VelocityCommand | None = None
    posture: Literal["stand", "down"] | None = None
    feedback: ExecutorState | None = None


@dataclass
class _Step:
    inference_id: int
    path: tuple[tuple[float, float], ...]
    received_at: float
    started_at: float
    distance_m: float = 0.0
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class _GoalReport:
    reached: bool
    image: tuple[float, float] | None
    distance_m: float


class Go2StepExecutor:
    """Thread-safe stop-go executor logic; Zenoh I/O lives in ExecutorRuntime."""

    def __init__(self, config: ExecutorConfig, stop_go: StopGoConfig, camera: GroundCamera,
                 clock: Callable[[], float] = time.monotonic):
        self.config, self.stop_go, self.clock = config, stop_go, clock
        self.projector = GroundProjector(camera)
        self.controller = PurePursuitController(config.motion.pure_pursuit())
        self.session_id = str(uuid4())
        self.lock = RLock()
        self.events = deque(maxlen=100)
        self.phase: Phase = "idle"
        self.error: str | None = None
        self.last_error: str | None = None
        self.node: NodeState | None = None
        self.node_at: float | None = None
        self.go2: Go2NodeState | None = None
        self.go2_at: float | None = None
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self._pose_valid = config.pose_source == "dead_reckoning"
        self._pose_delta = (0.0, 0.0)
        self.command = VelocityCommand()
        self._goal_anchor: tuple[tuple[str, int], tuple[float, float]] | None = None
        self._goal_report: _GoalReport | None = None
        self._turn: tuple[float, float] | None = None  # (target yaw, deadline)
        self._turns_without_view = 0
        self._cycle_key = None
        self._stopped_epoch = None
        self._pending: tuple[TrajectoryResult, float] | None = None
        self._step: _Step | None = None
        self._executed: int | None = None
        self._quiescent_since: float | None = None
        self._stand_deadline: float | None = None
        self._rest_deadline: float | None = None
        self._stop_error: str | None = None
        self._zero_pending = False
        self._posture_pending: Literal["stand", "down"] | None = None
        self._startup_rest_checked = False
        self._last_tick: float | None = None
        self.shutdown_requested = False
        self.shutdown_complete = False
        self.shutdown_error: str | None = None

    # Inputs -----------------------------------------------------------------
    def receive_node_state(self, state: NodeState):
        with self.lock:
            previous = self.node
            if (previous is not None and previous.node_session_id == state.node_session_id
                    and state.control_epoch < previous.control_epoch):
                return  # Out-of-order delivery cannot roll back control.
            self.node, self.node_at = state, self.clock()

    def receive_go2_state(self, state: Go2NodeState):
        with self.lock:
            self.go2, self.go2_at = state, self.clock()

    def receive_control(self, command: ControlCommand):
        with self.lock:
            if command.type == "stop" and self.node is not None:
                # Stop immediately; the node's own pause arrives with its next state.
                self._stopped_epoch = (self.node.node_session_id, self.node.control_epoch)

    def receive_result(self, result: TrajectoryResult):
        with self.lock:
            node = self.node
            if (node is None or result.node_session_id != node.node_session_id
                    or result.control_epoch != node.control_epoch or result.cycle_id != node.cycle.cycle_id):
                self._event("result_ignored", inference_id=result.inference_id, reason="obsolete cycle")
                return
            if self._pending is not None or self._step is not None or self._executed is not None:
                self._event("result_ignored", inference_id=result.inference_id, reason="duplicate")
                return
            # Movement still waits for the node's matching "moving" phase.
            self._pending = (result, self.clock())

    def request_shutdown(self):
        with self.lock:
            self.shutdown_requested = True

    # Observations -------------------------------------------------------------
    def _go2_fresh(self, now):
        return self.go2 is not None and now - self.go2_at <= self.config.go2.state_timeout_seconds

    def _go2_available(self, now):
        return self._go2_fresh(now) and self.go2.robot_connected and self.go2.accepting_commands

    def _resting(self):
        state = self.go2.robot_state
        return state.motion == "quiescent" and state.state in GO2_RESTING_STATES

    def _standing(self):
        return self.go2.robot_state.state in GO2_STANDING_STATES

    def _node_running(self, now):
        node = self.node
        return (node is not None and now - self.node_at <= self.config.node_state_timeout_seconds
                and node.inference.enabled
                and node.lifecycle not in ("error", "stopping", "stopped")
                and (node.node_session_id, node.control_epoch) != self._stopped_epoch)

    # Transitions --------------------------------------------------------------
    def _event(self, name, **details):
        self.events.append({"event": name, "phase": self.phase, **details})

    def _fail(self, message):
        if self.phase != "error":
            LOGGER.warning("Executor error: %s", message)
            self._event("error", message=message)
        self.phase, self.error = "error", message
        self._clear_step()

    def _clear_step(self):
        self._pending = self._step = self._turn = None
        self.command = VelocityCommand()

    def _begin_stop(self, now, reason):
        self._event("stop", reason=reason)
        self._stop_error = self.error
        self._clear_step()
        self._executed = None
        self.phase = "stopping"
        self._zero_pending = True
        self._posture_pending = "down" if self._go2_available(now) else None
        self._rest_deadline = now + self.config.go2.down_timeout_seconds
        self._stand_deadline = None

    def _finish_stop(self, error, rested):
        self.phase, self.error, self.last_error = "idle", None, error
        self._rest_deadline = None
        self._event("stopped", error=error, rested=rested)
        if not rested:
            LOGGER.warning("Stop finished without confirmed rest: %s", error)
        elif error:
            LOGGER.info("Go2 resting after: %s", error)
        if self.shutdown_requested:
            self.shutdown_complete, self.shutdown_error = True, None if rested else error

    def _advance_stop(self, now):
        if not self._go2_available(now):
            self._finish_stop(self._stop_error or "Go2 state became unavailable before rest was confirmed", False)
        elif self._resting():
            self._finish_stop(self._stop_error, True)
        elif now >= self._rest_deadline:
            self._finish_stop(self._stop_error or "Go2 did not confirm a resting state before the timeout", False)

    def _sync_cycle(self):
        node = self.node
        key = (node.node_session_id, node.control_epoch, node.cycle.cycle_id)
        if key == self._cycle_key:
            return
        epoch_changed = self._cycle_key is None or key[:2] != self._cycle_key[:2]
        self._cycle_key = key
        self._pending, self._executed, self._goal_report = None, None, None
        if self._step is not None or self._turn is not None:  # A new cycle never continues old motion.
            self._step = self._turn = None
            self.command = VelocityCommand()
            if self.phase in ("moving", "turning"):
                self.phase, self._quiescent_since = "holding", None
        if epoch_changed and self.phase == "error":
            self.phase, self.error = "idle", None
        elif self.phase == "settling":
            self.phase = "holding"

    # Pose and goal --------------------------------------------------------------
    def _update_pose(self, now, dt):
        before = self.pose
        if self.config.pose_source == "go2":
            state = self.go2.robot_state if self._go2_fresh(now) else None
            self._pose_valid = state is not None and state.position is not None and state.yaw is not None
            if self._pose_valid:
                self.pose = Pose2D(float(state.position[0]), float(state.position[1]), wrap(float(state.yaw)))
        else:
            # Integrate the command published during the last period.
            yaw = before.yaw + self.command.vyaw * dt
            self.pose = Pose2D(before.x + self.command.vx * math.cos(before.yaw) * dt,
                               before.y + self.command.vx * math.sin(before.yaw) * dt, wrap(yaw))
        self._pose_delta = (math.hypot(self.pose.x - before.x, self.pose.y - before.y),
                            abs(wrap(self.pose.yaw - before.yaw)))

    def _goal_relative(self) -> tuple[float, float]:
        (gx, gy), pose = self._goal_anchor[1], self.pose
        dx, dy = gx - pose.x, gy - pose.y
        cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
        return dx * cosine + dy * sine, -dx * sine + dy * cosine

    def _anchor_goal(self, node) -> bool:
        goal, camera = node.goal, self.projector.camera
        if goal is None:
            self._fail("the node has no goal")
            return False
        if goal.is_image:
            pixel = (min(goal.x * camera.image_width, camera.image_width - 1e-6),
                     min(goal.y * camera.image_height, camera.image_height - 1e-6))
            projected = self.projector.project_pixel(pixel)
            if projected.status != "ok":
                self._fail(f"image goal {goal.describe()} is not on visible ground in range ({projected.status})")
                return False
            relative = projected.point_robot_m
        else:
            relative = (goal.forward_m, goal.left_m)
        world = path_to_world([relative], self.pose)[0]
        self._goal_anchor = ((node.node_session_id, node.goal_revision), world)
        self._turns_without_view = 0
        self._event("goal_anchored", goal=goal.describe(), world=world, pose=asdict(self.pose))
        LOGGER.info("Goal %s anchored at world (%.2f, %.2f)", goal.describe(), *world)
        return True

    def _resolve_goal(self, now):
        """At a settled stop: report the goal's image point, its arrival, or turn toward it."""
        node, camera = self.node, self.projector.camera
        if self._goal_anchor is None or self._goal_anchor[0] != (node.node_session_id, node.goal_revision):
            if not self._anchor_goal(node):
                return
        forward, left = self._goal_relative()
        distance = math.hypot(forward, left)
        if distance <= self.config.goal.arrival_tolerance_m:
            self._goal_report = _GoalReport(True, None, distance)
            return
        pixel = self.projector.ground_to_pixel([(forward, left)])[0]
        if pixel is not None:
            self._turns_without_view = 0
            self._goal_report = _GoalReport(False, (pixel[0] / camera.image_width, pixel[1] / camera.image_height),
                                            distance)
            return
        bearing = math.atan2(left, forward)
        reach = min(camera.max_distance_m, 5.0)
        if self.projector.ground_to_pixel([(reach * math.cos(bearing), reach * math.sin(bearing))])[0] is not None:
            # In the horizontal field of view but nearer than the visible ground.
            self._event("goal_in_blind_zone", distance_m=distance)
            self._goal_report = _GoalReport(True, None, distance)
            return
        if self._turns_without_view >= MAX_TURNS_WITHOUT_VIEW:
            return self._fail("the goal stayed out of view after turning toward it")
        self._turns_without_view += 1
        deadline = now + abs(bearing) / self.config.motion.max_yaw_rate_rps + 3.0
        self._turn = (wrap(self.pose.yaw + bearing), deadline)
        self.phase = "turning"
        self._event("turn_toward_goal", bearing_rad=bearing, distance_m=distance)

    def _advance_turn(self, now):
        target, deadline = self._turn
        error = wrap(target - self.pose.yaw)
        if abs(error) <= self.config.goal.turn_tolerance_rad:
            self._turn, self.command = None, VelocityCommand()
            self.phase, self._quiescent_since = "holding", None
            return
        if now >= deadline:
            return self._fail("turning toward the goal timed out")
        limit = self.config.motion.max_yaw_rate_rps
        rate = max(-limit, min(limit, 2.0 * error))
        self.command = VelocityCommand(vyaw=math.copysign(max(abs(rate), min(limit, MIN_TURN_RATE_RPS)), error))

    def _start_step(self, now):
        node, (result, received_at) = self.node, self._pending
        self._pending = None
        if (result.inference_id != node.cycle.inference_id or result.goal_revision != node.goal_revision):
            return self._fail("result does not match the authorized inference")
        if now - received_at > self.config.result_validity_seconds:
            return self._fail("result expired before the node authorized movement")
        try:
            self.projector.camera.check_image_size(result.image_width, result.image_height)
        except ValueError as error:
            return self._fail(str(error))
        selection = select_candidate(result.trajectories, self.projector, strategy=self.config.selection,
                                     index=self.config.candidate_index, min_length_m=self.config.min_path_length_m)
        self._event("selection", inference_id=result.inference_id, candidate=selection.index,
                    reason=selection.reason, candidates=[report.as_dict() for report in selection.candidates])
        if not selection.usable:
            return self._fail(f"no executable path: {selection.reason}")
        # The robot has been held still since the observed frame, so the
        # capture-frame path starts at the current dead-reckoned pose.
        path = path_to_world(selection.path.points_robot_m, self.pose)
        self._step = _Step(result.inference_id, path, received_at, now)
        self.phase = "moving"
        LOGGER.info("Step for inference=%d candidate=%s (%s)", result.inference_id, selection.index,
                    selection.reason)

    def _finish_step(self, reason):
        step = self._step
        self._event("step_finished", inference_id=step.inference_id, reason=reason,
                    distance_m=step.distance_m, yaw_rad=step.yaw_rad, pose=asdict(self.pose))
        self._executed, self._step = step.inference_id, None
        self.command = VelocityCommand()
        self._quiescent_since = None
        self.phase = "settling"

    def _advance_step(self, now):
        step, bounds = self._step, self.stop_go
        step.distance_m += self._pose_delta[0]
        step.yaw_rad += self._pose_delta[1]
        if now - step.received_at > self.config.result_validity_seconds:
            return self._finish_step("result_validity")
        if now - step.started_at >= bounds.max_step_seconds:
            return self._finish_step("max_step_seconds")
        if step.distance_m >= bounds.max_step_distance_m:
            return self._finish_step("max_step_distance")
        if step.yaw_rad >= bounds.max_step_yaw_rad:
            return self._finish_step("max_step_yaw")
        tracking = self.controller.compute(step.path, self.pose)
        if tracking.status == "reached":
            return self._finish_step("path_end")
        if tracking.status not in ("tracking", "rotating"):
            return self._fail(f"path tracking stopped: {tracking.status}")
        self.command = tracking.command

    def _track_quiescence(self, now):
        moving = self.command != VelocityCommand() or self.go2.robot_state.motion != "quiescent"
        if moving:
            self._quiescent_since = None
        elif self._quiescent_since is None:
            self._quiescent_since = now

    def _settled(self, now):
        return (self._quiescent_since is not None
                and now - self._quiescent_since >= self.stop_go.stop_settle_seconds)

    # Control loop -------------------------------------------------------------
    def tick(self, now: float | None = None) -> ExecutorOutput:
        with self.lock:
            now = self.clock() if now is None else now
            dt = 0.0 if self._last_tick is None else min(0.1, max(0.0, now - self._last_tick))
            self._last_tick = now
            self._update_pose(now, dt)
            if self.shutdown_requested and not self.shutdown_complete and self.phase != "stopping":
                if self.phase in ACTIVE_PHASES or (self._go2_available(now) and not self._resting()):
                    self._begin_stop(now, "shutdown")
                else:
                    available = self._go2_available(now)
                    self._finish_stop(None if available else "Go2 state unavailable at shutdown", available)
            if self.phase == "stopping":
                self._advance_stop(now)
            elif not self._node_running(now) or self.shutdown_requested:
                if self.phase in ACTIVE_PHASES:
                    self._begin_stop(now, "node not running" if not self.shutdown_requested else "shutdown")
                elif (not self._startup_rest_checked and self._go2_available(now)):
                    self._startup_rest_checked = True
                    if not self._resting():
                        self._begin_stop(now, "robot not resting at startup")
            else:
                self._run_cycle(now)
            return self._outputs(now)

    def _run_cycle(self, now):
        self._sync_cycle()
        if self.phase == "error":
            self.command = VelocityCommand()
            return
        if not self._go2_available(now):
            return self._fail("Go2 state is stale, disconnected or not accepting commands")
        if not self._pose_valid:
            return self._fail("Go2 state has no position/yaw; use pose_source dead_reckoning")
        self._startup_rest_checked = True
        if self.phase == "idle":
            if self._standing():
                self.phase, self._quiescent_since = "holding", None
            elif self._resting():
                self.phase, self._posture_pending = "standing", "stand"
                self._stand_deadline = now + self.config.go2.stand_timeout_seconds
                self._event("stand_requested")
            else:
                return self._fail(f"Go2 is not in a startable posture: {self.go2.robot_state.state}")
        if self.phase == "standing":
            if self._standing():
                self.phase, self._quiescent_since, self._stand_deadline = "holding", None, None
            elif now >= self._stand_deadline:
                return self._fail("Go2 did not reach a standing state before the timeout")
            else:
                return
        if not self._standing():
            return self._fail(f"Go2 left the standing state: {self.go2.robot_state.state}")
        if self.phase == "turning":
            self._advance_turn(now)
        elif self.phase == "moving":
            self._advance_step(now)
        elif self.phase == "holding" and self._settled(now):
            if self.node.cycle.phase == "waiting_stop" and self._goal_report is None:
                self._resolve_goal(now)
                if self.phase == "turning":
                    self._advance_turn(now)
            elif self.node.cycle.phase == "moving" and self._pending is not None:
                self._start_step(now)
                if self.phase == "moving":
                    self._pose_delta = (0.0, 0.0)
                    self._advance_step(now)
        self._track_quiescence(now)

    def _outputs(self, now) -> ExecutorOutput:
        velocity = None
        if self._zero_pending:
            velocity, self._zero_pending = VelocityCommand(), False
        elif (self.phase in ("holding", "turning", "moving", "settling")
                or (self.phase == "error" and self._standing_now(now))):
            velocity = self.command
        posture, self._posture_pending = self._posture_pending, None
        return ExecutorOutput(velocity, posture, self._feedback(now))

    def _standing_now(self, now):
        return self._go2_fresh(now) and self._standing()

    def _feedback(self, now) -> ExecutorState | None:
        node = self.node
        if node is None or not node.node_session_id:
            return None
        if self.phase == "error":
            motion, message = "error", self.error
        elif self.phase == "moving":
            motion, message = "moving", "executing step"
        elif self.phase == "turning":
            motion, message = "moving", "turning toward the goal"
        elif self.phase in ("holding", "settling"):
            motion = "stopped" if self._settled(now) else "settling"
            message = "standing still" if motion == "stopped" else "settling"
            if (motion == "stopped" and self.phase == "holding" and node.cycle.phase == "waiting_stop"
                    and self._goal_report is None):
                motion, message = "settling", "locating the goal"
        elif self.phase == "idle" and self._go2_available(now) and self.go2.robot_state.motion == "quiescent":
            motion, message = "stopped", "idle; start stands the robot first"
        elif self.phase == "idle" and not self._go2_available(now):
            motion, message = "error", "Go2 state is stale, disconnected or not accepting commands"
        else:
            motion, message = "settling", self.phase
        report = self._goal_report if motion == "stopped" and self.phase == "holding" else None
        return ExecutorState(executor_session_id=self.session_id,
                             node_session_id=node.node_session_id, control_epoch=node.control_epoch,
                             cycle_id=node.cycle.cycle_id, motion=motion,
                             inference_id=self._executed if self.phase == "settling" else None,
                             pose=PlanarPose(**asdict(self.pose)),
                             path_m=list(self._step.path) if self._step is not None else [],
                             goal_image=None if report is None else report.image,
                             goal_distance_m=None if report is None else report.distance_m,
                             goal_reached=report is not None and report.reached,
                             goal_world_m=None if self._goal_anchor is None else self._goal_anchor[1],
                             message=message or "")

    def snapshot(self) -> dict:
        with self.lock:
            return {"phase": self.phase, "error": self.error, "last_error": self.last_error,
                    "command": self.command.as_message(), "pose": asdict(self.pose),
                    "goal_world": None if self._goal_anchor is None else self._goal_anchor[1],
                    "go2_state": None if self.go2 is None else self.go2.robot_state.state,
                    "events": list(self.events)}


def has_encoding(sample, expected: str) -> bool:
    return str(sample.encoding).split(";", 1)[0] == expected


class ExecutorRuntime:
    def __init__(self, executor: Go2StepExecutor, node_config: NodeConfig):
        self.executor = executor
        self.keys = node_config.keys()

    def _callback(self, model, handler, name):
        def receive(sample):
            try:
                if not has_encoding(sample, JSON_ENCODING):
                    raise ValueError("expected application/json")
                handler(model.model_validate_json(sample.payload.to_bytes()))
            except ValueError as error:
                LOGGER.warning("Rejected %s: %s", name, error)
        return receive

    def run(self, zenoh_config: zenoh.Config, stop: Event) -> None:
        executor, keys = self.executor, self.keys
        period = 1.0 / executor.config.control_frequency_hz
        with zenoh.open(zenoh_config) as session, ExitStack() as resources:
            velocity = resources.enter_context(session.declare_publisher(
                keys["go2_command"], encoding=JSON_ENCODING,
                congestion_control=zenoh.CongestionControl.DROP, reliability=zenoh.Reliability.BEST_EFFORT))
            posture = resources.enter_context(session.declare_publisher(
                keys["go2_command"], encoding=JSON_ENCODING,
                congestion_control=zenoh.CongestionControl.BLOCK, reliability=zenoh.Reliability.RELIABLE))
            feedback = resources.enter_context(session.declare_publisher(
                keys["executor_state"], encoding=JSON_ENCODING,
                congestion_control=zenoh.CongestionControl.DROP, reliability=zenoh.Reliability.BEST_EFFORT))
            for key, model, handler, name in (
                    (keys["state"], NodeState, executor.receive_node_state, "VAMOS state"),
                    (keys["trajectories"], TrajectoryResult, executor.receive_result, "VAMOS result"),
                    (keys["command"], ControlCommand, executor.receive_control, "control command"),
                    (keys["go2_state"], Go2NodeState, executor.receive_go2_state, "Go2 state")):
                resources.enter_context(session.declare_subscriber(key, self._callback(model, handler, name)))
            LOGGER.info("Go2 executor ready: commands=%s feedback=%s", keys["go2_command"], keys["executor_state"])
            deadline = time.monotonic()
            shutdown_deadline = None
            while True:
                if stop.is_set() and shutdown_deadline is None:
                    LOGGER.info("Shutdown requested; stopping the robot")
                    executor.request_shutdown()
                    shutdown_deadline = time.monotonic() + executor.config.shutdown_timeout_seconds
                output = executor.tick()
                if output.velocity is not None:
                    velocity.put(json.dumps(output.velocity.as_message()))
                if output.posture is not None:
                    LOGGER.info("Requesting Go2 posture %s", output.posture)
                    posture.put(json.dumps({"type": "posture", "posture": output.posture}))
                if output.feedback is not None:
                    feedback.put(output.feedback.model_dump_json())
                if executor.shutdown_complete:
                    break
                if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                    executor.shutdown_error = "shutdown timed out before Go2 rest was confirmed"
                    break
                now = time.monotonic()
                deadline += (max(0, math.floor((now - deadline) / period)) + 1) * period
                time.sleep(max(0.0, deadline - time.monotonic()))
        if executor.shutdown_error:
            raise RuntimeError(executor.shutdown_error)


def main() -> int:
    from node import stop_on_signals, tolerate_unencodable_output

    tolerate_unencodable_output()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node-config", type=Path, default=HERE / "inference-node-config.example.json5",
                        help="VAMOS node settings: keys and stop-go step bounds")
    parser.add_argument("--executor-config", type=Path, default=HERE / "go2-executor-config.example.json5")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "peer-zenoh-config.example.json5")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        node_config = load_node_config(args.node_config)
        if node_config.inference is None:
            raise ValueError("the Go2 executor requires a node config with an inference section")
        config = load_executor_config(args.executor_config)
        camera = load_camera(config, args.executor_config)
        executor = Go2StepExecutor(config, node_config.inference.stop_go, camera)
        with stop_on_signals() as stop:
            ExecutorRuntime(executor, node_config).run(zenoh.Config.from_file(str(args.zenoh_config)), stop)
    except (OSError, ValueError, RuntimeError, zenoh.ZError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
