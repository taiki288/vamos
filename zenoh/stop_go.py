"""Ideal 2D short-step executor for stop/observe/infer/move cycles.

The same static image is explicitly reused at each simulated stop pose. It does
not provide moving-camera perception or claim actual Go2 stopped feedback.
"""

from dataclasses import asdict
import math
import time
from uuid import uuid4

from config import CapturePose, ExecutorState, StopGoConfig
from controller import VelocityCommand
from stream_simulator import StreamFollower


class StopGoFollower(StreamFollower):
    def __init__(self, camera, settings: StopGoConfig, candidate=0, path_timeout=3.0,
                 state_timeout=1.0, clock=time.monotonic, wall_clock=time.time):
        self.settings = settings
        self.executor_session_id = str(uuid4())
        super().__init__(camera, candidate, path_timeout, state_timeout, clock, wall_clock)

    def reset(self):
        running = self.node_state is not None and self.node_state.inference.enabled
        super().reset()
        self._pending = None
        self._executed = None
        self._step_started = None
        self._step_finished = False
        self._step_distance = self._step_yaw = 0.0
        self._stopped_since = self.clock()
        self._error = "Local reset interrupted the cycle" if running else None
        self.steps_completed = 0

    def _zero(self, reason):
        if self.command != VelocityCommand():
            self._stopped_since = self.clock()
        self._invalidate(reason)

    def receive_state(self, state):
        with self.lock:
            previous = self.node_state
            changed = previous is None or (previous.node_session_id, previous.control_epoch, previous.cycle.cycle_id) != (
                state.node_session_id, state.control_epoch, state.cycle.cycle_id)
            if (previous is not None and previous.node_session_id == state.node_session_id
                    and state.control_epoch < previous.control_epoch):
                return
            if changed:
                self._zero("waiting_for_stop")
                self._pending = self._executed = self._step_started = None
                self._step_finished = False
                self._error = None
                self.cutoff_wall = self.wall_clock()
            super().receive_state(state)

    def receive_control(self, command):
        with self.lock:
            if command.type == "stop":
                self._zero("remote_stop")
                self._pending = None
            super().receive_control(command)

    def path_origin(self, capture):
        # The executor stays stopped during capture/inference. Reusing a static
        # image at this synthetic pose is a scheduling demo, not sensor odometry.
        return self.pose

    def expiry_origin(self, result):
        return result.generated_at_unix_seconds

    def result_age(self, result):
        # Static-image demo only: the pose is held throughout bounded inference.
        # Keep the original capture/source age; expire execution from generation.
        return self.wall_clock() - result.generated_at_unix_seconds

    def receive_result(self, result):
        with self.lock:
            state = self.node_state
            if (state is None or result.node_session_id != state.node_session_id
                    or result.control_epoch != state.control_epoch or result.goal_revision != state.goal_revision
                    or result.cycle_id != state.cycle.cycle_id):
                self.reject("obsolete_cycle_result", invalidate=False)
                return False
            if (not state.inference.enabled or self.remote_stopped or self.paused
                    or state.cycle.phase == "paused"):
                self.reject("cycle_not_accepting_results", invalidate=False)
                return False
            if self._executed is not None or self._pending is not None:
                self.reject("duplicate_cycle_result", invalidate=False)
                return False
            # A fast inference can finish between periodic node snapshots.
            # Buffer its result, but never move before a matching moving state.
            self._pending = result
            return True  # Actual path acceptance waits for the node's moving phase.

    def _apply_pending(self):
        state, result = self.node_state, self._pending
        if result is None or self._executed is not None or state.cycle.phase != "moving":
            return
        if result.inference_id != state.cycle.inference_id:
            self._error = "Result does not match the authorized inference"
            self._pending = None
            return
        cutoff = state.cycle.capture_after_unix_seconds
        if (cutoff is None or result.capture is None
                or result.capture.captured_at_unix_seconds <= cutoff):
            self._error = "Result was not captured after the stop confirmation"
        elif (result.generated_at_unix_seconds is None
                or result.generated_at_unix_seconds < result.capture.captured_at_unix_seconds
                or not math.isfinite(result.duration_seconds) or result.duration_seconds < 0
                or result.duration_seconds > self.settings.inference_timeout_seconds
                or result.source_age_seconds > self.settings.inference_timeout_seconds
                or result.generated_at_unix_seconds - result.capture.captured_at_unix_seconds
                    > self.settings.inference_timeout_seconds + 0.25):
            self._error = "Result exceeded stopped inference budget or is missing its generation time"
        elif not super().receive_result(result):
            self._error = f"Cannot execute path: {self.status}"
        else:
            self._executed = result.inference_id
            self._step_started = self.clock()
            self._step_distance = self._step_yaw = 0.0
            self.events.append({"event": "step_started", "cycle_id": result.cycle_id,
                                "anchor_source": "simulated_stop_pose", "pose": asdict(self.pose)})
        self._pending = None

    def _finish_step(self):
        if not self._step_finished:
            self._step_finished = True
            self.steps_completed += 1
            self.events.append({"event": "step_finished", "cycle_id": self.node_state.cycle.cycle_id,
                                "pose": asdict(self.pose), "distance_m": self._step_distance,
                                "turn_rad": self._step_yaw})
        self._zero("settling")

    def tick(self, dt=0.05):
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("dt must be in (0, 0.1]")
        with self.lock:
            reason = self._ready_reason()
            if reason:
                self._zero(reason)
                return
            if self.node_state.cycle.mode != "stop_go":
                self._error = "Node is not configured for stop-go execution"
            if self.paused:
                self._error = "Local pause interrupted the cycle"
            if self._error:
                self._zero("cycle_error")
                return
            if self.node_state.cycle.phase != "moving":
                self._zero(self.node_state.cycle.phase)
                return
            self._apply_pending()
            if self._error or self._executed is None:
                self._zero("waiting_for_path" if not self._error else "cycle_error")
                return
            if self._step_finished:
                self._zero("step_stopped")
                return
            age = self.path_age()
            if age > self.path_timeout or age < -0.25:
                self._error = "Path expired during execution"
                self._zero("cycle_error")
                return
            settings = self.settings
            if (self.clock() - self._step_started >= settings.max_step_seconds
                    or self._step_distance >= settings.max_step_distance_m
                    or self._step_yaw >= settings.max_step_yaw_rad
                    or age + dt >= self.path_timeout):
                self._finish_step()
                return
            before = self.pose
            super().tick(dt)
            distance = math.hypot(self.pose.x - before.x, self.pose.y - before.y)
            yaw = abs(math.atan2(math.sin(self.pose.yaw - before.yaw), math.cos(self.pose.yaw - before.yaw)))
            self._step_distance += distance
            self._step_yaw += yaw
            if distance > 0 or yaw > 0 or self.command != VelocityCommand():
                self._stopped_since = self.clock()
            if self.status not in ("tracking", "rotating", "reached"):
                self._error = f"Path execution stopped: {self.status}"
                self._zero("cycle_error")
            elif (self.status == "reached" or self._step_distance >= settings.max_step_distance_m
                    or self._step_yaw >= settings.max_step_yaw_rad):
                self._finish_step()

    def feedback(self):
        with self.lock:
            state = self.node_state
            if state is None or not state.node_session_id:
                return None
            error = self._error or ("Local pause interrupted the cycle" if self.paused else None)
            motion = ("error" if error else "moving" if self.command != VelocityCommand()
                      else "settling" if self.clock() - self._stopped_since < self.settings.stop_settle_seconds
                      else "stopped")
            return ExecutorState(executor_session_id=self.executor_session_id, node_session_id=state.node_session_id,
                control_epoch=state.control_epoch, cycle_id=state.cycle.cycle_id, motion=motion,
                inference_id=self._executed if self._step_finished else None,
                pose=CapturePose(**asdict(self.pose)), message=error or self.status)

    def snapshot(self):
        with self.lock:
            return {**super().snapshot(), "cycle_id": self.node_state.cycle.cycle_id if self.node_state else 0,
                    "cycle_phase": self.node_state.cycle.phase if self.node_state else "paused",
                    "steps_completed": self.steps_completed, "step_distance_m": self._step_distance,
                    "step_yaw_rad": self._step_yaw, "cycle_error": self._error}
