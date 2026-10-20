"""Start and operate VAMOS inference, with optional local demo inputs and views."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import logging
import math
from pathlib import Path
import subprocess
from threading import Condition, Event, Thread
import time
from uuid import uuid4

import zenoh

from config import ControlCommand, GoalConfig, JSON_ENCODING, NodeState, load_node_config
from node import HERE, has_encoding
from launcher import ManagedProcesses, check_listener_available, listener_addresses


class StateCache:
    def __init__(self, timeout: float, clock=time.monotonic):
        self.timeout, self.clock = timeout, clock
        self.condition = Condition()
        self.state = None
        self.received_at = None

    def store(self, state: NodeState):
        with self.condition:
            self.state, self.received_at = state, self.clock()
            self.condition.notify_all()

    def snapshot(self) -> NodeState | None:
        with self.condition:
            if self.received_at is None or self.clock() - self.received_at > self.timeout:
                return None
            return self.state

    def wait(self, predicate, timeout: float) -> NodeState | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                state = self.snapshot()
                if state is not None and predicate(state):
                    return state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(min(remaining, 0.1))


class CliRuntime:
    def __init__(self, config, *, heartbeat_interval=0.2, state_timeout=1.0,
                 command_timeout=15.0, output=print, processes=None):
        values = (heartbeat_interval, state_timeout, command_timeout)
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("CLI intervals/timeouts must be finite and positive")
        if heartbeat_interval >= config.cli_heartbeat_timeout_seconds:
            raise ValueError("heartbeat interval must be below the node heartbeat timeout")
        if state_timeout <= 1 / config.node_state_publish_frequency_hz or command_timeout <= state_timeout:
            raise ValueError("state timeout must exceed the state period; command timeout must exceed state timeout")
        self.config = config
        self.heartbeat_interval = heartbeat_interval
        self.command_timeout = command_timeout
        self.cache = StateCache(state_timeout)
        self.output = output
        self.processes = processes
        self._publisher = None
        self._heartbeat_publisher = None
        self._stop_heartbeat = Event()
        self._heartbeat_error = None

    def _on_state(self, sample):
        try:
            if not has_encoding(sample, JSON_ENCODING):
                raise ValueError("expected application/json")
            self.cache.store(NodeState.model_validate_json(sample.payload.to_bytes()))
        except ValueError as error:
            logging.warning("Rejected invalid VAMOS state: %s", error)

    def _heartbeat(self):
        try:
            while not self._stop_heartbeat.is_set():
                self._heartbeat_publisher.put('{"type":"heartbeat"}')
                self._stop_heartbeat.wait(self.heartbeat_interval)
        except Exception as error:
            self._heartbeat_error = error
            logging.error("Heartbeat failed; inference will pause when its heartbeat expires: %s", error)

    @contextmanager
    def connect(self, zenoh_config):
        keys = self.config.keys()
        self.cache = StateCache(self.cache.timeout)
        self._stop_heartbeat.clear()
        self._heartbeat_error = None
        with zenoh.open(zenoh_config) as session, ExitStack() as resources:
            resources.enter_context(session.declare_subscriber(keys["state"], self._on_state))
            self._publisher = resources.enter_context(session.declare_publisher(
                keys["command"], encoding=JSON_ENCODING,
                reliability=zenoh.Reliability.RELIABLE, congestion_control=zenoh.CongestionControl.BLOCK))
            self._heartbeat_publisher = resources.enter_context(session.declare_publisher(
                keys["command"], encoding=JSON_ENCODING,
                reliability=zenoh.Reliability.BEST_EFFORT, congestion_control=zenoh.CongestionControl.DROP))
            worker = Thread(target=self._heartbeat, name="vamos-cli-heartbeat", daemon=True)
            worker.start()
            try:
                if self.cache.wait(lambda state: state.cli_connected, self.command_timeout) is None:
                    raise TimeoutError("No VAMOS state received; check the node and connection settings")
                yield self
            finally:
                # Keep heartbeat and state subscription alive while draining the
                # final stop acknowledgement, even on EOF, Ctrl+C or exceptions.
                try:
                    self.stop()
                except (ValueError, RuntimeError, TimeoutError, zenoh.ZError) as error:
                    self.output(f"Stop could not be confirmed: {error}")
                finally:
                    self._stop_heartbeat.set()
                    worker.join(self.heartbeat_interval + 1)
                    self._publisher = self._heartbeat_publisher = None
            if self._heartbeat_error is not None:
                raise RuntimeError(f"heartbeat worker failed: {self._heartbeat_error}")

    def state(self) -> NodeState:
        if self.processes is not None:
            self.processes.check_required()
        if self._heartbeat_error is not None:
            raise RuntimeError(f"heartbeat worker failed: {self._heartbeat_error}")
        state = self.cache.snapshot()
        if state is None:
            raise RuntimeError("VAMOS node disconnected: no fresh state")
        return state

    def _request(self, kind: str, *, goal=None) -> NodeState:
        # Source/executor failures must not prevent an operator stop reaching
        # a still-connected node. Other controls retain service health checks.
        state = self.cache.snapshot() if kind == "stop" else self.state()
        if state is None:
            raise RuntimeError("VAMOS node disconnected: no fresh state")
        if not state.node_session_id:
            raise RuntimeError("Node does not support acknowledged controls; update the node")
        command = ControlCommand(type=kind, request_id=str(uuid4()), node_session_id=state.node_session_id,
                                 expected_goal_revision=state.goal_revision if kind in ("start", "goal") else None,
                                 goal=goal)
        if self._publisher is None:
            raise RuntimeError("CLI is not connected")
        self._publisher.put(command.model_dump_json(exclude_none=True))

        def acknowledged(update):
            if update.node_session_id != state.node_session_id:
                raise RuntimeError("Node restarted during the command; refresh state")
            ack = update.last_control_ack
            if ack is None or ack.request_id != command.request_id:
                return False
            if not ack.accepted:
                raise ValueError(f"Node rejected {kind}: {ack.message}")
            return kind != "stop" or (not update.inference.enabled and not update.inference.active
                and (update.stage != "inference" or
                     (update.cycle.executor_fresh and update.cycle.executor_motion == "stopped")))

        update = self.cache.wait(acknowledged, self.command_timeout)
        if update is None:
            raise TimeoutError(f"{kind} was not confirmed before timeout; request may have reached the node; check status")
        return update

    def start(self):
        state = self.state()
        if state.stage != "inference":
            raise ValueError("Connected node is transport-only; restart it with "
                             "--node-config zenoh/inference-node-config.example.json5")
        if state.inference.enabled:
            self.output("Inference is already enabled.")
            return state
        if state.inference.active:
            self.output("Waiting for the previous inference to finish before starting.")
            update = self.cache.wait(lambda value: value.node_session_id != state.node_session_id
                                     or not value.inference.active, self.command_timeout)
            if update is None:
                raise TimeoutError("Previous inference has not finished; start was not sent; check status")
            if update.node_session_id != state.node_session_id:
                raise RuntimeError("Node restarted while waiting; refresh state before starting")
        update = self.cache.wait(lambda value: value.node_session_id != state.node_session_id or
            (value.cycle.executor_fresh and value.cycle.executor_motion == "stopped"), self.command_timeout)
        if update is None:
            raise TimeoutError("No stopped executor confirmed; check go2_executor.py and Go2 state")
        if update.node_session_id != state.node_session_id:
            raise RuntimeError("Node restarted while waiting for the executor")
        update = self._request("start")
        self.output(f"Inference started (goal revision {update.goal_revision}).")
        return update

    def stop(self):
        # Even an apparently paused node receives a correlated stop, avoiding
        # a stale cached observation being mistaken for confirmation.
        update = self._request("stop")
        self.output("Stopped; pending inference finished" +
                    (" and the executor confirmed stopped." if update.stage == "inference" else "."))
        return update

    def set_goal(self, goal: GoalConfig):
        state = self.state()
        if state.stage != "inference":
            raise ValueError("Goal changes require an inference node")
        if state.inference.enabled or state.inference.active:
            self.stop()
        update = self._request("goal", goal=goal)
        self.output(f"Goal updated (revision {update.goal_revision}); use start to resume.")
        return update

    def status(self):
        state = self.state()
        inf = state.inference
        self.output(f"state={state.lifecycle} model_loaded={inf.model_loaded} device={inf.device} "
                    f"enabled={inf.enabled} active={inf.active}")
        self.output(f"camera_fresh={state.camera.fresh} heartbeat={state.cli_connected} "
                    f"completed={inf.completed} discarded={inf.discarded} last_seconds={inf.last_duration_seconds}")
        go2 = state.go2
        self.output(f"start_ready={state.start_ready} go2_fresh={go2.fresh} go2_state="
                    f"{go2.robot_state.state if go2.robot_state else None} accepting={go2.accepting_commands}")
        if state.goal:
            self.output(f"goal=({state.goal.x:g}, {state.goal.y:g}) revision={state.goal_revision} "
                        f"instruction={state.goal.instruction!r}")
        self.output(state.status_message)
        if state.stage == "inference":
            cycle = state.cycle
            self.output(f"cycle={cycle.cycle_id} phase={cycle.phase} completed_steps={cycle.completed} "
                        f"executor_fresh={cycle.executor_fresh} motion={cycle.executor_motion}")
        if inf.last_error:
            self.output(f"error={inf.last_error}")
        return state

    def execute(self, line: str) -> bool:
        name, _, rest = line.strip().partition(" ")
        name = name.lower()
        if not name:
            return True
        if name in ("quit", "exit"):
            return False
        if name in ("start", "stop", "status"):
            if rest.strip():
                raise ValueError(f"Usage: {name}")
            getattr(self, name)()
        elif name == "goal":
            parts = rest.split()
            if len(parts) != 2:
                raise ValueError("Usage: goal <x 0..1> <y 0..1>")
            current = self.state().goal
            if current is None:
                raise ValueError("This node has no goal")
            self.set_goal(GoalConfig(x=float(parts[0]), y=float(parts[1]), instruction=current.instruction))
        elif name == "ins":
            current = self.state().goal
            if current is None:
                raise ValueError("This node has no goal")
            self.set_goal(GoalConfig(x=current.x, y=current.y, instruction=rest.strip()))
        elif name == "watch":
            seconds = float(rest.strip() or 10)
            if not math.isfinite(seconds) or not 0 < seconds <= 3600:
                raise ValueError("watch duration must be in (0, 3600] seconds")
            deadline = time.monotonic() + seconds
            try:
                while time.monotonic() < deadline:
                    try:
                        self.status()
                    except RuntimeError as error:
                        self.output(str(error))
                    self._stop_heartbeat.wait(min(1, max(0, deadline - time.monotonic())))
            except KeyboardInterrupt:
                self.output("Watch stopped; inference remains in its current state.")
        elif name == "help":
            self.output("start | stop | status | goal <x 0..1> <y 0..1> | ins <text> | watch [seconds] | quit")
            self.output("processes | logs [node|camera|go2|executor|viewer]")
            self.output("Goal/instruction changes stop inference and leave it paused. This CLI sends no Go2 commands.")
        elif name in ("processes", "logs"):
            if self.processes is None:
                raise ValueError("No process manager is available")
            if name == "processes":
                if rest.strip():
                    raise ValueError("Usage: processes")
                self.processes.status()
            else:
                self.processes.logs(rest.strip() or "node")
        else:
            raise ValueError("Unknown command; use help")
        return True

    def run(self, zenoh_config, input_fn=input):
        with self.connect(zenoh_config):
            self.execute("help")
            self.status()
            while True:
                try:
                    line = input_fn("vamos> ")
                except (EOFError, KeyboardInterrupt):
                    break
                try:
                    if not self.execute(line):
                        break
                except KeyboardInterrupt:
                    break
                except (ValueError, RuntimeError, TimeoutError, zenoh.ZError) as error:
                    self.output(str(error))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, default=HERE / "inference-node-config.example.json5")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "peer-zenoh-config.example.json5")
    parser.add_argument("--heartbeat-interval", type=float, default=0.2)
    parser.add_argument("--state-timeout", type=float, default=1)
    parser.add_argument("--command-timeout", type=float, default=15)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="Robot-free demo: test camera, simulated Go2 and go2_executor.py with the demo config")
    mode.add_argument("--attach", action="store_true", help="Connect to an already running node instead of launching one")
    parser.add_argument("--viewer", action="store_true", help="Open the image-space path viewer")
    parser.add_argument("--simulator", action="store_true", help="Open the simulated Go2 window (requires --demo)")
    parser.add_argument("--node-zenoh-config", type=Path, default=HERE / "zenoh-config.example.json5")
    parser.add_argument("--image", type=Path, help="Demo image override (requires --demo)")
    parser.add_argument("--log-dir", type=Path, default=HERE.parent / "outputs/cli")
    args = parser.parse_args()
    if (args.image or args.simulator) and not args.demo:
        parser.error("--image and --simulator require --demo")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_node_config(args.node_config)
        if not args.attach and config.inference is None:
            raise ValueError("Local startup requires a node configuration with inference settings")
        with ManagedProcesses(args.log_dir,
                shutdown_timeout=5 + (config.inference.shutdown_timeout_seconds if config.inference else 30)) as processes:
            runtime = CliRuntime(config, heartbeat_interval=args.heartbeat_interval, state_timeout=args.state_timeout,
                                 command_timeout=args.command_timeout, processes=processes)
            shared = ("--node-config", args.node_config.resolve(), "--zenoh-config", args.zenoh_config.resolve())
            if not args.attach:
                addresses = listener_addresses(args.node_zenoh_config)
                check_listener_available(addresses)
                processes.start("node", HERE / "node.py", "--node-config", args.node_config.resolve(),
                                "--zenoh-config", args.node_zenoh_config.resolve())
                processes.wait_for_listener(addresses)
            if args.demo:
                # The CLI owns heartbeat. The executor is the real one, driving a simulated Go2.
                tools = HERE / "tools"
                camera_args = [*shared, *(("--image", args.image.resolve()) if args.image else ())]
                processes.start("camera", tools / "fake_camera.py", *camera_args)
                processes.start("go2", tools / "fake_go2.py", *shared, *(() if args.simulator else ("--headless",)))
                processes.start("executor", HERE / "go2_executor.py", *shared,
                                "--executor-config", tools / "demo-executor-config.json5")
            if args.viewer:
                processes.start("viewer", HERE / "tools" / "viewer.py", *shared)
            runtime.run(zenoh.Config.from_file(str(args.zenoh_config)))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired, zenoh.ZError) as error:
        logging.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
