"""Print the Go2 state once per interval, with odometry change since start.

For hardware checks: compare the printed distance and heading change with a
tape measure and floor marks, and confirm that position moves along yaw.
Read-only; it never publishes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zenoh

from config import DEFAULT_CONFIG_PATH, load_config
from node import stop_on_signals


def describe(robot_state: dict, origin: dict | None) -> str:
    text = (f"state={robot_state.get('state')} motion={robot_state.get('motion')} "
            f"velocity={robot_state.get('velocity')} yaw_speed={robot_state.get('yaw_speed')}")
    position, yaw = robot_state.get("position"), robot_state.get("yaw")
    if position is None or yaw is None:
        return text + " position/yaw=not published"
    text += f" position=({position[0]:.3f}, {position[1]:.3f}) yaw={math.degrees(yaw):.1f}deg"
    if origin is not None:
        dx, dy = position[0] - origin["position"][0], position[1] - origin["position"][1]
        turn = math.degrees(math.atan2(math.sin(yaw - origin["yaw"]), math.cos(yaw - origin["yaw"])))
        heading = f"{math.degrees(math.atan2(dy, dx)):.1f}deg" if math.hypot(dx, dy) > 0.05 else "-"
        text += f" | moved={math.hypot(dx, dy):.3f} m turned={turn:+.1f}deg motion_heading={heading}"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Settings file (default zenoh/vamos.json5)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between lines")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    try:
        config = load_config(args.config)
        key = config.node_config().keys()["go2_state"]
        state = {"latest": None, "origin": None, "printed": 0.0}

        def receive(sample):
            try:
                state["latest"] = json.loads(sample.payload.to_bytes())["robot_state"]
            except (ValueError, KeyError, TypeError):
                pass

        with stop_on_signals() as stop, zenoh.open(config.zenoh.session("client")) as session, \
                session.declare_subscriber(key, receive):
            print(f"Watching {key}; Ctrl+C to stop.", flush=True)
            while not stop.wait(args.interval):
                robot_state = state["latest"]
                if robot_state is None:
                    print("waiting for Go2 state...", flush=True)
                    continue
                if state["origin"] is None and robot_state.get("position") is not None \
                        and robot_state.get("yaw") is not None:
                    state["origin"] = {"position": robot_state["position"], "yaw": robot_state["yaw"]}
                print(describe(robot_state, state["origin"]), flush=True)
    except (OSError, ValueError, zenoh.ZError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
