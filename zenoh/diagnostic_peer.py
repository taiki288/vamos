"""Publish a test image, simulated Go2 state, and heartbeat; display VAMOS state."""

from __future__ import annotations

import argparse
import logging
import math
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from threading import Lock

import json5
import zenoh
from PIL import Image
from pydantic import StrictStr, field_validator

from config import JPEG_ENCODING, JSON_ENCODING, NodeState, PositiveSeconds, Settings, load_node_config
from node import HERE, stop_on_signals


class PeerConfig(Settings):
    image_path: StrictStr
    publish_frequency_hz: PositiveSeconds

    @field_validator("publish_frequency_hz")
    @classmethod
    def heartbeat_period(cls, value: float) -> float:
        if not math.isfinite(1.0 / value):
            raise ValueError("publish period must be finite")
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, default=HERE / "node-config.example.json5")
    parser.add_argument("--peer-config", type=Path, default=HERE / "diagnostic-peer-config.example.json5")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "peer-zenoh-config.example.json5")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_node_config(args.node_config)
        peer = PeerConfig.model_validate(json5.loads(args.peer_config.read_text(encoding="utf-8"), allow_duplicate_keys=False))
        image_path = (args.peer_config.resolve().parent / peer.image_path).resolve()
        with Image.open(image_path) as source:
            buffer = BytesIO()
            source.convert("RGB").save(buffer, format="JPEG", quality=90)
            jpeg = buffer.getvalue()
        keys = config.keys()
        latest: NodeState | None = None
        latest_at: float | None = None
        lock = Lock()

        def on_state(sample):
            nonlocal latest, latest_at
            try:
                if str(sample.encoding).split(";", 1)[0] != JSON_ENCODING:
                    raise ValueError("expected application/json")
                state = NodeState.model_validate_json(sample.payload.to_bytes())
                with lock:
                    latest, latest_at = state, time.monotonic()
            except ValueError as error:
                logging.warning("Invalid VAMOS state: %s", error)

        # This fixture describes an idle robot; it never publishes Go2 commands.
        fake_go2 = '{"robot_connected":true,"robot_state":{"state":"down","motion":"quiescent"},"accepting_commands":true}'
        with stop_on_signals() as stop, zenoh.open(zenoh.Config.from_file(str(args.zenoh_config))) as session, ExitStack() as resources:
            resources.enter_context(session.declare_subscriber(keys["state"], on_state))
            camera = resources.enter_context(session.declare_publisher(keys["camera"], encoding=JPEG_ENCODING))
            go2 = resources.enter_context(session.declare_publisher(keys["go2_state"], encoding=JSON_ENCODING))
            control = resources.enter_context(session.declare_publisher(keys["command"], encoding=JSON_ENCODING))
            logging.info("Sending %s + SIMULATED Go2 state + heartbeat. Ctrl+C to stop.", image_path.name)
            next_report = 0.0
            period = 1.0 / peer.publish_frequency_hz
            deadline = time.monotonic()
            while not stop.is_set():
                camera.put(jpeg)
                go2.put(fake_go2)
                control.put('{"type":"heartbeat"}')
                now = time.monotonic()
                if now >= next_report:
                    with lock:
                        state, received_at = latest, latest_at
                    if state is None or now - received_at > 2.0:
                        logging.info("Waiting for fresh VAMOS state...")
                    else:
                        logging.info("frames=%d %sx%s camera_fresh=%s go2_fresh=%s heartbeat=%s",
                                     state.camera.received_frames, state.camera.width, state.camera.height,
                                     state.camera.fresh, state.go2.fresh, state.cli_connected)
                    next_report = now + 1.0
                deadline += (max(0, math.floor((now - deadline) / period)) + 1) * period
                stop.wait(max(0.0, deadline - time.monotonic()))
    except (OSError, ValueError, zenoh.ZError) as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
