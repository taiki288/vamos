"""Publish one test image as a camera stream and print VAMOS node status.

Robot-free testing only. Heartbeat is opt-in because a second heartbeat
source would hide the loss of the operator CLI.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from io import BytesIO
import logging
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import zenoh

from config import DEFAULT_CONFIG_PATH, JPEG_ENCODING, JSON_ENCODING, NodeState, load_config
from node import HERE, has_encoding, stop_on_signals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Settings file (default zenoh/vamos.json5)")
    parser.add_argument("--image", type=Path, default=HERE.parent / "test_images/texas_person.jpg")
    parser.add_argument("--frequency-hz", type=float, default=5.0)
    parser.add_argument("--heartbeat", action="store_true", help="Also send operator heartbeat (no CLI running)")
    args = parser.parse_args()
    if not math.isfinite(args.frequency_hz) or args.frequency_hz <= 0:
        parser.error("--frequency-hz must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_config(args.config)
        keys = config.node_config().keys()
        with Image.open(args.image) as source:
            buffer = BytesIO()
            source.convert("RGB").save(buffer, format="JPEG", quality=90)
        jpeg = buffer.getvalue()
        latest: list[NodeState | float] = []

        def on_state(sample):
            if has_encoding(sample, JSON_ENCODING):
                try:
                    latest[:] = [NodeState.model_validate_json(sample.payload.to_bytes()), time.monotonic()]
                except ValueError as error:
                    logging.warning("Invalid VAMOS state: %s", error)

        with stop_on_signals() as stop, zenoh.open(config.zenoh.session("client")) as session, \
                ExitStack() as resources:
            resources.enter_context(session.declare_subscriber(keys["state"], on_state))
            camera = resources.enter_context(session.declare_publisher(keys["camera"], encoding=JPEG_ENCODING))
            control = resources.enter_context(session.declare_publisher(keys["command"], encoding=JSON_ENCODING))
            logging.info("Publishing %s on %s%s. Ctrl+C to stop.", args.image.name, keys["camera"],
                         " + heartbeat" if args.heartbeat else "")
            period, next_report = 1.0 / args.frequency_hz, 0.0
            while not stop.is_set():
                camera.put(jpeg)
                if args.heartbeat:
                    control.put('{"type":"heartbeat"}')
                now = time.monotonic()
                if now >= next_report:
                    if not latest or now - latest[1] > 2.0:
                        logging.info("Waiting for fresh VAMOS state...")
                    else:
                        state = latest[0]
                        logging.info("frames=%d camera_fresh=%s heartbeat=%s %s", state.camera.received_frames,
                                     state.camera.fresh, state.cli_connected, state.status_message)
                    next_report = now + 1.0
                stop.wait(period)
    except (OSError, ValueError, zenoh.ZError) as error:
        logging.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
