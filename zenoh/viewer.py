"""Display the latest VAMOS trajectory preview published over Zenoh."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from threading import Lock
import time

import zenoh

from config import JPEG_ENCODING, load_node_config
from node import HERE, stop_on_signals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", type=Path, default=HERE / "node-config.example.json5")
    parser.add_argument("--zenoh-config", type=Path, default=HERE / "peer-zenoh-config.example.json5")
    args = parser.parse_args()
    # GUI dependencies are not imported by transport-only nodes/tests.
    import cv2
    import numpy as np

    latest: tuple[bytes, float] | None = None
    lock = Lock()

    def on_preview(sample):
        nonlocal latest
        if str(sample.encoding).split(";", 1)[0] == JPEG_ENCODING:
            with lock:
                latest = sample.payload.to_bytes(), time.monotonic()

    title = "VAMOS paths - Q / Esc to close"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        config = load_node_config(args.node_config)
        with stop_on_signals() as stop, zenoh.open(zenoh.Config.from_file(str(args.zenoh_config))) as session:
            with session.declare_subscriber(config.keys()["preview"], on_preview):
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                logging.info("Waiting for %s", config.keys()["preview"])
                while not stop.is_set():
                    with lock:
                        item = latest
                    frame = None
                    if item is not None and time.monotonic() - item[1] <= 5.0:
                        frame = cv2.imdecode(np.frombuffer(item[0], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Waiting for a fresh VAMOS preview...", (20, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
    except (OSError, ValueError, zenoh.ZError, cv2.error) as error:
        logging.error("%s", error)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
