"""Save camera frames from Zenoh as JPEG files, e.g. for camera calibration.

Press Enter to save the latest frame (type q then Enter to quit), or use
--every to save periodically. Files are the exact JPEGs the camera node
publishes, so their size is the size VAMOS infers on. Read-only.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path
import sys
import time
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import zenoh

from config import DEFAULT_CONFIG_PATH, load_config
from node import stop_on_signals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Settings file (default zenoh/vamos.json5)")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the saved JPEG files")
    parser.add_argument("--every", type=float, help="Save a frame every N seconds instead of on Enter")
    args = parser.parse_args()
    if args.every is not None and (not math.isfinite(args.every) or args.every <= 0):
        parser.error("--every must be finite and positive")
    try:
        config = load_config(args.config)
        key = config.node_config().keys()["camera"]
        args.out.mkdir(parents=True, exist_ok=True)
        lock, latest = Lock(), {"jpeg": None, "at": 0.0}

        def receive(sample):
            with lock:
                latest["jpeg"], latest["at"] = sample.payload.to_bytes(), time.monotonic()

        def save() -> None:
            with lock:
                jpeg, at = latest["jpeg"], latest["at"]
            if jpeg is None or time.monotonic() - at > 2.0:
                print("no fresh frame; is the camera node publishing?", flush=True)
                return
            with Image.open(BytesIO(jpeg)) as image:
                size = image.size
            path = args.out / f"frame-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.jpg"
            path.write_bytes(jpeg)
            print(f"saved {path} ({size[0]}x{size[1]})", flush=True)

        with stop_on_signals() as stop, zenoh.open(config.zenoh.session("client")) as session, \
                session.declare_subscriber(key, receive):
            print(f"Receiving {key}; " + (f"saving every {args.every:g} s; Ctrl+C to stop." if args.every
                                          else "Enter = save, q + Enter = quit."), flush=True)
            if args.every:
                while not stop.wait(args.every):
                    save()
            else:
                for line in sys.stdin:
                    if line.strip().lower() == "q" or stop.is_set():
                        break
                    save()
    except (OSError, ValueError, zenoh.ZError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
