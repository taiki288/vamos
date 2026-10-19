# VAMOS Zenoh transport

The default configuration receives JPEG images and Go2 state, accepts heartbeat/stop
messages, and publishes `vamos/state`. An optional inference configuration connects
the received image to VAMOS and publishes image-space paths and JPEG previews.
Neither mode publishes velocity/posture commands to Go2.

The connection/key/QoS layout follows the
[NaVILA Zenoh node](https://github.com/Oya-Tomo/NaVILA/tree/feat/2-navila-zenoh-node/zenoh).
Settings use Pydantic v2, matching this VAMOS fork.

From the repository root, run the node:

```console
uv run --group zenoh python zenoh/node.py
```

In a second terminal, send a sample image, simulated Go2 state and heartbeat:

```console
uv run --group zenoh python zenoh/diagnostic_peer.py
```

The peer prints received dimensions, image/state freshness, and heartbeat status.
Ctrl+C stops each process. The examples use direct TCP on localhost port 7447;
multicast discovery is disabled. The diagnostic peer is a test input source,
not a real robot node. Do not run it on the same keys as a real input source.

| Key | Direction from VAMOS | Encoding |
| --- | --- | --- |
| `camera/front` | receive | `image/jpeg` binary JPEG |
| `unitree/go2/state` | receive | `application/json` |
| `vamos/command` | receive | `application/json`, `heartbeat` or `stop` |
| `vamos/state` | publish | `application/json` |
| `vamos/trajectories` | publish, inference mode | `application/json`, pixel paths and source frame metadata |
| `vamos/preview` | publish, inference mode | `image/jpeg`, paths overlaid on the source image |

`start_ready` stays false: robot navigation is not implemented. In transport mode,
`stop` is acknowledged in diagnostics only. Stale Go2 data is not exposed
as a current observation. Invalid samples do not refresh valid data timestamps.
Only the latest decoded RGB image is retained; no image files are written.

## Received-image inference

Run this instead of the transport-only node, using the same diagnostic peer:

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/inference-node-config.example.json5
```

In a third terminal, optionally show the result:

```console
uv run --group zenoh python zenoh/viewer.py
```

The inference example requires CUDA, uses `mateoguaman/vamos`, and generates
10 candidate paths. `goal.x` and `goal.y` are image-width/height fractions in
`[0, 1]`, quantized to VAMOS's location tokens. The example matches the goal and
instruction from `vamos_demo.py`. Edit a local copy of the configuration to change
the model, device (`cuda`, `cpu`, `auto`), goal, candidate count, or inference rate.
`device: cuda` fails explicitly instead of silently falling back to CPU.

State publication continues while the model loads or inference is running.
Once loaded, inference starts automatically on a fresh image and heartbeat; Go2
state is monitored but not required for this image-only stage. One worker processes
only the newest unprocessed frame, skips missed schedule slots, and never overlaps
inferences. Stop (`{"type":"stop"}` on `vamos/command`) pauses inference and
invalidates pending results; start resumes it after the model is loaded. Heartbeats
alone do not undo a stop. Model/generation failures are reported in
`state.inference.last_error`; generation errors pause until start or restart.

Results are discarded if stopped, shutting down, if camera/heartbeat information is
stale, or if the source frame exceeds `max_frame_age_seconds`. The JSON includes
the source frame sequence, original image dimensions, pixel coordinates, goal,
prompt, inference duration, source age, device and inference ID. The preview uses
the exact source image, with its inference ID in the Zenoh sample attachment.
These are candidate image paths, not metric trajectories or Go2 movement commands.
The viewer clears previews after five seconds without a new result.

Automatic tests use fake engines; real model/GPU checks are separate and require
the downloaded model and a compatible GPU environment. The GUI viewer is a manual
check; the automated tests verify preview JPEG content without opening a window.

The tracked `*.example.json5` files are runnable defaults. For a different setup,
copy them to the corresponding `*.json5` runtime paths and pass `--node-config`,
`--zenoh-config`, or `--peer-config` explicitly. Runtime files are ignored by Git.
For multiple computers, edit the listener/connect endpoints and keep keys aligned
with the existing camera/Go2 nodes. Windows and Linux use the same entry points.

Run unit and real localhost, separate-process transport tests:

```console
uv run --group zenoh python -m unittest discover -s zenoh/tests -v
```
