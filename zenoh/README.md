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

## Metric path follower

`controller.py` provides `PurePursuitController.compute(path, Pose2D(x, y, yaw))`.
Path points and the current pose share a fixed world frame in metres/radians.
It returns a robot-frame velocity (`vx` forward, `vy=0`, `vyaw` counter-clockwise),
tracking status, and the selected lookahead point. Configuration controls lookahead,
forward speed, turn-rate limit, approach slowdown, and endpoint tolerance.

The first version supports simple non-self-intersecting paths, rotates in place
for a target behind or far to the side, and stops on invalid input or endpoint
arrival. Endpoint arrival means the current path segment ended, not that a VAMOS
navigation task is complete. No pixel conversion or obstacle avoidance is included.
It is independent of the inference node and does not publish robot commands.

Print example commands for straight, left, right, behind, and endpoint cases:

```console
uv run --group zenoh python zenoh/controller.py
```

## 2D path-following simulator

Run the existing controller against an ideal planar kinematic model:

```console
uv run python zenoh/simulator.py
```

The window shows the reference path, robot position/heading, travelled trail,
lookahead point, endpoint, forward/turn speed, and tracking state. Select
`straight`, `curve`, `offset`, `behind`, or `overshoot` using the radio buttons
or Left/Right keys. Space pauses/resumes; R resets; Esc closes. The speed slider
changes playback (0.5x to 8x), while integration stays at a fixed 0.05 s step.
Reaching the endpoint stops simulation motion automatically.

Run without a window; optionally save the final plot:

```console
uv run python zenoh/simulator.py --scenario offset --headless
uv run python zenoh/simulator.py --scenario curve --headless --save outputs/simulator-curve.png
```

Headless runs print a JSON summary and exit with code 0 only on `reached`.
`--max-time 120` sets the simulated time budget; exhausting it exits with code 1.
The paths are synthetic metre-based examples. No VAMOS inference, GPU, Zenoh
connection, or robot is used. This checks path-following logic, not Go2 walking
physics, obstacle avoidance, image-to-ground calibration, sensing or latency.

## Flat-ground projection and frozen VAMOS path simulation

Generate 10 real VAMOS candidates, inspect the image/ground comparison, and
simulate candidate 0 in a separate window (Space pauses, R resets):

```console
uv run --group zenoh python zenoh/projection_demo.py --infer --output-dir outputs/projection
```

The inference configuration and image default to the existing inference example
and `test_images/texas_person.jpg`. `--candidate 3` explicitly chooses another
candidate; there is no automatic obstacle/safety ranking. `--headless` runs to
completion without windows. Exit codes: 0 = usable GUI path / headless reached,
1 = unusable selected path or headless time budget exhausted, 2 = input/error.

Replay the saved result without loading a model or using the GPU:

```console
uv run --group zenoh python zenoh/projection_demo.py --result outputs/projection/input_result.json --image outputs/projection/source.png
```

Saved outputs include `source.png`, `input_result.json`, `projection_report.json`,
`projection.png`, and `simulation.png` (initial GUI pose or final headless pose).
An unusable selection is reported and is never replaced by another candidate.

`projection.py` uses the same pinhole ray/plane intersection as the original
`trajectory_projection/Projector.pixels_to_ground`, adding invalid-ray checks.
`ground-camera.example.json` supplies **assumed**, unmeasured values for the
sample image: 1280x720, fx=fy=640 px, camera height 0.5 m, downward pitch 10 deg,
forward offset 0.15 m. These distances are illustrative, not measurements.
Copy to `zenoh/ground-camera.json` and pass `--camera` to change them. Input
images/results must match calibration dimensions exactly; intrinsics are never
silently resized. Replay must use the exact source image for that inference.

The model assumes a rectified pinhole image with zero roll and skew, flat ground
z=0, and a fixed mounting. Pixels use right/down axes; projected robot axes are
forward/left in metres. Above-horizon, near-parallel, behind-robot, out-of-image,
nonfinite, and over-range intersections are rejected. Only the contiguous valid
prefix is retained, ending also at an oversized segment. Invalid gaps are never
bridged; consecutive duplicate points are removed; at least two retained points
are required to simulate. Retained order is the model's original order.

All candidates are shown, with the explicitly selected prefix highlighted.
`path_to_world` anchors that prefix at the image-capture pose (0,0,0 in this
demo), and the simulator updates the robot pose in that fixed frame. This is
one frozen inference, not streaming replanning. Real camera calibration,
timestamped capture/current poses, obstacle checks, candidate selection, and
Go2 command publication remain separate work. The simple controller's existing
non-self-intersecting path assumption still applies.

## Live Zenoh inference -> projection -> 2D following

In three terminals at the repository root, start the inference node, image
fixture, and streaming simulator, respectively:

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/inference-node-config.example.json5
uv run --group zenoh python zenoh/diagnostic_peer.py
uv run --group zenoh python zenoh/stream_simulator.py --output-dir outputs/stream
```

The simulator consumes `vamos/trajectories`, `vamos/state`, and `vamos/command`.
It projects the selected candidate (default 0), replaces the previous path,
and computes velocities locally at 20 Hz. It never publishes robot commands.
Space pauses/resumes local motion; R resets the simulated pose and source;
Esc closes. A fresh result is required after pause/resume, reset or remote
stop/start. Heartbeats alone do not cancel a stop.

The diagnostic peer adds a JSON `CaptureContext` to the JPEG attachment. The
node preserves this on the exact source frame and includes it as optional
`capture` in the result. For this static-image demo the capture pose is always
(0,0,0) in `static_demo_world`, so new results stay anchored to that pose even
after the simulated robot moves. The timestamp denotes fixture publication,
not a newly observed moving-camera image. This demonstrates live transport
and path replacement; the camera view does not respond to simulated motion.

Only annotated `static_image_demo` inputs can drive this simulator. Existing
camera messages without metadata still work for inference/preview, but the
simulator stops with `missing_capture_pose`. The source ID is pinned until R
or process restart, preventing a different source from silently replacing the
world. The single-PC demo uses Unix timestamps plus local monotonic expiry;
cross-machine capture poses and clock synchronization are not implemented.

Motion stops when the inference node/state/heartbeat becomes unavailable,
when a path expires (default source age 3 s, including inference time), on
stop/pause, or on invalid projection. Duplicate/out-of-order results cannot
refresh expiry. `--headless --duration 30` runs without a window for 30 wall
seconds; `--path-timeout` and `--state-timeout` adjust expiry. The optional
output directory receives `stream_report.json` and `stream.png` on exit.

For the other bundled image, stop the previous node/peer and instead use:

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/ground-inference-node-config.example.json5
uv run --group zenoh python zenoh/diagnostic_peer.py --image test_images/texas_ground.jpg
uv run --group zenoh python zenoh/stream_simulator.py --output-dir outputs/stream
```

`--image` overrides the peer's image path relative to the working directory.
The ground-image inference example uses a goal at (0.5,0.6) and a paved-path
instruction. For later images, configure the image-specific goal/instruction
and camera intrinsics/mounting; image dimensions must match calibration.
The assumed example calibration only illustrates projection. This does not
verify obstacle clearance, metric accuracy, moving-camera navigation or Go2
walking. The optional preview viewer can run as a fourth process.

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
