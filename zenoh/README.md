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
| `vamos/command` | receive | `application/json`, `heartbeat`, `start`, `stop`, or `goal` |
| `vamos/state` | publish | `application/json` |
| `vamos/trajectories` | publish, inference mode | `application/json`, pixel paths and source frame metadata |
| `vamos/preview` | publish, inference mode | `image/jpeg`, paths overlaid on the source image |
| `vamos/executor/state` | receive, stop-go mode | `application/json`, ideal 2D stop/movement feedback |

`start_ready` stays false: robot navigation is not implemented. In transport mode,
`stop` is acknowledged in diagnostics only. Stale Go2 data is not exposed
as a current observation. Invalid samples do not refresh valid data timestamps.
Only the latest decoded RGB image is retained; no image files are written.

## Received-image inference

For a local demo, one command starts the inference node, sample image publisher
and image-space viewer. Inference is initially paused:

```console
uv run --group zenoh python zenoh/cli.py --demo --viewer
```

Enter `status`, then `start` after `model_loaded=True` and `camera_fresh=True`.
The inference examples now use stop/observe/infer/move cycles. A headless 2D
executor is automatically included with `--demo`; `--simulator` opens its window
instead. `quit`, EOF
or Ctrl+C at the prompt stops inference and closes the processes started by this
CLI. Child output goes to per-session logs under `outputs/cli`; `logs` and
`processes` inspect these from the same prompt. Closing a viewer alone leaves
inference running. No extra console windows are opened.

To run the components separately, start the inference node:

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/inference-node-config.example.json5
```

Run the image fixture without heartbeat, then the operator CLI in a third terminal:

```console
uv run --group zenoh python zenoh/diagnostic_peer.py --no-heartbeat
uv run --group zenoh python zenoh/cli.py --attach --simulator
```

Enter `start` in the CLI after the model loads and the image is fresh.
In a fourth terminal, optionally show the result:

```console
uv run --group zenoh python zenoh/viewer.py
```

The inference example requires CUDA, uses `mateoguaman/vamos`, and generates
10 candidate paths. `goal.x` and `goal.y` are image-width/height fractions in
`[0, 1]`, quantized to VAMOS's location tokens. The example matches the goal and
instruction from `vamos_demo.py`. Edit a local copy of the configuration to change
the model, device (`cuda`, `cpu`, `auto`), goal, candidate count, or inference rate.
`device: cuda` fails explicitly instead of silently falling back to CPU.
`inference.disable_compile` defaults to true for predictable streaming latency;
it is passed explicitly to generation. Set it to false only when evaluating
compiled inference, including its initialization and repeated-call overhead.

State publication continues while the model loads or inference is running.
Inference starts on an explicit CLI `start` with a fresh image and heartbeat; Go2
state is monitored but not required for this simulated stage. Stop-go mode also
requires fresh stopped-executor feedback. `auto_start` defaults to false and is
only supported in continuous mode. One worker processes
only the newest unprocessed frame, skips missed schedule slots, and never overlaps
inferences. Stop (`{"type":"stop"}` on `vamos/command`) pauses inference and
invalidates pending results; start resumes it after the model is loaded. Heartbeats
alone do not undo a stop. Image/heartbeat expiry also latches a pause and requires
an explicit start after recovery. Model/generation failures are reported in
`state.inference.last_error`; generation errors pause until start or restart.

Results are discarded if stopped, shutting down, or if camera/heartbeat information
is stale. Continuous mode enforces `max_frame_age_seconds`. The stopped static-image
demo instead uses its separate bounded inference budget, described below. The JSON includes
the source frame sequence, original image dimensions, pixel coordinates, goal,
prompt, inference duration, source age, device and inference ID. The preview uses
the exact source image, with its inference ID in the Zenoh sample attachment.
These are candidate image paths, not metric trajectories or Go2 movement commands.
The viewer clears previews after five seconds without a new result.

## Stop / observe / infer / move cycles

Both inference examples set `inference.execution_mode: "stop_go"`. One operator
`start` runs repeated cycles, without requiring a start for every step:

1. The node requests a stop with a new cycle ID.
2. The local 2D executor holds zero velocity for the settling interval and echoes
   stopped feedback for that node session, control epoch and cycle ID.
3. The node discards pre-stop frames and waits for a later annotated image whose
   publication timestamp is after stop confirmation.
4. Exactly one inference runs while the executor remains stopped.
5. The executor accepts only the matching authorized result and follows a short
   prefix. It then sets zero velocity, settles, and confirms step completion.
6. The node waits for another stop confirmation and a new image before repeating.

Default bounds in `inference.stop_go` are `max_step_seconds: 1.0`,
`max_step_distance_m: 0.2`, `max_step_yaw_rad: 0.35` (about 20 degrees), and
`stop_settle_seconds: 0.2`. Execution ends at the first bound, path endpoint, or
the last integration tick before its execution validity budget runs out.
Delayed result delivery therefore shortens the move. A result already expired
on arrival, or a timing stall past expiry, still aborts the cycle.
Distance/angle limits have up to one integration tick of discretization error.
`executor_timeout_seconds: 1.0`, `image_wait_timeout_seconds: 5.0`, and
`movement_timeout_seconds: 4.0` bound missing feedback, image and movement waits.
Stopped inference has its own `inference_timeout_seconds: 60.0` budget. The
executor must continuously report the same stopped pose throughout observation
and inference. This static-image simulator uses the result's explicit
`generated_at_unix_seconds` for its short execution validity window (default
3 seconds), so an 11-second inference can be followed by a 1-second move. The
original capture timestamp and `source_age_seconds` are preserved; the image is
never relabelled as newly captured. Missing generation timestamps, excessive
inference age and delayed/stale results are rejected. Continuous mode retains
its original image-age limits. This stopped-pose policy applies only to the
static-image demo; a real camera scene can change while the robot is stationary
and needs its own observation-age policy before hardware use.

`status`/`watch` show `cycle`, `phase`, `completed_steps`, executor freshness and
motion. The 2D window shows phase and velocities, with zero motion during image
waiting/inference. `stop` waits for both pending inference and stopped feedback.
Goal/instruction changes stop the cycle first and require another `start`.
Loss of camera/heartbeat/executor, phase timeout, unusable paths, unexpected
executor replacement or motion during inference latches a pause. Recovery alone
does not resume the cycle. Local Space/R also interrupts the current cycle.

This executor publishes only `vamos/executor/state`, tagged
`kind: stop_go_simulator`. It never publishes Go2 velocity or posture commands.
An actual robot backend and measured stopped-motion confirmation are not
implemented. In particular, an idle Go2 fixture state cannot advance a cycle.
Without `--demo`, start a simulator separately (or use `--simulator`) to exercise
this protocol. Raw, unannotated camera images cannot drive these demo cycles.

The image pixels remain unchanged in `--demo`. At each stopped pose, the path
is projected relative to the **simulated** current pose, reusing the image as if
observed there. This tests scheduling and short-step control, not a changing
camera view, real-world target tracking, metric accuracy or obstacle clearance.
The original fixture capture pose is preserved in inference results. The step
events explicitly tag their synthetic anchor as `simulated_stop_pose`.

For legacy transport/inference experiments, `execution_mode: "continuous"`
retains latest-frame streaming and the fixed-capture-pose follower. It is not
the default in the runnable inference examples.

## Operator CLI

```console
uv run --group zenoh python zenoh/cli.py
```

By default, the CLI starts a local inference node with the selected inference
settings, then connects to it. It waits for images from an external camera node.
Use `--demo` to also publish a test image and simulated Go2 state; this option is
for local tests only. In deployment, a camera node for the robot's camera must
publish JPEGs on the configured camera key (`camera/front` by default), and its
Zenoh connection must reach the inference node. The CLI does not yet acquire
Go2 camera images itself. The default connection is localhost-only; remote
deployment needs matching network endpoints on both machines.

Use `--viewer` and/or `--simulator` to open the existing result views. Stop-go
demo mode includes a headless executor even without a simulator window. Use
`--attach` for an already running node; its process is not owned by the CLI.
An occupied listener causes startup to fail with an instruction to use
`--attach`; existing processes are never adopted or terminated automatically.
The CLI sends heartbeat independently of terminal input and confirms operations
using request IDs echoed in periodic node state.

| Command | Operation |
| --- | --- |
| `status` | Show model/device, paused/active state, image/heartbeat freshness, counts, current goal and instruction |
| `start` | Start/resume inference when ready |
| `stop` | Pause and wait for any pending inference to finish and be discarded |
| `goal 0.5 0.6` | Set image-width/height fractions in [0,1], preserving the instruction |
| `ins Follow the open path` | Set the instruction, preserving the goal; bare `ins` clears it |
| `watch 30` | Print status every second for 30 seconds; Ctrl+C returns to the prompt |
| `quit` / `exit` | Request and confirm stop before disconnecting |
| `help` | Show commands |
| `processes` | Show owned processes, exit codes and log paths |
| `logs` / `logs source` | Show the last 20 lines of a managed process's log |

Goal/instruction changes automatically stop and drain pending inference first,
then update the goal atomically and remain paused. Use `start` to resume. Model
weights stay loaded. After input loss, `start` waits for any remaining inference
to finish before resuming. EOF or Ctrl+C at the prompt also requests stop. A timeout
or disconnected node is reported as unconfirmed, never as a successful stop.
The CLI controls inference and the local simulation only; it publishes no Go2
velocity or posture commands. On exit it requests cooperative shutdown of only
its own children, with bounded process-tree cleanup if a child does not respond.
In `--attach` mode it pauses inference but leaves the external node running.

`--node-config` selects the existing node settings for keys and heartbeat limits;
`--zenoh-config` selects the connection. No extra CLI JSON is needed. Defaults are
`inference-node-config.example.json5` and `peer-zenoh-config.example.json5`.
Optional `--heartbeat-interval`, `--state-timeout` and `--command-timeout` default
to 0.2, 1 and 15 seconds. The stop timeout includes waiting for slow inference.
`--node-zenoh-config` selects the managed node's listener (default
`zenoh-config.example.json5`); `--zenoh-config` is the client connection shared
by the CLI and optional local views/source. `--peer-config` and `--image` select
demo input (`--image` requires `--demo`). `--camera` selects the simulator's
calibration, and `--log-dir` sets the parent folder for session logs. The model
loads in the background; check `status` before starting inference.
Keep exactly one operator heartbeat source when checking CLI-loss stopping:
use `diagnostic_peer.py --no-heartbeat`, otherwise its legacy fixture heartbeat
can keep the connection alive after the CLI exits unexpectedly.

The control protocol adds optional correlation/session/revision fields. The
node echoes `last_control_ack`, current `goal`, `goal_revision`, `control_epoch`
and `node_session_id` in state. Result metadata carries the request's immutable
goal revision, epoch and node session. Identical retried request IDs are not
executed twice; stale session/revision writes are rejected. The stream follower
rejects results from an obsolete control epoch/session, including late network
delivery across stop/start or goal changes. Goal updates require inference to
be disabled with no active request, which the CLI coordinates automatically.

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

One command starts the node, test source, CLI and streaming simulator:

```console
uv run --group zenoh python zenoh/cli.py --demo --viewer --simulator
```

Alternatively, in four terminals at the repository root:

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/inference-node-config.example.json5
uv run --group zenoh python zenoh/diagnostic_peer.py --no-heartbeat
uv run --group zenoh python zenoh/cli.py --attach
uv run --group zenoh python zenoh/stream_simulator.py --output-dir outputs/stream
```

Enter `start` in the CLI once the model is ready. A goal/instruction change stops
following until the operator explicitly starts the new goal.

The simulator consumes `vamos/trajectories`, `vamos/state`, and `vamos/command`.
It projects the selected candidate (default 0) and computes local velocities at
20 Hz. In stop-go mode each result drives one bounded step, followed by stopped
feedback; no result replaces a path during motion. In continuous mode a fresh
result replaces the previous path. It never publishes robot commands.
Space pauses/resumes local motion; R resets the simulated pose and source;
Esc closes. A fresh result is required after pause/resume, reset or remote
stop/start. Heartbeats alone do not cancel a stop.

The diagnostic peer adds a JSON `CaptureContext` to the JPEG attachment. The
node preserves this on the exact source frame and includes it as optional
`capture` in the result. The fixture's capture pose is always (0,0,0) in
`static_demo_world`. Continuous mode anchors new results there. Stop-go mode
explicitly reuses the static image at each simulated stopped pose instead, as
described above. The timestamp denotes fixture publication,
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
uv run --group zenoh python zenoh/cli.py --demo --viewer --simulator --node-config zenoh/ground-inference-node-config.example.json5 --image test_images/texas_ground.jpg
```

Or start its components separately:

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/ground-inference-node-config.example.json5
uv run --group zenoh python zenoh/diagnostic_peer.py --image test_images/texas_ground.jpg --no-heartbeat
uv run --group zenoh python zenoh/cli.py --attach --node-config zenoh/ground-inference-node-config.example.json5
uv run --group zenoh python zenoh/stream_simulator.py --output-dir outputs/stream
```

`--image` overrides the peer's image path relative to the working directory.
The ground-image inference example uses a goal at (0.5,0.6) and a paved-path
instruction. For later images, configure the image-specific goal/instruction
and camera intrinsics/mounting; image dimensions must match calibration.
The assumed example calibration only illustrates projection. This does not
verify obstacle clearance, metric accuracy, moving-camera navigation or Go2
walking. The optional preview viewer can run as a fifth process.

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

## Linux / WSL2 validation

Use a separate checkout and virtual environment inside the Linux filesystem;
do not reuse the Windows `.venv`. From the Linux checkout, with Linux uv on PATH:

```console
uv sync --locked --group zenoh --python 3.11
uv run --frozen --group zenoh python -m unittest discover -s zenoh/tests -v
uv run --frozen python zenoh/simulator.py --headless
uv run --frozen --group zenoh python zenoh/projection_demo.py --infer --headless --output-dir outputs/projection
```

The existing node/peer/viewer/simulator commands also apply on Linux. Start the
node first and wait for `Transport ready` before launching clients. Model load
continues in the background. The TCP tests wait for the listener explicitly;
starting a thread/process alone does not guarantee it is accepting connections.

The lock file selects platform-specific Torch builds: the Windows AMD64 CUDA
12.8 source override does not apply to Linux. Check `torch.cuda.is_available()`
in the Linux environment before GPU checks. With Torch 2.13, automatic generation
compilation exceeded the freshness window in local checks. Streaming adapters
therefore disable compilation by default on every platform, preserving the
expiry limits rather than treating old results as usable. The standalone demo
retains its previous default unless a caller explicitly overrides compilation.

If reusing downloaded weights from Windows, set `HF_HUB_CACHE` to that hub
directory as seen under `/mnt/c`. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` can then verify cache-only inference. Otherwise the
normal Hugging Face model download applies. This does not require sharing the
Python environment between Windows and Linux.

Validated on Ubuntu 24.04 under WSL2 x86_64 with RTX 5060: Python 3.11.17,
Torch 2.13.0+cu130, real 10-path
inference/projection/following, and WSLg Matplotlib/TkAgg and OpenCV window
creation. Real three-process checks received four 10-path updates for each of
the two bundled images and stopped simulated motion after the image fixture
exited. This is x86_64 Linux validation, not Jetson/aarch64 or physical Go2
validation.
