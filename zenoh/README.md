# VAMOS Zenoh nodes

Run VAMOS image-path inference over Zenoh and execute it on a Unitree Go2 as
short **stop / observe / infer / move** steps. The key/QoS layout and the Go2
safety contract follow the
[NaVILA Zenoh node](https://github.com/Oya-Tomo/NaVILA/tree/feat/2-navila-zenoh-node/zenoh);
the camera and robot bridges are separate projects:

- [camera-zenoh-node](https://github.com/Oya-Tomo/camera-zenoh-node) publishes JPEGs on `camera/front`
- [unitree-go2-zenoh-node](https://github.com/Oya-Tomo/unitree-go2-zenoh-node) (`develop` State contract)
  publishes `unitree/go2/state` and executes `unitree/go2/command`

> [!WARNING]
> `go2_executor.py` moves a real robot. Validate robot-free first, support the
> Go2 for the first powered trials, keep the remote at hand and agree on an
> emergency stop beforehand. Run exactly one executor and never run the
> diagnostic peer's `--fake-go2-state` next to a real robot.

## Components

| Process | Role | Publishes robot commands |
| --- | --- | --- |
| `node.py` | Receives camera/Go2/operator input, runs VAMOS, coordinates stop-go cycles, publishes `vamos/state`, results and previews | no |
| `cli.py` | Operator prompt and heartbeat; optionally launches a local robot-free demo | no |
| `go2_executor.py` | Executes authorized steps on the Go2: stand, bounded velocity, down | **yes** |
| `stream_simulator.py` | Robot-free 2D executor for the same protocol (static image demo) | no |
| `diagnostic_peer.py` | Test JPEG source; heartbeat and fake Go2 state only when asked | no |
| `viewer.py` | Shows `vamos/preview` | no |
| `calibrate_camera.py` | Camera intrinsics, mount and verification for ground projection | no |
| `projection_demo.py`, `simulator.py`, `controller.py` | Offline projection/path-following tools | no |

| Key | Encoding | Publisher → subscribers |
| --- | --- | --- |
| `camera/front` | `image/jpeg` | camera node → node |
| `unitree/go2/state` | `application/json` | Go2 node → node, executor |
| `unitree/go2/command` | `application/json` | **executor** → Go2 node |
| `vamos/command` | `application/json` (`heartbeat`, `start`, `stop`, `goal`) | CLI → node, executor, simulator |
| `vamos/state` | `application/json` | node → CLI, executor, simulator |
| `vamos/trajectories` | `application/json` pixel paths + metadata | node → executor, simulator |
| `vamos/preview` | `image/jpeg` | node → viewer |
| `vamos/executor/state` | `application/json` stop/move feedback | executor or simulator → node |

All keys are concrete (no wildcards) and configurable. Every runtime file named
`zenoh/*.json5` / `zenoh/ground-camera.json` is ignored by Git; copy the tracked
`*.example.*` files to create them.

## Quick start: robot-free demo

From the repository root:

```console
uv sync --group zenoh
uv run --group zenoh python zenoh/cli.py --demo --viewer --simulator
```

This starts the inference node, a test image source (with a fake resting Go2
state), the preview viewer and the 2D stop-go simulator. Enter `status`, then
`start` once `model_loaded=True` and `start_ready=True`. `quit`, EOF or Ctrl+C
stops inference and closes only the processes this CLI started; their logs are
under `outputs/cli` (`logs`, `processes`).

Use `--node-config zenoh/ground-inference-node-config.example.json5 --image
test_images/texas_ground.jpg` for the second bundled image. The demo reuses one
static image at every simulated stop pose: it tests scheduling and short-step
control, not perception, metric accuracy or obstacle clearance.

The inference examples require CUDA and the `mateoguaman/vamos` weights
(`device: cuda` fails explicitly instead of falling back to CPU).

## Deployment on a Go2

Do these in order; each step can be checked without the next one.

1. **Jetson environment** (see the root README): `uv sync --group zenoh`, then
   check `torch.cuda.is_available()` and measure model load time and inference
   latency for the `num_samples` you plan to use. Stop-go tolerates slow
   inference (`inference_timeout_seconds`), but every second is spent standing.
2. **Camera calibration** (below): produce `zenoh/ground-camera.json` with
   `calibration_status: "calibrated"` for the exact image size the camera node
   publishes.
3. **Robot-free run with real inputs**: camera node + Go2 node with the robot
   resting (or the Go2 node alone), VAMOS node, CLI, viewer. Check `status`:
   camera fresh, Go2 fresh, `start_ready` only once an executor reports.
4. **Executor dry run**: start `go2_executor.py` with the Go2 node connected but
   the robot supported or lifted; confirm `stand` on `start`, zero velocity while
   observing/inferring, one bounded step, and `down` on `stop`, heartbeat loss
   (kill the CLI) and executor Ctrl+C.
5. **Powered trials**: low speeds, short steps, open floor.

Processes, each from the repository root (copy the example configs first):

```console
uv run --group zenoh python zenoh/node.py --node-config zenoh/inference-node-config.json5 --zenoh-config zenoh/zenoh-config.json5
uv run --group zenoh python zenoh/go2_executor.py --node-config zenoh/inference-node-config.json5 --executor-config zenoh/go2-executor-config.json5 --zenoh-config zenoh/peer-zenoh-config.json5
uv run --group zenoh python zenoh/cli.py --attach --node-config zenoh/inference-node-config.json5 --zenoh-config zenoh/peer-zenoh-config.json5
uv run --group zenoh python zenoh/viewer.py
```

The executor reads the node config for keys and step bounds, and its own
`go2-executor-config.json5` for calibration, candidate selection, speeds and
timeouts. It refuses a calibration that is not `calibrated` unless
`allow_assumed_calibration: true` (robot-free tests only), and refuses a node
config that is not in stop-go mode. For several computers, edit the
listen/connect endpoints so every process reaches the node's listener.

## Operator CLI

| Command | Operation |
| --- | --- |
| `status` | Model/device, lifecycle, `start_ready`, status message, input freshness, cycle phase, goal |
| `start` | Start stop-go cycles when ready |
| `stop` | Pause, wait for pending inference to be discarded and for stopped feedback |
| `goal 0.5 0.6` | Set the goal as image width/height fractions in [0, 1] (stops first) |
| `ins Follow the open path` | Set the instruction (stops first); bare `ins` clears it |
| `watch 30` | Print status every second for 30 s |
| `processes`, `logs [name]` | Inspect processes started by this CLI |
| `quit` / `exit` | Request and confirm stop, then disconnect |

The CLI sends heartbeat every 0.2 s independently of terminal input and
confirms operations by request IDs echoed in node state. Without `--attach` it
starts a local node; `--demo`, `--viewer` and `--simulator` add robot-free
helpers. It never starts the Go2 executor. A timeout or disconnected node is
reported as unconfirmed, never as a successful stop.

## Stop / observe / infer / move

One `start` runs repeated cycles until stopped:

1. The node opens a cycle (`waiting_stop`). The executor stands a resting Go2
   if needed, holds zero velocity, and reports `stopped` once Go2 observes a
   standing, quiescent state for `stop_settle_seconds`
   (`stop_confirm_timeout_seconds`, default 15 s, includes the stand-up).
2. `waiting_image`: the node uses the first camera frame **received** at least
   `observation_delay_seconds` (default 0.2 s) after that confirmation, so
   camera latency cannot pass off a pre-stop image. Raw camera frames work;
   annotated demo frames must also be captured after the stop.
3. `inferencing`: exactly one inference while the executor keeps reporting the
   same stopped pose. Movement during observation or inference pauses the run.
4. `moving`: the executor selects a candidate, projects it to the ground from
   the stopped pose and follows it until the first of: path end,
   `max_step_seconds`, `max_step_distance_m`, `max_step_yaw_rad`, or
   `result_validity_seconds` after receiving the result.
5. The executor zeroes velocity, settles, and reports `stopped` for that
   inference; the node opens the next cycle.

Any of these pauses the node (an explicit `start` is then required): operator
`stop`, image or heartbeat expiry, executor feedback expiry or replacement,
executor `error`, phase timeouts, inference failure or expired result, and
goal/instruction changes.

`start_ready` and `status_message` in `vamos/state` report whether `start`
would currently be accepted and why not.

### Go2 executor safety contract

- **Stop path** (operator stop, node pause, node state loss, executor error,
  heartbeat loss via the node, shutdown): one zero velocity, then one reliable
  `down` if Go2 state is fresh, then wait for an observed resting state
  (`damping`/`down`, quiescent) up to `go2.down_timeout_seconds`.
- **Startup**: a robot that is not resting when Go2 state first arrives is sent
  `down` once, even without a CLI.
- **Velocity** is published at `control_frequency_hz` (≥ 10 Hz; the Go2 node
  drops velocity after 0.25 s) only while a cycle runs and the robot stands;
  idle publishes nothing. Speed caps: `motion.forward_speed_mps ≤ 1.0`,
  `motion.max_yaw_rate_rps ≤ 1.5` (examples use 0.3 and 0.6).
- **Posture commands are never sent on stale Go2 state.** Stale, disconnected
  or non-accepting Go2 state during a cycle is an error; the stop then ends
  without a rest confirmation and is logged as such.
- **Shutdown** (Ctrl+C/SIGTERM) runs the stop path and exits non-zero if rest
  is not confirmed within `shutdown_timeout_seconds`.
- **Pose** is dead reckoning of the executor's own commands (Go2 publishes no
  pose). It anchors each step at the stopped pose and enforces step bounds;
  bounded steps keep drift small, but it is not odometry.
- **Quiescence** uses the Go2 node's thresholds. If a standing Go2 sways above
  them, the node pauses with "Executor moved during observation"; tune the Go2
  node's `robot_state` thresholds rather than this check.

## Camera calibration

`GroundCamera` (`ground-camera*.json`) holds image size, intrinsics, optional
lens distortion (`distortion_model`: `none`, `opencv` with 4/5/8/12/14
coefficients, or `fisheye` with k1..k4), mounting (height, downward pitch,
left yaw, forward/left offsets of the optical centre from the robot's ground
origin) and `calibration_status`. Projection assumes flat ground and zero roll.
Image sizes must match exactly; intrinsics are never rescaled silently.
`ground-camera.example.json` is **assumed**, not measured.

```console
# 1. Intrinsics/distortion from 15-30 checkerboard photos at the inference resolution
uv run --group zenoh python zenoh/calibrate_camera.py intrinsics --images calib/ --board 9x6 --square-m 0.025 --model fisheye --base zenoh/ground-camera.example.json --output zenoh/ground-camera.json
# 2. Mount from >= 4 non-collinear floor markers (CSV: u,v,x_m,y_m in the robot frame)
uv run --group zenoh python zenoh/calibrate_camera.py mount --camera zenoh/ground-camera.json --markers mount-markers.csv --output zenoh/ground-camera.json
# 3. Verify with different markers; marks the file calibrated only within tolerance
uv run --group zenoh python zenoh/calibrate_camera.py verify --camera zenoh/ground-camera.json --markers check-markers.csv --tolerance-m 0.10 --mark-calibrated zenoh/ground-camera.json
```

Use `fisheye` for wide-angle lenses unless the camera node already publishes
rectified images (then `none`). `mount` warns when the solved roll exceeds 2°,
which the projection model cannot represent. Markers for `verify` should cover
the near and far range the steps use.

## Candidate selection

VAMOS samples `num_samples` paths. The original ROS stack ranks them with an
affordance/value map from a height map, which a camera-only Go2 lacks. Here
`selection.py` projects every candidate and uses:

- `medoid` (default): among usable candidates, the one with the smallest mean
  distance to the others after arc-length resampling — the consensus path;
- `index`: one fixed candidate (`candidate_index`), the earlier behaviour.

Unusable projections (above the horizon, behind the robot, out of range,
oversized segments, shorter than `min_path_length_m`) are excluded and
reported; an unusable selection is never silently replaced. The executor logs
the chosen index, reason and per-candidate report for every step. This is not
obstacle detection.

## Configuration reference

Node (`*inference-node-config.example.json5`):

| Setting | Meaning |
| --- | --- |
| `camera.frame_freshness_seconds`, `cli_heartbeat_timeout_seconds` | Input freshness; expiry pauses the run |
| `go2.node_state_timeout_seconds` | Go2 freshness shown in node state |
| `inference.execution_mode` | `stop_go` for robots; `continuous` is inference-only (see below) |
| `inference.num_samples`, `max_new_tokens`, `temperature`, `disable_compile` | VAMOS generation |
| `inference.goal` | Default goal (image fractions, quantized to VAMOS location tokens) and instruction |
| `inference.stop_go.max_step_seconds` / `max_step_distance_m` / `max_step_yaw_rad` | Step bounds |
| `inference.stop_go.stop_settle_seconds` | Required quiescent standing time before `stopped` |
| `inference.stop_go.stop_confirm_timeout_seconds` | Wait for `stopped`, including stand-up |
| `inference.stop_go.observation_delay_seconds` | Camera latency margin after `stopped` |
| `inference.stop_go.image_wait_timeout_seconds`, `inference_timeout_seconds`, `movement_timeout_seconds`, `executor_timeout_seconds` | Phase and feedback timeouts |

Executor (`go2-executor-config.example.json5`): `camera_calibration`,
`allow_assumed_calibration`, `selection`/`candidate_index`/`min_path_length_m`,
`control_frequency_hz`, `node_state_timeout_seconds`, `result_validity_seconds`,
`shutdown_timeout_seconds`, `go2.{state,stand,down}_timeout_seconds`, and
`motion` (Pure Pursuit speeds, lookahead, tolerances).

## Continuous mode and offline tools

`execution_mode: "continuous"` streams the newest frame through inference at
`frequency_hz` without stopping, with `max_frame_age_seconds` limiting result
age. It is kept for inference/transport experiments and its tests; the Go2
executor refuses it, and with multi-second inference it cannot drive a robot
safely. `stream_simulator.py` can follow its results in the static-image 2D
view.

- `projection_demo.py --infer` saves one real inference with its ground
  projection; `--result ... --image ...` replays it without a GPU.
- `simulator.py` runs the Pure Pursuit controller on synthetic metric paths.
- `node.py` without an `inference` section is a transport diagnostic.

## Tests

```console
uv run --group zenoh python -m unittest discover -s zenoh/tests -v
```

Tests use fake inference engines and a fake Go2 (posture transitions, motion
following velocity); the executor tests drive the real node state machine.
Separate-process localhost tests exercise Zenoh transport. GPU inference,
Jetson performance, the real Go2 node and camera calibration with a physical
camera are not covered and must be checked as described above.

## Linux / WSL2 notes

Use a separate checkout and virtual environment on the Linux filesystem; do not
reuse the Windows `.venv`. Start the node and wait for `Transport ready` before
clients. With Torch 2.13, automatic generation compilation exceeded the
freshness window, so streaming inference disables compilation by default
(`disable_compile`). To reuse Windows-downloaded weights, point `HF_HUB_CACHE`
at that hub directory under `/mnt/c`; `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` then verify cache-only inference.

Validated before the Go2 executor was added: Ubuntu 24.04 under WSL2 x86_64 with
an RTX 5060 (Python 3.11.17, Torch 2.13.0+cu130) ran real 10-path inference,
projection and 2D following, including WSLg windows. Jetson/aarch64 and
physical Go2 operation have not been validated.
