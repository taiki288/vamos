# VAMOS Zenoh nodes

Run VAMOS image-path inference over Zenoh and execute it on a Unitree Go2 as
short **stop / observe / infer / move** steps toward a goal that is given once,
held fixed on the ground, and ends the run when reached. The key/QoS layout and the Go2
safety contract follow the
[NaVILA Zenoh node](https://github.com/Oya-Tomo/NaVILA/tree/feat/2-navila-zenoh-node/zenoh);
the camera and robot bridges are separate projects:

- [camera-zenoh-node](https://github.com/Oya-Tomo/camera-zenoh-node) publishes JPEGs on `camera/front`
- [unitree-go2-zenoh-node](https://github.com/Oya-Tomo/unitree-go2-zenoh-node) (`develop` State contract)
  publishes `unitree/go2/state` and executes `unitree/go2/command`

> [!WARNING]
> `go2_executor.py` moves a real robot. Validate robot-free first, support the
> Go2 for the first powered trials, keep the remote at hand and agree on an
> emergency stop beforehand. Run exactly one executor, and never run
> `tools/fake_go2.py` or `tools/demo.json5` with a real robot.

## Components

| Process | Role | Publishes robot commands |
| --- | --- | --- |
| `node.py` | Receives camera/Go2/operator input, runs VAMOS, coordinates stop-go cycles, publishes `vamos/state`, results and previews | no |
| `cli.py` | Operator prompt and heartbeat; optionally launches a local robot-free demo | no |
| `go2_executor.py` | Executes authorized steps on the Go2: stand, bounded velocity, down | **yes** |

Libraries: `config.py` (settings and wire models), `inference.py` (VAMOS
adapter), `projection.py` (pixels → ground), `selection.py` (candidate choice),
`controller.py` (Pure Pursuit), `launcher.py` (CLI-managed processes).

Development and setup tools live in `tools/`:

| Tool | Role |
| --- | --- |
| `tools/fake_camera.py` | Publishes one test image as the camera stream |
| `tools/fake_go2.py` | Simulated Go2 node (same Zenoh contract, plus the proposed position/yaw; `--no-pose` omits them) with an optional 2D view |
| `tools/viewer.py` | Shows `vamos/preview` |
| `tools/snapshot.py` | Saves camera frames as JPEG (on Enter or `--every`), e.g. for calibration |
| `tools/watch_go2.py` | Prints Go2 state and odometry change since start (read-only), for hardware checks |
| `tools/calibrate_camera.py` | Camera intrinsics, mount and verification for ground projection |
| `tools/projection_demo.py` | One saved or live inference projected to the ground, with the executor's selection |

| Key | Encoding | Publisher → subscribers |
| --- | --- | --- |
| `camera/front` | `image/jpeg` | camera node → node |
| `unitree/go2/state` | `application/json` | Go2 node → node, executor |
| `unitree/go2/command` | `application/json` | **executor** → Go2 node |
| `vamos/command` | `application/json` (`heartbeat`, `start`, `stop`, `goal`) | CLI → node, executor |
| `vamos/state` | `application/json` | node → CLI, executor |
| `vamos/trajectories` | `application/json` pixel paths + metadata | node → executor |
| `vamos/preview` | `image/jpeg` | node → viewer |
| `vamos/executor/state` | `application/json` stop/move feedback and current step path | executor → node |

All keys are concrete (no wildcards) and configurable.

## Settings

Every process reads **one** JSON5 file and uses its own sections:

| File | Use |
| --- | --- |
| `zenoh/vamos.json5` | Your robot settings, copied from the tracked `zenoh/vamos.example.json5` (ignored by Git). The default for every process. |
| `zenoh/tools/demo.json5` | Robot-free demo; `cli.py --demo` uses it automatically. |
| `zenoh/ground-camera.json` | Camera calibration, written by `tools/calibrate_camera.py` and named by `executor.camera_calibration` (ignored by Git). |

```console
cp zenoh/vamos.example.json5 zenoh/vamos.json5
```

Sections: `keys` (Zenoh keys), `zenoh` (session configs passed to Zenoh as-is:
`node` listens, `client` is used by every other process), `node`, `executor`
and `cli`. Pass another file with `--config` to any process. Unknown fields,
wrong types, key collisions and invalid Zenoh configs are rejected at startup;
a missing `zenoh/vamos.json5` is reported with the copy command. Copy the same
file to every machine: point `zenoh.client` (and the node's `listen`) at the
node host's LAN address.

## Quick start: robot-free demo

From the repository root:

```console
uv sync --group zenoh
uv run --group zenoh python zenoh/cli.py --demo --viewer --simulator
```

This starts the inference node, a test camera, a simulated Go2 (window with
`--simulator`), the **real** `go2_executor.py` with
the demo settings, and the preview viewer. Enter `status`, then
`start` once `model_loaded=True` and `start_ready=True`: the executor stands the
simulated Go2, anchors the example goal on the ground, runs one bounded step per
inference until the goal is reached, and lays the robot down. Try `goal_m 2 1`
or `goal_m -1 0.5` (behind: the robot turns first) and `start` again.
`quit`, EOF or Ctrl+C stops inference and closes only the processes this CLI
started (the executor first, so it can rest the robot); their logs are under
`outputs/cli` (`logs`, `processes`).

For the second bundled image, add `--image test_images/texas_ground.jpg` and set a
goal on its path at the prompt (for example `goal 0.5 0.6`). The camera image
never changes as the simulated robot moves: the demo exercises the protocol,
selection and executor, not perception, metric accuracy or obstacle clearance.

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
4. **Odometry check** (with `executor.pose_source: "go2"`): turn the robot about
   90° by hand or with the keyboard example, walk it forward, and confirm that
   `robot_state.position` moves along the reported `yaw`. Goals are anchored in
   that frame, so a mismatch misplaces every goal.
5. **Executor dry run**: start `go2_executor.py` with the Go2 node connected but
   the robot supported or lifted; confirm `stand` on `start`, zero velocity while
   observing/inferring, one bounded step, and `down` on `stop`, heartbeat loss
   (kill the CLI) and executor Ctrl+C.
6. **Powered trials**: low speeds, short steps, open floor.

Processes, each from the repository root, all reading `zenoh/vamos.json5`:

```console
uv run --group zenoh python zenoh/node.py
uv run --group zenoh python zenoh/go2_executor.py
uv run --group zenoh python zenoh/cli.py --attach
uv run --group zenoh python zenoh/tools/viewer.py
```

The executor refuses a calibration that is not `calibrated` unless
`executor.allow_assumed_calibration` is true (robot-free tests only). Use
`executor.pose_source: "go2"` (the robot's own odometry, as the original VAMOS
uses Spot's) with a Go2 node that publishes `robot_state.position`/`yaw` from
`SportModeState.position` and `imu_state.rpy[2]`; until then
`"dead_reckoning"` drifts with distance, so keep goals short.

## Operator CLI

| Command | Operation |
| --- | --- |
| `status` | Model/device, lifecycle, `start_ready`, status message, input freshness, cycle phase, goal |
| `start` | Start stop-go cycles when ready |
| `stop` | Pause, wait for pending inference to be discarded and for stopped feedback |
| `goal 0.5 0.6` | Set the goal as an image point (width/height fractions in [0, 1]) on the first standing view (stops first) |
| `goal_m 3 -0.5` | Set the goal as metres forward/left of the robot when it is anchored (stops first) |
| `ins Follow the open path` | Set the instruction (stops first); bare `ins` clears it |
| `watch 30` | Print status every second for 30 s |
| `processes`, `logs [name]` | Inspect processes started by this CLI |
| `quit` / `exit` | Request and confirm stop, then disconnect |

The CLI sends heartbeat every 0.2 s independently of terminal input and
confirms operations by request IDs echoed in node state. Without `--attach` it
starts a local node; `--demo` (optionally with `--simulator`) and `--viewer`
add robot-free helpers. Outside `--demo` it never starts the Go2 executor. A timeout or disconnected node is
reported as unconfirmed, never as a successful stop.

## Stop / observe / infer / move

One `start` runs repeated cycles until stopped:

1. The node opens a cycle (`waiting_stop`). The executor stands a resting Go2
   if needed and holds zero velocity until Go2 observes a standing, quiescent
   state for `stop_settle_seconds`. It then locates the goal (see Goals): it
   reports `stopped` with the goal's current image point, turns toward a goal
   out of view first, or reports the goal reached
   (`stop_confirm_timeout_seconds`, default 15 s, covers stand-up and turning).
2. `waiting_image`: the node uses the first camera frame **received** at least
   `observation_delay_seconds` (default 0.2 s) after that confirmation, so
   camera latency cannot pass off a pre-stop image.
3. `inferencing`: exactly one inference, using that cycle's goal image point,
   while the executor keeps reporting the same stopped pose (within
   `observation_pose_tolerance_m`/`_rad`). Movement pauses the run.
4. `moving`: the executor selects a candidate, projects it to the ground from
   the stopped pose and follows it until the first of: path end,
   `max_step_seconds`, `max_step_distance_m`, `max_step_yaw_rad`, or
   `result_validity_seconds` after receiving the result.
5. The executor zeroes velocity, settles, and reports `stopped` for that
   inference; the node opens the next cycle.

Any of these pauses the node (an explicit `start` is then required): operator
`stop`, image or heartbeat expiry, executor feedback expiry or replacement,
executor `error`, phase timeouts, inference failure or expired result,
goal/instruction changes, and reaching the goal (`goal_reached` in state).

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
- **Pose** comes from `pose_source`: Go2 odometry (proposed
  `robot_state.position`/`yaw`; missing fields are an error) or dead reckoning
  of the executor's own commands. It anchors goals and steps and enforces step
  bounds; dead reckoning drifts with distance.
- **Quiescence** uses the Go2 node's thresholds. If a standing Go2 sways above
  them, the node pauses with "Executor moved during observation"; tune the Go2
  node's `robot_state` thresholds rather than this check.

## Goals

VAMOS needs a goal point in the image; the original ROS stack takes a goal in
the odometry frame and reprojects it every cycle. The executor does the same:

- **Anchoring**: once per goal revision, at the first settled stop, the goal
  becomes a ground point in the odometry frame. An image goal (`goal x y`) is
  projected through the camera calibration and must hit visible ground within
  `max_distance_m`; a metric goal (`goal_m forward left`) is offset from the
  current pose. `stop`/`start` keep the anchor; a goal change replaces it.
  Image goals use the standing camera view, which differs from what a resting
  robot sees, so prefer `goal_m` when picking goals from a resting preview.
- **Each cycle**: the goal is transformed to the robot frame and projected into
  the image with the same calibration, and that point goes to VAMOS with the
  instruction.
- **Out of view**: the robot turns in place toward the goal's bearing, settles
  and rechecks (at most twice without seeing it).
- **Reached**: within `goal.arrival_tolerance_m`, or when the goal lies in the
  horizontal field of view but nearer than the visible ground (the camera's
  near blind zone, which depends on the mount). The node pauses with
  `Goal reached.` and the executor lays the robot down.

Instructions (`ins`) still steer how VAMOS reaches the goal; they do not choose
the goal.

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
uv run --group zenoh python zenoh/tools/calibrate_camera.py intrinsics --images calib/ --board 9x6 --square-m 0.025 --model fisheye --base zenoh/ground-camera.example.json --output zenoh/ground-camera.json
# 2. Mount from >= 4 non-collinear floor markers (CSV: u,v,x_m,y_m in the robot frame)
uv run --group zenoh python zenoh/tools/calibrate_camera.py mount --camera zenoh/ground-camera.json --markers mount-markers.csv --output zenoh/ground-camera.json
# 3. Verify with different markers; marks the file calibrated only within tolerance
uv run --group zenoh python zenoh/tools/calibrate_camera.py verify --camera zenoh/ground-camera.json --markers check-markers.csv --tolerance-m 0.10 --mark-calibrated zenoh/ground-camera.json
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

## Settings reference

Only `zenoh`, `node.inference.goal` and `executor.camera_calibration` are
required; everything else has a default (see `config.py`), and
`vamos.example.json5` lists the settings most often changed.

| Setting | Meaning |
| --- | --- |
| `keys.prefix`, `keys.camera`, `keys.go2_prefix` | `vamos/...`, the camera key, and the Go2 node prefix |
| `zenoh.node`, `zenoh.client` | Zenoh session configs for the node's listener and for every other process |
| `node.camera_freshness_seconds`, `node.cli_heartbeat_timeout_seconds`, `node.go2_state_timeout_seconds` | Input freshness; camera/heartbeat expiry pauses the run |
| `node.inference.num_samples`, `max_new_tokens`, `temperature`, `disable_compile`, `device`, `model_key` | VAMOS generation |
| `node.inference.goal` | Default goal: `x`/`y` image fractions or `forward_m`/`left_m`, plus instruction |
| `node.inference.stop_go.max_step_seconds` / `max_step_distance_m` / `max_step_yaw_rad` | Step bounds |
| `node.inference.stop_go.stop_settle_seconds` | Required quiescent standing time before `stopped` |
| `node.inference.stop_go.stop_confirm_timeout_seconds` | Wait for `stopped`, including stand-up and turning |
| `node.inference.stop_go.observation_delay_seconds` | Camera latency margin after `stopped` |
| `node.inference.stop_go.observation_pose_tolerance_m` / `_rad` | Pose noise allowed while observing/inferring |
| `node.inference.stop_go.image_wait_timeout_seconds`, `inference_timeout_seconds`, `movement_timeout_seconds`, `executor_timeout_seconds` | Phase and feedback timeouts |
| `executor.camera_calibration`, `allow_assumed_calibration` | Calibration file (relative to the settings file) and the robot-free override |
| `executor.pose_source` | `dead_reckoning` or `go2` |
| `executor.goal.arrival_tolerance_m`, `turn_tolerance_rad` | Arrival and turn-in-place tolerances |
| `executor.selection`, `candidate_index`, `min_path_length_m` | Candidate choice |
| `executor.motion.*` | Pure Pursuit speeds (capped at 1.0 m/s, 1.5 rad/s), lookahead, tolerances |
| `executor.control_frequency_hz`, `node_state_timeout_seconds`, `result_validity_seconds`, `shutdown_timeout_seconds`, `go2.{state,stand,down}_timeout_seconds` | Control rate and timeouts |
| `cli.heartbeat_interval_seconds`, `state_timeout_seconds`, `command_timeout_seconds` | Operator CLI timing |

## Offline tools

- `tools/projection_demo.py --infer` runs one real inference and shows every
  candidate on the ground with the executor's selection;
  `--result ... --image ... --camera ...` replays a saved result without a GPU
  or settings file.

## Tests

```console
uv run --group zenoh python -m unittest discover -s zenoh/tests -v
```

Tests use fake inference engines and the same simulated Go2 as the demo; the
executor tests drive the real node state machine, and one TCP test runs node,
executor, simulated Go2 and CLI together. Separate-process localhost tests
exercise Zenoh transport. GPU inference,
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

On Windows with an RTX 5060, `cli.py --demo` ran the real model, executor and
simulated Go2: the example image goal was anchored about 2.8 m ahead and
reached in 10 stop-go cycles (about 0.65 s per inference), then the robot lay
down. Earlier, before the Go2 executor: Ubuntu 24.04 under WSL2 x86_64 with
an RTX 5060 (Python 3.11.17, Torch 2.13.0+cu130) ran real 10-path inference,
projection and 2D following, including WSLg windows. Jetson/aarch64 and
physical Go2 operation have not been validated.
