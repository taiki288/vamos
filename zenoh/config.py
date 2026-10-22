"""The single VAMOS settings file and the Zenoh wire contracts.

Every process reads one JSON5 file (default `zenoh/vamos.json5`, copied from
`zenoh/vamos.example.json5`) and uses its own sections. Camera calibration stays
a separate, tool-generated file referenced from the executor section.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import json5
import zenoh
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator

from controller import PurePursuitConfig

JSON_ENCODING = "application/json"
JPEG_ENCODING = "image/jpeg"
PositiveSeconds = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def concrete_key(value: str) -> str:
    if not value or value != value.strip() or "*" in value:
        raise ValueError("expected a concrete Zenoh key without whitespace or wildcards")
    try:
        zenoh.KeyExpr(value)
    except zenoh.ZError as error:
        raise ValueError(f"invalid Zenoh key: {value!r}") from error
    return value


class CameraConfig(Settings):
    key: StrictStr
    frame_freshness_seconds: PositiveSeconds

    _key = field_validator("key")(concrete_key)


class Go2Config(Settings):
    zenoh_key_prefix: StrictStr
    node_state_timeout_seconds: PositiveSeconds

    _key = field_validator("zenoh_key_prefix")(concrete_key)


Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Metres = Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]


class GoalConfig(Settings):
    """A ground goal, given once and then held fixed in the world by the executor.

    Either an image point (x, y as fractions of width/height, interpreted on
    the first standing observation) or a metric offset from the robot when the
    goal is first anchored (forward_m, left_m). Inference always receives image
    fractions, quantized to VAMOS's 0..1023 location tokens.
    """

    x: Fraction | None = None
    y: Fraction | None = None
    forward_m: Metres | None = None
    left_m: Metres | None = None
    instruction: Annotated[StrictStr, Field(max_length=4096)] = ""

    @model_validator(mode="after")
    def one_form(self):
        image, metric = (self.x, self.y), (self.forward_m, self.left_m)
        if not ((None not in image and metric == (None, None)) or (None not in metric and image == (None, None))):
            raise ValueError("a goal needs either x and y (image fractions) or forward_m and left_m (metres)")
        return self

    @property
    def is_image(self) -> bool:
        return self.x is not None

    def describe(self) -> str:
        return (f"image ({self.x:g}, {self.y:g})" if self.is_image
                else f"{self.forward_m:g} m forward, {self.left_m:g} m left")

    def tokens(self) -> tuple[int, int]:
        if not self.is_image:
            raise ValueError("a metric goal has no image point until the executor projects it")
        return min(1023, int(self.x * 1024)), min(1023, int(self.y * 1024))

    def pixels(self, width: int, height: int) -> tuple[float, float]:
        x, y = self.tokens()
        return min(width - 1.0, x * width / 1024), min(height - 1.0, y * height / 1024)


class StopGoConfig(Settings):
    inference_timeout_seconds: PositiveSeconds = 60.0
    max_step_seconds: PositiveSeconds = 1.0
    max_step_distance_m: PositiveSeconds = 0.2
    max_step_yaw_rad: PositiveSeconds = 0.35
    stop_settle_seconds: PositiveSeconds = 0.2
    executor_timeout_seconds: PositiveSeconds = 1.0
    # Waiting for stopped feedback includes a Go2 stand-up on the first cycle.
    stop_confirm_timeout_seconds: PositiveSeconds = 15.0
    # Frames must be received this long after stopped feedback, so camera
    # pipeline latency cannot pass off a pre-stop image as the observation.
    observation_delay_seconds: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] = 0.2
    image_wait_timeout_seconds: PositiveSeconds = 5.0
    movement_timeout_seconds: PositiveSeconds = 4.0
    # Executor pose changes up to these are odometry noise, not motion.
    observation_pose_tolerance_m: PositiveSeconds = 0.05
    observation_pose_tolerance_rad: PositiveSeconds = 0.05

    @model_validator(mode="after")
    def timeouts(self):
        if self.movement_timeout_seconds <= self.max_step_seconds + self.stop_settle_seconds:
            raise ValueError("movement timeout must exceed step duration plus settling time")
        if self.image_wait_timeout_seconds <= self.observation_delay_seconds:
            raise ValueError("image wait timeout must exceed the observation delay")
        return self


class InferenceConfig(Settings):
    stop_go: StopGoConfig = Field(default_factory=StopGoConfig)
    model_key: Annotated[StrictStr, Field(min_length=1)] = "mateoguaman/vamos"
    device: Literal["auto", "cuda", "cpu"] = "cuda"
    # "auto" keeps bfloat16 except on CUDA GPUs older than Ampere (compute
    # capability < 8.0, e.g. Pascal P100), which lack bfloat16 and use float32.
    dtype: Literal["auto", "bfloat16", "float32"] = "auto"
    num_samples: Annotated[int, Field(strict=True, ge=1, le=50)] = 10
    max_new_tokens: Annotated[int, Field(strict=True, ge=2, le=128)] = 10
    temperature: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.05
    disable_compile: StrictBool = True
    frequency_hz: PositiveSeconds = 1.0  # How often the worker checks for an observation to infer.
    shutdown_timeout_seconds: PositiveSeconds = 30.0
    goal: GoalConfig

    @model_validator(mode="after")
    def generation_settings(self) -> InferenceConfig:
        if self.temperature == 0 and self.num_samples != 1:
            raise ValueError("greedy decoding requires num_samples=1")
        if self.max_new_tokens % 2:
            raise ValueError("max_new_tokens must contain coordinate pairs")
        if not math.isfinite(1.0 / self.frequency_hz):
            raise ValueError("inference period must be finite")
        return self


class NodeConfig(Settings):
    zenoh_key_prefix: StrictStr
    node_state_publish_frequency_hz: PositiveSeconds
    cli_heartbeat_timeout_seconds: PositiveSeconds
    camera: CameraConfig
    go2: Go2Config
    inference: InferenceConfig | None = None

    _key = field_validator("zenoh_key_prefix")(concrete_key)

    @field_validator("node_state_publish_frequency_hz")
    @classmethod
    def finite_period(cls, value: float) -> float:
        if not math.isfinite(1.0 / value) or 1.0 / value <= 0:
            raise ValueError("publish frequency must have a finite positive period")
        return value

    @model_validator(mode="after")
    def distinct_keys(self) -> NodeConfig:
        keys = list(self.keys().values())
        if len(set(keys)) != len(keys):
            raise ValueError("camera, control, state, and Go2 keys must be distinct")
        return self

    def keys(self) -> dict[str, str]:
        return {
            "camera": self.camera.key,
            "command": f"{self.zenoh_key_prefix}/command",
            "state": f"{self.zenoh_key_prefix}/state",
            "trajectories": f"{self.zenoh_key_prefix}/trajectories",
            "preview": f"{self.zenoh_key_prefix}/preview",
            "go2_state": f"{self.go2.zenoh_key_prefix}/state",
            "go2_command": f"{self.go2.zenoh_key_prefix}/command",
            "executor_state": f"{self.zenoh_key_prefix}/executor/state",
        }


class Go2RobotState(BaseModel):
    # Go2's independently versioned node publishes additional observation fields.
    model_config = ConfigDict(extra="ignore", frozen=True)
    state: StrictStr
    motion: Literal["moving", "quiescent", "unknown"]
    # Proposed odometry fields (SportModeState.position and IMU yaw); optional
    # until unitree-go2-zenoh-node publishes them.
    position: tuple[float, float, float] | None = None
    yaw: float | None = None


class Go2NodeState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    robot_connected: StrictBool
    robot_state: Go2RobotState
    accepting_commands: StrictBool


class ControlCommand(Settings):
    type: Literal["heartbeat", "start", "stop", "goal"]
    request_id: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None = None
    node_session_id: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None = None
    expected_goal_revision: Annotated[int, Field(strict=True, ge=0)] | None = None
    goal: GoalConfig | None = None

    @model_validator(mode="after")
    def command_fields(self):
        if (self.type == "goal") != (self.goal is not None):
            raise ValueError("only a goal command must contain a complete goal")
        if self.type == "heartbeat" and any(v is not None for v in
                (self.request_id, self.node_session_id, self.expected_goal_revision)):
            raise ValueError("heartbeat must not contain control metadata")
        if self.type == "stop" and self.expected_goal_revision is not None:
            raise ValueError("stop must not depend on a goal revision")
        return self


class ControlAck(Settings):
    request_id: str
    type: Literal["start", "stop", "goal"]
    accepted: bool
    message: str


class CameraState(Settings):
    received_frames: int
    fresh: bool
    age_seconds: float | None
    width: int | None
    height: int | None


class Go2State(Settings):
    received_states: int
    fresh: bool
    age_seconds: float | None
    robot_connected: bool
    accepting_commands: bool
    robot_state: Go2RobotState | None


class InferenceState(Settings):
    model_loaded: bool = False
    enabled: bool = False
    active: bool = False
    device: str | None = None
    completed: int = 0
    discarded: int = 0
    last_frame_sequence: int | None = None
    last_duration_seconds: float | None = None
    last_error: str | None = None


class CycleState(Settings):
    phase: Literal["paused", "waiting_stop", "waiting_image", "inferencing", "moving"] = "paused"
    cycle_id: int = 0
    completed: int = 0
    inference_id: int | None = None
    executor_fresh: bool = False
    executor_motion: Literal["stopped", "moving", "settling", "error"] | None = None
    goal_image: tuple[float, float] | None = None  # The goal fractions used for this cycle.
    goal_distance_m: float | None = None


class NodeState(Settings):
    cycle: CycleState = Field(default_factory=CycleState)
    node_session_id: str = ""
    control_epoch: int = 0
    goal_revision: int = 0
    goal: GoalConfig | None = None
    last_control_ack: ControlAck | None = None
    stage: Literal["transport", "inference"] = "transport"
    lifecycle: Literal["loading", "idle", "running", "paused", "error", "stopping", "stopped"]
    start_ready: bool = False
    status_message: str = "Transport diagnostics only; no inference configured."
    cli_connected: bool
    received_commands: int
    last_command: str | None
    rejected_samples: dict[str, int]
    camera: CameraState
    go2: Go2State
    inference: InferenceState = Field(default_factory=InferenceState)
    goal_reached: bool = False


FinitePixel = Annotated[float, Field(allow_inf_nan=False)]
FiniteFloat = FinitePixel


class PlanarPose(Settings):
    x: FiniteFloat = 0.0
    y: FiniteFloat = 0.0
    yaw: FiniteFloat = 0.0


class ExecutorState(Settings):
    # Published by go2_executor.py. "stopped" is confirmed by Go2 observations;
    # pose and path are dead-reckoned from the executor's own commands.
    executor_session_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    node_session_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    control_epoch: Annotated[int, Field(strict=True, ge=0)]
    cycle_id: Annotated[int, Field(strict=True, ge=0)]
    motion: Literal["stopped", "moving", "settling", "error"]
    inference_id: int | None = None
    pose: PlanarPose
    path_m: list[tuple[FiniteFloat, FiniteFloat]] = []  # Current step's ground path, same frame as pose.
    # With "stopped" while the node waits for it: where the world-fixed goal
    # appears now, or that it has been reached.
    goal_image: tuple[Fraction, Fraction] | None = None
    goal_distance_m: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    goal_reached: bool = False
    goal_world_m: tuple[FiniteFloat, FiniteFloat] | None = None  # Anchored goal, same frame as pose.
    message: str = ""


class TrajectoryResult(Settings):
    generated_at_unix_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    cycle_id: int | None = None
    node_session_id: str | None = None
    control_epoch: int | None = None
    goal_revision: int | None = None
    inference_id: int
    frame_sequence: int
    image_width: Annotated[int, Field(gt=0)]
    image_height: Annotated[int, Field(gt=0)]
    coordinate_frame: Literal["image_pixels"] = "image_pixels"
    goal_pixels: tuple[FinitePixel, FinitePixel]
    prompt: str
    trajectories: list[list[tuple[FinitePixel, FinitePixel]]]
    generated_texts: list[str]
    duration_seconds: float
    source_age_seconds: float
    device: str

    @model_validator(mode="after")
    def valid_paths(self) -> TrajectoryResult:
        if not self.trajectories or any(not path for path in self.trajectories):
            raise ValueError("inference must produce nonempty paths")
        for path in self.trajectories:
            for x, y in path:
                if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                    raise ValueError("trajectory coordinate is outside the source image")
        return self


# --- The settings file --------------------------------------------------------------

ZENOH = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ZENOH / "vamos.json5"
EXAMPLE_CONFIG_PATH = ZENOH / "vamos.example.json5"
DEMO_CONFIG_PATH = ZENOH / "tools" / "demo.json5"


class KeysConfig(Settings):
    prefix: StrictStr = "vamos"
    camera: StrictStr = "camera/front"
    go2_prefix: StrictStr = "unitree/go2"

    _keys = field_validator("prefix", "camera", "go2_prefix")(concrete_key)


class ZenohConnections(Settings):
    """Zenoh session configs, passed to Zenoh as-is: node listens, every other process connects."""

    node: dict[str, Any]
    client: dict[str, Any]

    @field_validator("node", "client")
    @classmethod
    def valid_zenoh_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            zenoh.Config.from_json5(json.dumps(value))
        except (zenoh.ZError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Zenoh config: {error}") from error
        return value

    def session(self, role: Literal["node", "client"]) -> zenoh.Config:
        return zenoh.Config.from_json5(json.dumps(getattr(self, role)))


class NodeSettings(Settings):
    state_publish_frequency_hz: PositiveSeconds = 10.0
    cli_heartbeat_timeout_seconds: PositiveSeconds = 1.0
    camera_freshness_seconds: PositiveSeconds = 2.5
    go2_state_timeout_seconds: PositiveSeconds = 0.5
    inference: InferenceConfig


class Go2Timing(Settings):
    state_timeout_seconds: PositiveSeconds = 0.5
    stand_timeout_seconds: PositiveSeconds = 10.0
    down_timeout_seconds: PositiveSeconds = 10.0


class ExecutorMotion(Settings):
    # Deliberately low caps for first hardware trials; the Go2 contract allows more.
    forward_speed_mps: Annotated[float, Field(strict=True, gt=0, le=1.0, allow_inf_nan=False)] = 0.3
    max_yaw_rate_rps: Annotated[float, Field(strict=True, gt=0, le=1.5, allow_inf_nan=False)] = 0.6
    lookahead_distance_m: PositiveSeconds = 0.4
    goal_tolerance_m: PositiveSeconds = 0.05
    slowdown_distance_m: PositiveSeconds = 0.2
    rotate_threshold_rad: PositiveSeconds = 0.6

    def pure_pursuit(self) -> PurePursuitConfig:
        return PurePursuitConfig(**self.model_dump())


class GoalSettings(Settings):
    arrival_tolerance_m: PositiveSeconds = 0.5
    turn_tolerance_rad: PositiveSeconds = 0.05


class ExecutorConfig(Settings):
    camera_calibration: Annotated[StrictStr, Field(min_length=1)]
    pose_source: Literal["dead_reckoning", "go2"] = "dead_reckoning"
    allow_assumed_calibration: StrictBool = False
    selection: Literal["medoid", "index"] = "medoid"
    candidate_index: Annotated[int, Field(strict=True, ge=0)] = 0
    min_path_length_m: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] = 0.05
    control_frequency_hz: Annotated[float, Field(strict=True, ge=10, le=100, allow_inf_nan=False)] = 20.0
    node_state_timeout_seconds: PositiveSeconds = 1.0
    result_validity_seconds: PositiveSeconds = 3.0
    shutdown_timeout_seconds: PositiveSeconds = 15.0
    go2: Go2Timing = Field(default_factory=Go2Timing)
    motion: ExecutorMotion = Field(default_factory=ExecutorMotion)
    goal: GoalSettings = Field(default_factory=GoalSettings)

    @model_validator(mode="after")
    def valid_controller(self):
        self.motion.pure_pursuit()
        if self.shutdown_timeout_seconds <= self.go2.down_timeout_seconds:
            raise ValueError("shutdown timeout must exceed the Go2 down timeout")
        return self


class CliSettings(Settings):
    heartbeat_interval_seconds: PositiveSeconds = 0.2
    state_timeout_seconds: PositiveSeconds = 1.0
    command_timeout_seconds: PositiveSeconds = 15.0


class VamosConfig(Settings):
    keys: KeysConfig = Field(default_factory=KeysConfig)
    zenoh: ZenohConnections
    node: NodeSettings
    executor: ExecutorConfig
    cli: CliSettings = Field(default_factory=CliSettings)

    @model_validator(mode="after")
    def consistent(self):
        self.node_config()  # Key collisions and periods are checked by the node model.
        if self.cli.heartbeat_interval_seconds >= self.node.cli_heartbeat_timeout_seconds:
            raise ValueError("cli.heartbeat_interval_seconds must be below node.cli_heartbeat_timeout_seconds")
        return self

    def node_config(self) -> NodeConfig:
        """The node's runtime view: keys, freshness limits and inference settings."""
        return NodeConfig(
            zenoh_key_prefix=self.keys.prefix,
            node_state_publish_frequency_hz=self.node.state_publish_frequency_hz,
            cli_heartbeat_timeout_seconds=self.node.cli_heartbeat_timeout_seconds,
            camera=CameraConfig(key=self.keys.camera, frame_freshness_seconds=self.node.camera_freshness_seconds),
            go2=Go2Config(zenoh_key_prefix=self.keys.go2_prefix,
                          node_state_timeout_seconds=self.node.go2_state_timeout_seconds),
            inference=self.node.inference,
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> VamosConfig:
    """Load and validate the settings file; relative calibration paths follow the file."""
    if not path.is_file():
        hint = (f"; copy {EXAMPLE_CONFIG_PATH.name} to {path.name} and edit it"
                if path == DEFAULT_CONFIG_PATH else "")
        raise ValueError(f"settings file not found: {path}{hint}")
    document = json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False)
    try:
        config = VamosConfig.model_validate(document)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    calibration = Path(config.executor.camera_calibration)
    if not calibration.is_absolute():
        calibration = (path.resolve().parent / calibration).resolve()
        executor = config.executor.model_copy(update={"camera_calibration": str(calibration)})
        config = config.model_copy(update={"executor": executor})
    return config
