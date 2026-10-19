"""Transport settings and wire contracts, based on NaVILA's Zenoh node."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import json5
import zenoh
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator

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


class GoalConfig(Settings):
    # Fractions of image width/height; quantized to VAMOS's 0..1023 loc tokens.
    x: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    y: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    instruction: Annotated[StrictStr, Field(max_length=4096)] = ""

    def tokens(self) -> tuple[int, int]:
        return min(1023, int(self.x * 1024)), min(1023, int(self.y * 1024))

    def pixels(self, width: int, height: int) -> tuple[float, float]:
        x, y = self.tokens()
        return min(width - 1.0, x * width / 1024), min(height - 1.0, y * height / 1024)


class InferenceConfig(Settings):
    model_key: Annotated[StrictStr, Field(min_length=1)] = "mateoguaman/vamos"
    device: Literal["auto", "cuda", "cpu"] = "cuda"
    num_samples: Annotated[int, Field(strict=True, ge=1, le=50)] = 10
    max_new_tokens: Annotated[int, Field(strict=True, ge=2, le=128)] = 10
    temperature: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.05
    disable_compile: StrictBool = True
    frequency_hz: PositiveSeconds = 1.0
    max_frame_age_seconds: PositiveSeconds = 5.0
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
        }


class Go2RobotState(BaseModel):
    # Go2's independently versioned node publishes additional observation fields.
    model_config = ConfigDict(extra="ignore", frozen=True)
    state: StrictStr
    motion: Literal["moving", "quiescent", "unknown"]


class Go2NodeState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    robot_connected: StrictBool
    robot_state: Go2RobotState
    accepting_commands: StrictBool


class ControlCommand(Settings):
    type: Literal["heartbeat", "start", "stop"]


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


class NodeState(Settings):
    stage: Literal["transport", "inference"] = "transport"
    lifecycle: Literal["loading", "idle", "running", "paused", "error", "stopping", "stopped"]
    start_ready: Literal[False] = False
    status_message: str = "Transport only; inference and robot commands are not implemented."
    cli_connected: bool
    received_commands: int
    last_command: str | None
    rejected_samples: dict[str, int]
    camera: CameraState
    go2: Go2State
    inference: InferenceState = Field(default_factory=InferenceState)


FinitePixel = Annotated[float, Field(allow_inf_nan=False)]


class CapturePose(Settings):
    x: FinitePixel = 0.0
    y: FinitePixel = 0.0
    yaw: FinitePixel = 0.0


class CaptureContext(Settings):
    # Demo-only pose annotation. Ordinary camera JPEGs need no attachment and
    # still support image inference, but cannot drive the streaming simulator.
    kind: Literal["static_image_demo"] = "static_image_demo"
    source_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    frame_id: Annotated[int, Field(strict=True, ge=0)]
    captured_at_unix_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    world_frame: Literal["static_demo_world"] = "static_demo_world"
    pose: CapturePose = Field(default_factory=CapturePose)


class TrajectoryResult(Settings):
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
    capture: CaptureContext | None = None

    @model_validator(mode="after")
    def valid_paths(self) -> TrajectoryResult:
        if not self.trajectories or any(not path for path in self.trajectories):
            raise ValueError("inference must produce nonempty paths")
        for path in self.trajectories:
            for x, y in path:
                if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                    raise ValueError("trajectory coordinate is outside the source image")
        return self


def load_node_config(path: Path) -> NodeConfig:
    document = json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False)
    return NodeConfig.model_validate(document)
