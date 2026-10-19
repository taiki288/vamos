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


class NodeConfig(Settings):
    zenoh_key_prefix: StrictStr
    node_state_publish_frequency_hz: PositiveSeconds
    cli_heartbeat_timeout_seconds: PositiveSeconds
    camera: CameraConfig
    go2: Go2Config

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
    # Navigation start and goal commands will be added with inference/control.
    type: Literal["heartbeat", "stop"]


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


class NodeState(Settings):
    stage: Literal["transport"] = "transport"
    lifecycle: Literal["idle", "stopped"]
    start_ready: Literal[False] = False
    status_message: str = "Transport only; inference and robot commands are not implemented."
    cli_connected: bool
    received_commands: int
    last_command: str | None
    rejected_samples: dict[str, int]
    camera: CameraState
    go2: Go2State


def load_node_config(path: Path) -> NodeConfig:
    document = json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False)
    return NodeConfig.model_validate(document)
