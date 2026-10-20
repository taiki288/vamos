"""Adapter for the existing VAMOS demo; transport-only mode imports no ML stack."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw

from config import GoalConfig, InferenceConfig, TrajectoryResult


@dataclass(frozen=True)
class InferenceOutput:
    trajectories: list
    generated_texts: list[str]
    prompt: str
    device: str


def build_prompt(config: InferenceConfig, goal: GoalConfig | None = None) -> str:
    goal = goal or config.goal
    x, y = goal.tokens()
    prompt = f"Navigate to x=<loc{x:04d}>, y=<loc{y:04d}>."
    instruction = goal.instruction.strip()
    return f"{prompt} {instruction}" if instruction else prompt


class VAMOSEngine:
    def __init__(self, model, processor, device, generate, config: InferenceConfig):
        self.model, self.processor, self.device = model, processor, device
        self.generate, self.config = generate, config

    @classmethod
    def load(cls, config: InferenceConfig) -> VAMOSEngine:
        import torch

        # Script entry points must also work when launched outside the repo root.
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from vamos_demo import generate_trajectories, load_vamos_model

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
        model, processor, actual_device = load_vamos_model(config.model_key, device=device)
        if model is None or processor is None:
            raise RuntimeError(f"Could not load VAMOS model: {config.model_key}")
        model.eval()
        return cls(model, processor, actual_device, generate_trajectories, config)

    def infer(self, image: Image.Image, goal: GoalConfig | None = None) -> InferenceOutput:
        import torch

        prompt = build_prompt(self.config, goal)
        paths, texts, error = self.generate(
            self.model, self.processor, image, prompt,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            num_samples=self.config.num_samples,
            device=self.device,
            disable_compile=self.config.disable_compile,
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        if error:
            raise RuntimeError(error)
        valid_paths, valid_texts = [], []
        for path, text in zip(paths, texts):
            if path and all(len(point) == 2 and all(math.isfinite(v) for v in point)
                            and 0 <= point[0] < image.width and 0 <= point[1] < image.height
                            for point in path):
                valid_paths.append(path)
                valid_texts.append(text)
        if not valid_paths:
            raise RuntimeError("VAMOS returned no valid paths inside the source image")
        return InferenceOutput(valid_paths, valid_texts, prompt, self.device)


def render_preview(image: Image.Image, result: TrajectoryResult) -> bytes:
    """Overlay on the exact input frame, not on a newer camera observation."""
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    colors = ("cyan", "yellow", "magenta", "lime", "orange", "red", "blue", "white")
    for index, path in enumerate(result.trajectories):
        points = [(float(x), float(y)) for x, y in path]
        color = colors[index % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    x, y = result.goal_pixels
    draw.line((x - 9, y - 9, x + 9, y + 9), fill="red", width=3)
    draw.line((x - 9, y + 9, x + 9, y - 9), fill="red", width=3)
    draw.rectangle((0, 0, min(preview.width, 600), 25), fill="black")
    draw.text((5, 5), f"frame={result.frame_sequence} paths={len(result.trajectories)} "
              f"{result.duration_seconds:.2f}s {result.device}", fill="white")
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
