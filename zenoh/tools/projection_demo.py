"""Inspect one VAMOS result on flat ground with the executor's candidate selection.

Runs one real inference (--infer) or replays a saved TrajectoryResult
(--result), projects every candidate with the given camera calibration, and
shows which one go2_executor.py would choose and why. Nothing moves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from config import DEFAULT_CONFIG_PATH, InferenceConfig, TrajectoryResult, load_config
from projection import GroundCamera, GroundProjector
from selection import Selection, select_candidate

ZENOH = Path(__file__).resolve().parents[1]


def infer_once(image: Image.Image, config: InferenceConfig) -> TrajectoryResult:
    from inference import VAMOSEngine  # Replay never imports torch.

    engine = VAMOSEngine.load(config)
    started = time.monotonic()
    output = engine.infer(image)
    duration = time.monotonic() - started
    return TrajectoryResult(inference_id=1, frame_sequence=1, image_width=image.width,
                            image_height=image.height, goal_pixels=config.goal.pixels(*image.size),
                            prompt=output.prompt, trajectories=output.trajectories,
                            generated_texts=output.generated_texts, duration_seconds=duration,
                            source_age_seconds=duration, device=output.device)


def comparison_figure(image, result: TrajectoryResult, camera: GroundCamera, selection: Selection):
    import matplotlib.pyplot as plt

    figure, (image_axes, ground_axes) = plt.subplots(1, 2, figsize=(13, 6))
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.16, top=0.83, wspace=0.22)
    figure.suptitle("VAMOS image paths -> flat-ground paths", fontsize=16, y=0.98)
    status = camera.calibration_status.upper()
    figure.text(0.04, 0.91, f"{status} camera | flat ground, zero roll", fontsize=10,
                color="darkred" if status == "ASSUMED" else "dimgray")
    image_axes.imshow(image)
    image_axes.set_title("Source image / pixel coordinates")
    horizon = camera.cy - camera.fy * math.tan(math.radians(camera.pitch_down_degrees))
    if 0 <= horizon < image.height:
        image_axes.axhline(horizon, color="yellow", lw=1, ls=":", label="Assumed horizon")
    projected = {path.candidate_index: path for path in GroundProjector(camera).project_paths(result.trajectories)}
    for index, pixels in enumerate(result.trajectories):
        color, chosen = plt.get_cmap("tab10")(index % 10), index == selection.index
        image_axes.plot(*zip(*pixels), "o-" if chosen else ".--", color=color, lw=3 if chosen else 1,
                        alpha=1 if chosen else 0.6, label=f"Selected #{index}" if chosen else None)
        points = projected[index].points_robot_m
        if points:
            ground_axes.plot(*zip(*points), "o-", color=color, lw=3 if chosen else 1, alpha=1 if chosen else 0.4)
            ground_axes.text(points[-1][0], points[-1][1], str(index), color=color, fontsize=9)
    image_axes.plot(*result.goal_pixels, "rx", ms=10, mew=2)
    image_axes.set_xlim(0, image.width)
    image_axes.set_ylim(image.height, 0)
    image_axes.legend(loc="upper left", fontsize=8)
    ground_axes.plot(0, 0, ">", color="black", ms=10, label="Robot")
    ground_axes.set_title("Ground projection (robot frame)")
    ground_axes.set_xlabel("Forward x [m]")
    ground_axes.set_ylabel("Left y [m]")
    ground_axes.set_aspect("equal", adjustable="datalim")
    ground_axes.grid(True, alpha=0.25)
    ground_axes.legend(loc="best", fontsize=8)
    figure.text(0.04, 0.055, f"{selection.strategy}: {selection.reason}.\n"
                "Geometric validity does not establish obstacle clearance.", fontsize=9)
    return figure


def run(args) -> int:
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    # The settings file is only needed for live inference or the default calibration.
    settings = load_config(args.config) if args.infer or args.camera is None else None
    camera = GroundCamera.load(args.camera or Path(settings.executor.camera_calibration))
    camera.check_image_size(*image.size)
    result = (infer_once(image, settings.node.inference) if args.infer else
              TrajectoryResult.model_validate_json(args.result.read_text(encoding="utf-8")))
    camera.check_image_size(result.image_width, result.image_height)
    selection = select_candidate(result.trajectories, GroundProjector(camera),
                                 strategy="index" if args.candidate is not None else "medoid",
                                 index=args.candidate or 0)
    report = {"inference_id": result.inference_id, "camera": camera.as_json(),
              "coordinate_frame": "robot_ground_flu_m", "selection_strategy": selection.strategy,
              "selected_candidate": selection.index, "selection_reason": selection.reason,
              "selected_path_m": selection.path.points_robot_m if selection.usable else None,
              "candidates": [report.as_dict() for report in selection.candidates]}
    if args.headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = comparison_figure(image, result, camera, selection)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        image.save(args.output_dir / "source.png")
        (args.output_dir / "input_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (args.output_dir / "projection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                                encoding="utf-8")
        figure.savefig(args.output_dir / "projection.png", dpi=140)
        print(f"Output: {args.output_dir.resolve()}")
    print(json.dumps({key: report[key] for key in ("selection_strategy", "selected_candidate", "selection_reason")}))
    if args.headless:
        plt.close("all")
    else:
        plt.show()
    return 0 if selection.usable else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--infer", action="store_true", help="Run one real VAMOS inference")
    mode.add_argument("--result", type=Path, help="Replay a saved TrajectoryResult JSON for this exact image")
    parser.add_argument("--image", type=Path, default=ZENOH.parent / "test_images/texas_person.jpg")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Settings file for --infer and the default calibration (default zenoh/vamos.json5)")
    parser.add_argument("--camera", type=Path, help="Calibration file instead of the settings file's")
    parser.add_argument("--candidate", type=int, help="Fixed candidate index instead of the medoid")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Save source, input JSON, report and plot")
    args = parser.parse_args()
    if args.candidate is not None and args.candidate < 0:
        parser.error("--candidate must be nonnegative")
    try:
        return run(args)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"Projection demo failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
