"""Inspect one frozen VAMOS result on flat ground and simulate one candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from PIL import Image

from controller import Pose2D
from projection import GroundCamera, GroundProjector, path_to_world
from simulator import Scenario, Simulation, SimulatorView


ROOT = Path(__file__).resolve().parents[1]


def infer_once(image: Image.Image, config_path: Path):
    # Offline replay does not import torch or load model weights.
    from config import TrajectoryResult, load_node_config
    from inference import VAMOSEngine

    config = load_node_config(config_path).inference
    if config is None:
        raise ValueError("node configuration must include inference settings")
    engine = VAMOSEngine.load(config)
    started = time.monotonic()
    output = engine.infer(image)
    duration = time.monotonic() - started
    return TrajectoryResult(inference_id=1, frame_sequence=1, image_width=image.width,
                            image_height=image.height, goal_pixels=config.goal.pixels(*image.size),
                            prompt=output.prompt, trajectories=output.trajectories,
                            generated_texts=output.generated_texts, duration_seconds=duration,
                            source_age_seconds=duration, device=output.device)


def comparison_figure(image, result, camera, candidates, selected_index):
    import matplotlib.pyplot as plt

    figure, (image_axes, ground_axes) = plt.subplots(1, 2, figsize=(13, 6))
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.16, top=0.83, wspace=0.22)
    figure.suptitle("VAMOS image paths -> flat-ground paths", fontsize=16, y=0.98)
    status = camera.calibration_status.upper()
    figure.text(0.04, 0.91, f"{status} camera | flat ground, zero roll | capture pose fixed at (0, 0, 0)",
                color="darkred" if status == "ASSUMED" else "dimgray", fontsize=10)
    image_axes.imshow(image)
    image_axes.set_title("Source image / pixel coordinates")
    image_axes.set_xlabel("u [px]")
    image_axes.set_ylabel("v [px]")
    # Yaw leaves vertical ray components unchanged for this zero-roll model.
    import math

    horizon = camera.cy - camera.fy * math.tan(math.radians(camera.pitch_down_degrees))
    if 0 <= horizon < image.height:
        image_axes.axhline(horizon, color="yellow", lw=1, ls=":", label="Assumed horizon")
    for index, (pixels, candidate) in enumerate(zip(result.trajectories, candidates)):
        color = plt.get_cmap("tab10")(index % 10)
        us, vs = zip(*pixels)
        image_axes.plot(us, vs, ".--", color=color, lw=1, alpha=0.65)
        image_axes.text(us[-1], vs[-1], str(index), fontsize=8, color="white",
                        bbox={"facecolor": color, "alpha": 0.75, "pad": 1, "edgecolor": "none"})
        if candidate.points_robot_m:
            xs, ys = zip(*candidate.points_robot_m)
            selected = index == selected_index
            ground_axes.plot(xs, ys, "o-", color=color, lw=3 if selected else 1,
                             alpha=1 if selected else 0.4, ms=5 if selected else 3)
            ground_axes.text(xs[-1], ys[-1], str(index), color=color, fontsize=9)
        if index == selected_index and candidate.retained_source_indices:
            retained = [pixels[i] for i in candidate.retained_source_indices]
            image_axes.plot(*zip(*retained), "o-", color=color, lw=3, ms=5, label=f"Selected #{index}")
    image_axes.plot(*result.goal_pixels, "rx", ms=10, mew=2)
    image_axes.set_xlim(0, image.width)
    image_axes.set_ylim(image.height, 0)
    image_axes.legend(loc="upper left", fontsize=8)
    ground_axes.plot(0, 0, ">", color="black", ms=10, label="Robot at capture")
    ground_axes.set_title(f"Ground projection / selected candidate #{selected_index}")
    ground_axes.set_xlabel("Forward x [m]")
    ground_axes.set_ylabel("Left y [m]")
    ground_axes.set_aspect("equal", adjustable="datalim")
    ground_axes.grid(True, alpha=0.25)
    ground_axes.legend(loc="best", fontsize=8)
    candidate = candidates[selected_index]
    reason = candidate.stop_reason or "all points projected"
    figure.text(0.04, 0.055,
                f"Selected #{selected_index}: {len(candidate.points_robot_m)} retained points; {reason}.\n"
                "Assumed calibration gives illustrative distances. Geometric validity does not establish obstacle clearance.",
                fontsize=9)
    return figure


def run(args) -> int:
    from config import TrajectoryResult

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    camera = GroundCamera.load(args.camera)
    camera.check_image_size(*image.size)
    result = (infer_once(image, args.inference_config) if args.infer else
              TrajectoryResult.model_validate_json(args.result.read_text(encoding="utf-8")))
    camera.check_image_size(result.image_width, result.image_height)
    if not 0 <= args.candidate < len(result.trajectories):
        raise ValueError(f"candidate must be in [0, {len(result.trajectories) - 1}]")
    candidates = GroundProjector(camera).project_paths(result.trajectories)
    selected = candidates[args.candidate]
    capture_pose = Pose2D(0, 0, 0)
    path_world = path_to_world(selected.points_robot_m, capture_pose)
    report = {"inference_id": result.inference_id, "frame_sequence": result.frame_sequence,
              "camera": asdict(camera), "coordinate_frame": "capture_robot_ground_flu_m",
              "capture_pose_world": asdict(capture_pose), "selected_candidate": args.candidate,
              "selected_path_world_m": path_world,
              "candidates": [candidate.as_dict() for candidate in candidates]}
    if args.headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = comparison_figure(image, result, camera, candidates, args.candidate)
    sim = None
    view = None
    reached = False
    if selected.usable:
        scenario = Scenario(f"Projected candidate #{args.candidate}", path_world, capture_pose)
        sim = Simulation(scenario)
        if args.headless:
            reached = sim.run(args.max_time)
            report["simulation"] = sim.summary()
        view = SimulatorView(sim, show_scenarios=False,
                             description=f"{camera.calibration_status.upper()} camera / frozen projected path / ideal motion")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        image.save(args.output_dir / "source.png")
        (args.output_dir / "input_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (args.output_dir / "projection_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        figure.savefig(args.output_dir / "projection.png", dpi=140)
        if view is not None:
            view.figure.savefig(args.output_dir / "simulation.png", dpi=140)
        else:
            # Do not leave a previous successful run's plot beside a rejected
            # selection's new report when reusing the output directory.
            (args.output_dir / "simulation.png").unlink(missing_ok=True)
        print(f"Output: {args.output_dir.resolve()}")
    print(json.dumps({"calibration_status": camera.calibration_status,
                      "selected_candidate": args.candidate, "usable": selected.usable,
                      "truncated": selected.truncated, "stop_reason": selected.stop_reason,
                      "retained_points": len(selected.points_robot_m),
                      "simulation": sim.summary() if sim is not None else None}, allow_nan=False))
    if args.headless:
        plt.close("all")
        return 0 if reached else 1
    if view is not None:
        view.show()  # Opens the comparison and simulation windows together.
    else:
        print("Selected candidate has fewer than two valid prefix points; simulation skipped.", file=sys.stderr)
        plt.show()
    return 0 if selected.usable else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--infer", action="store_true", help="Run one real VAMOS inference")
    mode.add_argument("--result", type=Path, help="Replay a saved TrajectoryResult JSON for this exact image")
    parser.add_argument("--image", type=Path, default=ROOT / "test_images/texas_person.jpg")
    parser.add_argument("--camera", type=Path, default=Path(__file__).with_name("ground-camera.example.json"))
    parser.add_argument("--inference-config", type=Path,
                        default=Path(__file__).with_name("inference-node-config.example.json5"))
    parser.add_argument("--candidate", type=int, default=0, help="Explicit candidate index; no automatic safety ranking")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-time", type=float, default=120, help="Headless simulation time budget [s]")
    parser.add_argument("--output-dir", type=Path, help="Save source, input JSON, projection report and plots")
    args = parser.parse_args()
    import math

    if not math.isfinite(args.max_time) or args.max_time <= 0:
        parser.error("--max-time must be finite and positive")
    try:
        return run(args)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"Projection demo failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
