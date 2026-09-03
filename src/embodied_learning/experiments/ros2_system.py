"""Windows entry: export lesson-19 inputs, run real ROS nodes in WSL, retain evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PureWindowsPath

import numpy as np

from embodied_learning.experiments.landmark_fusion import (
    DEFAULT_RESULTS as SOURCE_RESULTS,
)
from embodied_learning.experiments.landmark_fusion import (
    load_recording,
)

DEFAULT_RESULTS = "results/ros2_system_2026-09-03_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wsl_path(path):
    value = PureWindowsPath(str(path))
    if not value.is_absolute() or len(value.drive) != 2 or value.drive[1] != ":":
        raise ValueError("Need an absolute Windows drive path for WSL")
    return f"/mnt/{value.drive[0].lower()}/" + "/".join(value.parts[1:])


def prepare(output, source, route="square", run_index=0):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise FileExistsError(output)
    routes, report = load_recording(source)
    if route not in routes or type(run_index) is not int or not 0 <= run_index < report["runs"]:
        raise ValueError("Unknown route or invalid run index")
    selected = routes[route]
    output.mkdir(parents=True, exist_ok=False)
    # Only source publishes these readings. No truth or precomputed pose is in this file.
    np.savez_compressed(
        output / "sensor_input.npz",
        encoders=selected["encoders"][run_index],
        observations=selected["observations"][run_index],
        observation_frames=selected["observation_frames"],
    )
    # Inspector-only reference; the localizer receives neither file path.
    np.savez_compressed(
        output / "reference.npz",
        truth=selected["truth"],
        odom=selected["odom"][run_index],
        fused=selected["fused"][run_index],
    )
    source_names = [
        "differential_drive.py",
        "odometry.py",
        "pose_fusion.py",
        "landmark_localization.py",
        "ros2_stream.py",
        "ros2_nodes.py",
        "ros2_probe.py",
        "experiments/ros2_system.py",
    ]
    manifest = {
        "route": route,
        "run_index": run_index,
        "steps": len(selected["truth"]) - 1,
        "observation_frames": selected["observation_frames"].tolist(),
        "source_recording": str(source),
        "source_recording_sha256": digest(source / "trajectories.npz"),
        "source_summary_sha256": digest(source / "summary.json"),
        "sha256": {name: digest(output / name) for name in ("sensor_input.npz", "reference.npz")},
        "source_sha256": {
            name: digest(PROJECT_ROOT / "src" / "embodied_learning" / name) for name in source_names
        },
    }
    (output / "input.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_experiment(
    output,
    *,
    source=SOURCE_RESULTS,
    route="square",
    run_index=0,
    observations_first=False,
    hold_seconds=0,
    domain_id=87,
):
    if os.name != "nt":
        raise RuntimeError(
            "Use this entry on Windows; Linux prepared runs use embodied_learning.ros2_probe"
        )
    if type(domain_id) is not int or not 0 <= domain_id <= 232 or not 0 <= hold_seconds <= 600:
        raise ValueError("Invalid ROS domain or hold duration")
    prepare(output, source, route, run_index)
    output = Path(output).resolve()
    # Arguments are positional shell parameters, never interpolated shell commands.
    script = """set -e
source /opt/ros/jazzy/setup.bash
cd "$1"
export PYTHONPATH="$1/src:$PYTHONPATH"
export ROS_LOG_DIR="$2/ros_logs"
export ROS_DOMAIN_ID="$3"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONUNBUFFERED=1
exec python3 -m embodied_learning.ros2_probe --directory "$2" --hold-seconds "$4" $5
"""
    command = [
        "wsl.exe",
        "-d",
        "Ubuntu-24.04",
        "--",
        "bash",
        "-s",
        "--",
        wsl_path(PROJECT_ROOT),
        wsl_path(output),
        str(domain_id),
        str(hold_seconds),
        "--observations-first" if observations_first else "",
    ]
    # The Linux runner owns/cleans up its two child processes; no daemon or service left running.
    with (output / "wsl.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            input=script.encode("utf-8"),  # Preserve LF instead of Windows text-mode CRLF.
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"WSL run failed ({result.returncode}); inspect {output / 'wsl.log'}")
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if not report.get("owned_processes_stopped"):
        raise RuntimeError("ROS child cleanup was not verified")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_RESULTS))
    parser.add_argument("--route", choices=["straight", "square", "long"], default="square")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--observations-first", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0)
    parser.add_argument("--domain-id", type=int, default=87)
    args = parser.parse_args()
    report = run_experiment(
        args.output,
        source=args.source,
        route=args.route,
        run_index=args.run_index,
        observations_first=args.observations_first,
        hold_seconds=args.hold_seconds,
        domain_id=args.domain_id,
    )
    print("ROS messages:", report["received_counts"])
    print("Maximum difference from lesson 19:", report["reference_max_abs_difference"])
    print("TF chain difference:", report["tf_chain_max_abs_difference"])
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
