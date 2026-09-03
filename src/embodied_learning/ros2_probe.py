"""Linux ROS integration runner: launch two owned processes, audit, then stop them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time as RosTime
from sensor_msgs.msg import JointState, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from embodied_learning.differential_drive import SENSOR_IN_BODY, compose
from embodied_learning.odometry import heading_error
from embodied_learning.ros2_nodes import (
    ENCODERS,
    FUSED,
    LANDMARK_POINTS,
    ODOM,
    QOS,
    READY,
    STEP,
    read_frame,
    read_pose,
    stamp,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_pose(transform):
    t, q = transform.translation, transform.rotation
    return np.array([t.x, t.y, 2 * np.arctan2(q.z, q.w)])


def pose_difference(a, b):
    return max(
        float(np.max(np.abs(np.asarray(a)[:2] - np.asarray(b)[:2]))),
        abs(float(heading_error(a[2], b[2]))),
    )


class Inspector(Node):
    def __init__(self):
        super().__init__(
            "lesson20_inspector", parameter_overrides=[Parameter("use_sim_time", value=True)]
        )
        self.received = {key: {} for key in ("encoders", "landmarks", "odom", "fused")}
        self.events = []
        self.last_tf_error = "not queried"
        self.subscriptions_owned = [
            self.create_subscription(
                message_type, topic, lambda msg, k=key, f=frame: self.receive(k, msg, f), QOS
            )
            for key, message_type, topic, frame in (
                ("encoders", JointState, ENCODERS, "base_link"),
                ("landmarks", PointCloud2, LANDMARK_POINTS, "sensor"),
                ("odom", PoseStamped, ODOM, "odom"),
                ("fused", PoseStamped, FUSED, "map"),
            )
        ]
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.client = self.create_client(Trigger, STEP)
        self.ready_client = self.create_client(Trigger, READY)

    def receive(self, key, msg, coordinate_frame):
        frame = read_frame(msg.header, coordinate_frame)
        if frame in self.received[key]:
            raise ValueError(f"Duplicate received {key} frame {frame}")
        self.received[key][frame] = msg
        self.events.append({"frame": frame, "topic": key})

    def wait_for(self, predicate, children, seconds=15):
        deadline = time.monotonic() + seconds
        while not predicate():
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A lesson node exited early; inspect sensor.log/localizer.log")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Waiting for matching messages/TF timed out: "
                    f"counts={ {k: len(v) for k, v in self.received.items()} }; "
                    f"TF={self.last_tf_error}; frames={self.buffer.all_frames_as_yaml()}"
                )
            rclpy.spin_once(self, timeout_sec=0.005)

    def tf_at(self, frame):
        when = RosTime.from_msg(stamp(frame))
        try:
            return {
                "map_to_sensor": transform_pose(
                    self.buffer.lookup_transform("map", "sensor", when).transform
                ),
                "map_to_odom": transform_pose(
                    self.buffer.lookup_transform("map", "odom", when).transform
                ),
            }
        except TransformException as error:
            self.last_tf_error = str(error)
            return None


def run(directory, *, observations_first=False, hold_seconds=0):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "input.json").read_text(encoding="utf-8"))
    for name in ("sensor_input.npz", "reference.npz"):
        if digest(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    with np.load(directory / "reference.npz", allow_pickle=False) as data:
        reference = {key: value.copy() for key, value in data.items()}
    trace_path = directory / "ros_trace.jsonl"
    if trace_path.exists() or (directory / "summary.json").exists():
        raise FileExistsError("ROS output already exists; select a fresh directory")
    children, logs, rows = [], [], []
    inspector = None
    rclpy.init()
    start = time.monotonic()
    success = False
    try:
        for role in ("sensor", "localizer"):
            log = (directory / f"{role}.log").open("x", encoding="utf-8")
            logs.append(log)
            command = [sys.executable, "-m", "embodied_learning.ros2_nodes", role]
            if role == "sensor":
                command += ["--directory", str(directory)]
                if observations_first:
                    command.append("--observations-first")
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        inspector = Inspector()
        inspector.wait_for(
            lambda: (
                inspector.client.service_is_ready()
                and inspector.count_publishers(FUSED) == 1
                and inspector.count_publishers(ODOM) == 1
                and inspector.count_subscribers(ENCODERS) == 2
                and inspector.count_subscribers(LANDMARK_POINTS) == 2
            ),
            children,
        )
        inspector.wait_for(inspector.ready_client.service_is_ready, children)
        deadline = time.monotonic() + 15
        while True:
            ready = inspector.ready_client.call_async(Trigger.Request())
            inspector.wait_for(ready.done, children)
            if ready.result().success:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Localizer endpoints did not match")
        expected_nodes = {"lesson20_sensor_replay", "lesson20_localizer", "lesson20_inspector"}
        inspector.wait_for(
            lambda: expected_nodes.issubset(set(inspector.get_node_names())), children
        )
        graph = {
            "nodes": sorted(inspector.get_node_names()),
            "topics": dict(inspector.get_topic_names_and_types()),
            "services": dict(inspector.get_service_names_and_types()),
        }
        expected_frames = range(manifest["steps"] + 1)
        observed_frames = set(manifest["observation_frames"])
        with trace_path.open("x", encoding="utf-8") as trace:
            for frame in expected_frames:
                future = inspector.client.call_async(Trigger.Request())
                inspector.wait_for(future.done, children)
                response = future.result()
                if not response.success or response.message != str(frame):
                    raise RuntimeError(f"Step failed at {frame}: {response.message}")
                inspector.wait_for(
                    lambda frame=frame: (
                        all(
                            frame in inspector.received[key]
                            for key in ("encoders", "odom", "fused")
                        )
                        and (
                            frame not in observed_frames or frame in inspector.received["landmarks"]
                        )
                        and inspector.tf_at(frame) is not None
                    ),
                    children,
                )
                odom = read_pose(inspector.received["odom"][frame].pose)
                fused = read_pose(inspector.received["fused"][frame].pose)
                transforms = inspector.tf_at(frame)
                difference = max(
                    pose_difference(odom, reference["odom"][frame]),
                    pose_difference(fused, reference["fused"][frame]),
                )
                tf_difference = pose_difference(
                    transforms["map_to_sensor"], compose(fused, SENSOR_IN_BODY)
                )
                if difference > 1e-9 or tf_difference > 1e-9:
                    raise ValueError(
                        f"Frame {frame} changed math: pose={difference}, TF={tf_difference}"
                    )
                row = {
                    "frame": frame,
                    "time_s": frame * 0.04,
                    "stamp_sec": stamp(frame).sec,
                    "stamp_nanosec": stamp(frame).nanosec,
                    "observation": frame in observed_frames,
                    "odom": odom.tolist(),
                    "fused": fused.tolist(),
                    "truth": reference["truth"][frame].tolist(),
                    "map_to_odom": transforms["map_to_odom"].tolist(),
                    "map_to_sensor": transforms["map_to_sensor"].tolist(),
                    "reference_max_abs_difference": difference,
                    "tf_chain_max_abs_difference": tf_difference,
                    "received_counts": {
                        key: len(value) for key, value in inspector.received.items()
                    },
                }
                rows.append(row)
                trace.write(json.dumps(row) + "\n")
                if frame % 100 == 0:
                    print(
                        f"ROS frame {frame}/{manifest['steps']}: messages + TF checked", flush=True
                    )
        # End-of-recording service response must not silently replay the last frame.
        future = inspector.client.call_async(Trigger.Request())
        inspector.wait_for(future.done, children)
        if future.result().success or future.result().message != "finished":
            raise ValueError("Unexpected end-of-recording response")
        # A newly subscribed tf2 listener must receive the retained static mount.
        late_buffer = Buffer()
        late_listener = TransformListener(late_buffer, inspector)
        inspector.wait_for(
            lambda: late_buffer.can_transform("base_link", "sensor", RosTime()), children
        )
        late = transform_pose(
            late_buffer.lookup_transform("base_link", "sensor", RosTime()).transform
        )
        if pose_difference(late, SENSOR_IN_BODY) > 1e-12:
            raise ValueError("Late listener got a wrong static transform")
        late_listener.unregister()
        if hold_seconds:
            print(
                f"Holding live ROS graph for {hold_seconds:g} seconds for CLI inspection",
                flush=True,
            )
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(inspector, timeout_sec=0.1)
        report = {
            "experiment": "ros2_message_and_tf_bridge",
            "schema_version": 1,
            "route": manifest["route"],
            "run_index": manifest["run_index"],
            "steps": manifest["steps"],
            "dt_s": 0.04,
            "observation_frames": manifest["observation_frames"],
            "process_ids": {
                "inspector": os.getpid(),
                "sensor": children[0].pid,
                "localizer": children[1].pid,
            },
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "discovery_range": os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "graph": graph,
            "received_counts": rows[-1]["received_counts"],
            "reference_max_abs_difference": max(
                row["reference_max_abs_difference"] for row in rows
            ),
            "tf_chain_max_abs_difference": max(row["tf_chain_max_abs_difference"] for row in rows),
            "late_static_listener_passed": True,
            "observations_published_first": observations_first,
            "wall_seconds": time.monotonic() - start,
            "trace_sha256": digest(trace_path),
            "input_manifest_sha256": digest(directory / "input.json"),
            "limitations": [
                "Recorded sensor replay, not a new physics simulation or hardware run",
                "Reliable local DDS with step/ack flow control; not a real-time/dropout/latency benchmark",
                "Matched stamps are known from one simulation clock; not physical clock synchronisation",
                "PointCloud2 contains exactly 3 synthetic known points, not image recognition",
                "ROS transports data and TF; no new localisation algorithm or accuracy improvement",
            ],
        }
        success = True
    finally:
        if inspector is not None:
            inspector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in logs:
            log.close()
    if success:
        report["owned_processes_stopped"] = all(child.poll() is not None for child in children)
        (directory / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "counts": report["received_counts"],
                    "max_difference": report["reference_max_abs_difference"],
                    "tf_difference": report["tf_chain_max_abs_difference"],
                    "children_stopped": report["owned_processes_stopped"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--observations-first", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 600:
        parser.error("--hold-seconds must be between 0 and 600")
    run(args.directory, observations_first=args.observations_first, hold_seconds=args.hold_seconds)


if __name__ == "__main__":
    main()
