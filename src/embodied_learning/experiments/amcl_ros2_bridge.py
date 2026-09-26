"""ROS 2 bridge: replay frozen input to official nav2_amcl and record output.

Run inside WSL with ROS 2 Jazzy sourced:
    source /opt/ros/jazzy/setup.bash
    cd /mnt/d/项目/具身人工智能
    uv run python src/embodied_learning/experiments/amcl_ros2_bridge.py \
        --input results/amcl_bridge_v2 --output results/amcl_official_v1

This script:
  1. Starts map_server, AMCL, and lifecycle_manager as ROS 2 nodes
  2. Activates them via lifecycle transitions
  3. Publishes /scan, /odom, and TF from the frozen input
  4. Sets the initial pose
  5. Records /amcl_pose output
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


class AmclBridge(Node):
    def __init__(self, frozen_dir, output_dir, max_frames=0):
        super().__init__("amcl_bridge_replay")

        frozen = Path(frozen_dir)
        data = dict(np.load(frozen / "frozen_input.npz", allow_pickle=False))
        manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
        self.truth = data["truth"]
        self.odom = data["odom"]
        self.ranges = data["ranges"]
        self.timestamps = data["timestamps"]
        self.n_frames = min(len(self.odom), max_frames) if max_frames else len(self.odom)
        self.map_res = manifest["map_resolution_m"]

        # publishers
        scan_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.scan_pub = self.create_publisher(LaserScan, "scan", scan_qos)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 10)

        # subscriber for AMCL output
        self.amcl_poses = []
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "amcl_pose", self._amcl_callback, 10
        )

        # state
        self._output_dir = Path(output_dir)
        self._frame_idx = 0
        self._start_time = time.monotonic()

    def _amcl_callback(self, msg):
        self.amcl_poses.append(
            {
                "stamp_sec": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "th": msg.pose.pose.orientation.z * 2 + msg.pose.pose.orientation.w,
                "cov_xx": msg.pose.covariance[0],
                "cov_yy": msg.pose.covariance[7],
                "cov_tt": msg.pose.covariance[35],
            }
        )

    def make_scan_msg(self, k):
        msg = LaserScan()
        msg.header.stamp.sec = int(self.timestamps[k])
        msg.header.stamp.nanosec = int((self.timestamps[k] % 1.0) * 1e9)
        msg.header.frame_id = "laser"
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi * 63 / 64
        msg.angle_increment = 2 * math.pi / 64
        msg.time_increment = 0.0
        msg.scan_time = 0.04
        msg.range_min = 0.02
        msg.range_max = 4.0
        msg.ranges = [float(r) for r in self.ranges[k]]
        return msg

    def make_odom_msg(self, k):
        msg = Odometry()
        msg.header.stamp.sec = int(self.timestamps[k])
        msg.header.stamp.nanosec = int((self.timestamps[k] % 1.0) * 1e9)
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(self.odom[k][0])
        msg.pose.pose.position.y = float(self.odom[k][1])
        half = self.odom[k][2] / 2.0
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)
        return msg

    def make_tf_odom_base(self, k):
        t = TransformStamped()
        t.header.stamp.sec = int(self.timestamps[k])
        t.header.stamp.nanosec = int((self.timestamps[k] % 1.0) * 1e9)
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(self.odom[k][0])
        t.transform.translation.y = float(self.odom[k][1])
        half = self.odom[k][2] / 2.0
        t.transform.rotation.z = math.sin(half)
        t.transform.rotation.w = math.cos(half)
        return t

    def make_tf_base_laser(self, k):
        t = TransformStamped()
        t.header.stamp.sec = int(self.timestamps[k])
        t.header.stamp.nanosec = int((self.timestamps[k] % 1.0) * 1e9)
        t.header.frame_id = "base_link"
        t.child_frame_id = "laser"
        t.transform.translation.z = 0.18  # lidar height
        t.transform.rotation.w = 1.0
        return t

    def make_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp.sec = int(self.timestamps[0])
        msg.header.frame_id = "map"
        init = [
            float(self.truth[0][0]),
            float(self.truth[0][1]),
            float(self.truth[0][2]),
        ]
        msg.pose.pose.position.x = init[0]
        msg.pose.pose.position.y = init[1]
        half = init[2] / 2.0
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)
        # covariance: x, y, yaw variances from init_spread
        spread = 0.04
        msg.pose.covariance[0] = spread**2  # x
        msg.pose.covariance[7] = spread**2  # y
        msg.pose.covariance[35] = 0.1**2  # yaw
        return msg

    def publish_frame(self, k):
        self.scan_pub.publish(self.make_scan_msg(k))
        self.odom_pub.publish(self.make_odom_msg(k))
        self.tf_broadcaster.sendTransform(self.make_tf_odom_base(k))
        self.tf_broadcaster.sendTransform(self.make_tf_base_laser(k))


def main():
    parser = argparse.ArgumentParser(description="ROS 2 AMCL bridge replay")
    parser.add_argument("--input", default="results/amcl_bridge_v2")
    parser.add_argument("--output", default="results/amcl_official_v1")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--replay-rate", type=float, default=10.0, help="frames per second")
    args = parser.parse_args()

    rclpy.init()
    bridge = AmclBridge(args.input, args.output, args.max_frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # start map_server, AMCL, and lifecycle_manager via launch file
    import subprocess

    map_file = str(Path(args.input).resolve() / "map.yaml")
    launch_file = "/mnt/d/项目/具身人工智能/scripts/amcl_bridge_launch.py"
    procs = []
    try:
        procs.append(
            subprocess.Popen(
                ["ros2", "launch", launch_file, f"map_file:={map_file}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        # wait for lifecycle manager to activate all nodes
        print("  waiting for lifecycle manager to activate nodes...")
        time.sleep(5.0)

        # publish initial pose
        bridge.initial_pose_pub.publish(bridge.make_initial_pose())
        time.sleep(1.0)

        # replay
        rate = 1.0 / args.replay_rate
        log_every = max(1, bridge.n_frames // 10)
        for k in range(bridge.n_frames):
            bridge.publish_frame(k)
            rclpy.spin_once(bridge, timeout_sec=0.001)
            time.sleep(rate)
            if (k + 1) % log_every == 0:
                n_out = len(bridge.amcl_poses)
                print(f"  frame {k + 1}/{bridge.n_frames}, amcl outputs: {n_out}")

        # save results
        time.sleep(1.0)
        output_data = {
            "amcl_poses": bridge.amcl_poses,
            "n_frames_replayed": bridge.n_frames,
            "replay_rate_hz": args.replay_rate,
        }
        (output_dir / "amcl_output.json").write_text(
            json.dumps(output_data, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            output_dir / "amcl_poses.npz",
            x=np.array([p["x"] for p in bridge.amcl_poses]),
            y=np.array([p["y"] for p in bridge.amcl_poses]),
            th=np.array([p["th"] for p in bridge.amcl_poses]),
            stamp=np.array([p["stamp_sec"] for p in bridge.amcl_poses]),
        )
        print(f"recorded {len(bridge.amcl_poses)} AMCL poses -> {output_dir}")

    finally:
        for proc in procs:
            proc.terminate()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
