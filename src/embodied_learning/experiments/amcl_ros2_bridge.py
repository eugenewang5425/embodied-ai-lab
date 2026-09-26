"""ROS 2 bridge: replay frozen input to official nav2_amcl and record output.

Run inside WSL with ROS 2 Jazzy sourced.

P1 fixes over the first version:
  1. Quaternion -> yaw via standard atan2 (was z*2+w, producing wrong headings)
  2. use_sim_time=true on all nodes + /clock publisher from the bridge node
  3. Single initialization source: /initialpose from frozen data (no hard-coded launch pose)
  4. Subprocess: launch output redirected to file (no PIPE deadlock)
  5. Post-replay spin period to collect remaining AMCL messages
  6. Proper subprocess cleanup (terminate, wait, kill if needed)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


def quat_to_yaw(x, y, z, w):
    """Standard quaternion -> yaw (rotation about Z axis)."""
    sin_y_plus_z = 2.0 * (w * z + x * y)
    cos_y_plus_z = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_y_plus_z, cos_y_plus_z)


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

        # use sim time: bridge node and all published messages use ROS time
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)]
        )

        scan_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.scan_pub = self.create_publisher(LaserScan, "scan", scan_qos)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.clock_pub = self.create_publisher(Clock, "clock", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "initialpose",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.amcl_poses = []
        self.amcl_raw = []
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "amcl_pose", self._amcl_callback, 10
        )
        self._output_dir = Path(output_dir)

    def _amcl_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.amcl_poses.append(
            {
                "stamp_sec": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "yaw": yaw,
                "cov_xx": msg.pose.covariance[0],
                "cov_yy": msg.pose.covariance[7],
                "cov_yaw": msg.pose.covariance[35],
            }
        )
        self.amcl_raw.append(
            {
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
            }
        )

    def publish_clock(self, sim_time_sec):
        msg = Clock()
        msg.clock.sec = int(sim_time_sec)
        msg.clock.nanosec = int((sim_time_sec % 1.0) * 1e9)
        self.clock_pub.publish(msg)

    def make_scan_msg(self, k):
        msg = LaserScan()
        sim_t = self.timestamps[k]
        msg.header.stamp.sec = int(sim_t)
        msg.header.stamp.nanosec = int((sim_t % 1.0) * 1e9)
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
        sim_t = self.timestamps[k]
        msg.header.stamp.sec = int(sim_t)
        msg.header.stamp.nanosec = int((sim_t % 1.0) * 1e9)
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
        sim_t = self.timestamps[k]
        t.header.stamp.sec = int(sim_t)
        t.header.stamp.nanosec = int((sim_t % 1.0) * 1e9)
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
        sim_t = self.timestamps[k]
        t.header.stamp.sec = int(sim_t)
        t.header.stamp.nanosec = int((sim_t % 1.0) * 1e9)
        t.header.frame_id = "base_link"
        t.child_frame_id = "laser"
        t.transform.translation.z = 0.18
        t.transform.rotation.w = 1.0
        return t

    def make_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        sim_t = self.timestamps[0]
        msg.header.stamp.sec = int(sim_t)
        msg.header.stamp.nanosec = int((sim_t % 1.0) * 1e9)
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(self.truth[0][0])
        msg.pose.pose.position.y = float(self.truth[0][1])
        half = self.truth[0][2] / 2.0
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)
        msg.pose.covariance[0] = 0.04**2
        msg.pose.covariance[7] = 0.04**2
        msg.pose.covariance[35] = 0.1**2
        return msg

    def publish_frame(self, k):
        sim_t = self.timestamps[k]
        self.publish_clock(sim_t)
        self.scan_pub.publish(self.make_scan_msg(k))
        self.odom_pub.publish(self.make_odom_msg(k))
        self.tf_broadcaster.sendTransform(self.make_tf_odom_base(k))
        self.tf_broadcaster.sendTransform(self.make_tf_base_laser(k))


def stop_process(proc):
    """Terminate a subprocess gracefully, then kill if needed."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="ROS 2 AMCL bridge replay")
    parser.add_argument("--input", default="results/amcl_bridge_v2")
    parser.add_argument("--output", default="results/amcl_official_v1")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--replay-rate", type=float, default=50.0, help="frames per wall-second")
    parser.add_argument("--post-spin-s", type=float, default=3.0)
    args = parser.parse_args()

    rclpy.init()
    bridge = AmclBridge(args.input, args.output, args.max_frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "launch_log.txt"

    map_file = str(Path(args.input).resolve() / "map.yaml")
    launch_file = "/mnt/d/项目/具身人工智能/scripts/amcl_bridge_launch.py"
    procs = []
    exit_code = 0
    try:
        # launch with log file output (not PIPE, to prevent blocking)
        with open(log_file, "w") as lf:
            procs.append(
                subprocess.Popen(
                    ["ros2", "launch", launch_file, f"map_file:={map_file}"],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            )
            # wait for lifecycle activation (poll with spin)
            print("  waiting for lifecycle activation...")
            for _ in range(50):
                time.sleep(0.2)
                rclpy.spin_once(bridge, timeout_sec=0.001)

            # publish initial pose (the ONLY init source)
            for _ in range(5):
                bridge.initial_pose_pub.publish(bridge.make_initial_pose())
                rclpy.spin_once(bridge, timeout_sec=0.01)
                time.sleep(0.1)
            print("  initial pose published")

            # replay
            rate = 1.0 / args.replay_rate
            log_every = max(1, bridge.n_frames // 10)
            for k in range(bridge.n_frames):
                bridge.publish_frame(k)
                rclpy.spin_once(bridge, timeout_sec=0.001)
                time.sleep(rate)
                if (k + 1) % log_every == 0:
                    print(f"  frame {k + 1}/{bridge.n_frames}, amcl: {len(bridge.amcl_poses)}")

            # post-replay spin: collect remaining AMCL messages
            print(f"  post-replay spin {args.post_spin_s}s...")
            deadline = time.monotonic() + args.post_spin_s
            while time.monotonic() < deadline:
                rclpy.spin_once(bridge, timeout_sec=0.01)
                time.sleep(0.01)

    except (RuntimeError, OSError, ValueError) as e:
        print(f"ERROR: {e}")
        exit_code = 1
    finally:
        for proc in procs:
            stop_process(proc)
        rclpy.shutdown()

    # save results
    output_data = {
        "amcl_poses": bridge.amcl_poses,
        "amcl_raw_quaternions": bridge.amcl_raw,
        "n_frames_replayed": bridge.n_frames,
        "replay_rate_hz": args.replay_rate,
        "exit_code": exit_code,
    }
    (output_dir / "amcl_output.json").write_text(
        json.dumps(output_data, indent=2) + "\n", encoding="utf-8"
    )
    poses = bridge.amcl_poses
    np.savez_compressed(
        output_dir / "amcl_poses.npz",
        x=np.array([p["x"] for p in poses]) if poses else np.zeros(0),
        y=np.array([p["y"] for p in poses]) if poses else np.zeros(0),
        yaw=np.array([p["yaw"] for p in poses]) if poses else np.zeros(0),
        stamp=np.array([p["stamp_sec"] for p in poses]) if poses else np.zeros(0),
    )
    print(f"recorded {len(poses)} AMCL poses -> {output_dir} (exit {exit_code})")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
