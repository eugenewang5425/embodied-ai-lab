"""WSL-only ROS 2 nodes. Three processes communicate through actual DDS topics.

Run via experiments.ros2_system on Windows or ros2_probe on Linux. No ROS import
is required by the earlier Windows lessons. No physical device is connected.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from embodied_learning.differential_drive import SENSOR_IN_BODY
from embodied_learning.ros2_stream import StreamingFusion, frame_stamp, stamp_frame

PREFIX = "/lesson20"
ENCODERS = PREFIX + "/encoders"
LANDMARK_POINTS = PREFIX + "/landmark_points"
ODOM = PREFIX + "/odom_pose"
FUSED = PREFIX + "/fused_pose"
STEP = PREFIX + "/step"
READY = PREFIX + "/ready"
QOS = QoSProfile(
    depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)


def stamp(frame):
    sec, ns = frame_stamp(frame)
    return Time(sec=sec, nanosec=ns)


def read_frame(header, expected_frame):
    if header.frame_id != expected_frame:
        raise ValueError(f"Expected frame_id={expected_frame}, got {header.frame_id}")
    return stamp_frame(header.stamp.sec, header.stamp.nanosec)


def fill_pose(target, pose):
    target.position.x, target.position.y = float(pose[0]), float(pose[1])
    target.orientation.z = math.sin(pose[2] / 2)
    target.orientation.w = math.cos(pose[2] / 2)


def read_pose(pose):
    values = np.array(
        [pose.position.x, pose.position.y, 2 * math.atan2(pose.orientation.z, pose.orientation.w)]
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite pose message")
    return values


def pose_message(pose, frame, coordinate_frame):
    msg = PoseStamped(header=Header(stamp=stamp(frame), frame_id=coordinate_frame))
    fill_pose(msg.pose, pose)
    return msg


def transform_message(pose, frame, parent, child):
    msg = TransformStamped(header=Header(stamp=stamp(frame), frame_id=parent), child_frame_id=child)
    msg.transform.translation.x = float(pose[0])
    msg.transform.translation.y = float(pose[1])
    msg.transform.rotation.z = math.sin(pose[2] / 2)
    msg.transform.rotation.w = math.cos(pose[2] / 2)
    return msg


def points_message(readings, frame):
    # FLOAT64 preserves the existing measurements; these are 3 synthetic control
    # points, not a camera/depth/LiDAR point cloud or a new sensor model.
    fields = [
        PointField(name=name, offset=offset, datatype=datatype, count=1)
        for name, offset, datatype in (
            ("x", 0, PointField.FLOAT64),
            ("y", 8, PointField.FLOAT64),
            ("z", 16, PointField.FLOAT64),
            ("landmark_id", 24, PointField.UINT32),
        )
    ]
    points = [
        (float(r * np.cos(b)), float(r * np.sin(b)), 0.0, i + 1)
        for i, (r, b) in enumerate(readings)
    ]
    return point_cloud2.create_cloud(Header(stamp=stamp(frame), frame_id="sensor"), fields, points)


def readings_from_points(msg):
    rows = point_cloud2.read_points_list(
        msg, field_names=("x", "y", "z", "landmark_id"), skip_nans=False
    )
    if len(rows) != 3 or sorted(int(row.landmark_id) for row in rows) != [1, 2, 3]:
        raise ValueError("Expected exactly the three known landmark IDs")
    points = np.array(
        [[row.x, row.y, row.z] for row in sorted(rows, key=lambda row: row.landmark_id)]
    )
    if not np.isfinite(points).all() or np.any(points[:, 2] != 0):
        raise ValueError("Expected finite planar sensor points")
    return np.column_stack(
        [np.linalg.norm(points[:, :2], axis=1), np.arctan2(points[:, 1], points[:, 0])]
    )


class SensorReplay(Node):
    def __init__(self, directory, observations_first=False):
        super().__init__("lesson20_sensor_replay")
        self.observations_first = observations_first
        with np.load(Path(directory) / "sensor_input.npz", allow_pickle=False) as data:
            self.encoders = data["encoders"].copy()
            self.frames = data["observation_frames"].copy()
            self.observations = data["observations"].copy()
        self.encoder_pub = self.create_publisher(JointState, ENCODERS, QOS)
        self.landmark_pub = self.create_publisher(PointCloud2, LANDMARK_POINTS, QOS)
        self.clock_pub = self.create_publisher(Clock, "/clock", QOS)
        self.step_service = self.create_service(Trigger, STEP, self.step)
        self.index = 0
        self.get_logger().info(f"sensor process PID={os.getpid()}, waiting for step service calls")

    def step(self, request, response):
        if self.index >= len(self.encoders):
            response.success, response.message = False, "finished"
            return response
        if (
            self.encoder_pub.get_subscription_count() < 2
            or self.landmark_pub.get_subscription_count() < 2
        ):
            response.success, response.message = False, "waiting for sensor subscribers"
            return response
        frame = self.index
        self.clock_pub.publish(Clock(clock=stamp(frame)))
        encoders = JointState(
            header=Header(stamp=stamp(frame), frame_id="base_link"),
            name=["left_wheel", "right_wheel"],
            position=self.encoders[frame].tolist(),
        )
        match = np.flatnonzero(self.frames == frame)
        cloud = points_message(self.observations[int(match[0])], frame) if len(match) else None
        if cloud is not None and self.observations_first:
            self.landmark_pub.publish(cloud)
        self.encoder_pub.publish(encoders)
        if cloud is not None and not self.observations_first:
            self.landmark_pub.publish(cloud)
        self.index += 1
        response.success, response.message = True, str(frame)
        return response


class Localizer(Node):
    def __init__(self):
        super().__init__(
            "lesson20_localizer", parameter_overrides=[Parameter("use_sim_time", value=True)]
        )
        self.stream = StreamingFusion()
        self.odom_pub = self.create_publisher(PoseStamped, ODOM, QOS)
        self.fused_pub = self.create_publisher(PoseStamped, FUSED, QOS)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform(transform_message(SENSOR_IN_BODY, 0, "base_link", "sensor"))
        self.encoder_sub = self.create_subscription(JointState, ENCODERS, self.on_encoders, QOS)
        self.landmark_sub = self.create_subscription(
            PointCloud2, LANDMARK_POINTS, self.on_landmarks, QOS
        )
        self.get_logger().info(f"localizer PID={os.getpid()}; no recording/truth file loaded")
        self.ready_service = self.create_service(Trigger, READY, self.ready)

    def ready(self, request, response):
        response.success = (
            self.odom_pub.get_subscription_count() >= 1
            and self.fused_pub.get_subscription_count() >= 1
            and self.encoder_sub.get_publisher_count() >= 1
            and self.landmark_sub.get_publisher_count() >= 1
        )
        response.message = "ready" if response.success else "waiting for matched endpoints"
        return response

    def on_encoders(self, msg):
        frame = read_frame(msg.header, "base_link")
        if frame == 0:
            self.get_logger().info("received initial encoders")
        if (
            len(msg.name) != 2
            or set(msg.name) != {"left_wheel", "right_wheel"}
            or len(msg.position) != 2
        ):
            raise ValueError("Invalid wheel message")
        wheels = [msg.position[msg.name.index(name)] for name in ("left_wheel", "right_wheel")]
        self.publish(self.stream.add_encoders(frame, wheels))

    def on_landmarks(self, msg):
        frame = read_frame(msg.header, "sensor")
        self.publish(self.stream.add_observation(frame, readings_from_points(msg)))

    def publish(self, rows):
        for row in rows:
            frame = row["frame"]
            if frame == 0:
                self.get_logger().info("publishing initial poses")
            self.odom_pub.publish(pose_message(row["odom"], frame, "odom"))
            self.fused_pub.publish(pose_message(row["fused"], frame, "map"))
            self.tf.sendTransform(
                [
                    transform_message(row["map_to_odom"], frame, "map", "odom"),
                    transform_message(row["odom"], frame, "odom", "base_link"),
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["sensor", "localizer"])
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--observations-first", action="store_true")
    args = parser.parse_args()
    rclpy.init()
    node = (
        SensorReplay(args.directory, args.observations_first)
        if args.role == "sensor"
        else Localizer()
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
