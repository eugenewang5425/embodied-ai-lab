"""ROS 2 launch file for the AMCL bridge (map_server + AMCL + lifecycle_manager)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    map_file = LaunchConfiguration("map_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # map_server node
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": map_file},
            {"use_sim_time": use_sim_time},
        ],
    )

    # AMCL node
    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"alpha1": 0.10},
            {"alpha2": 0.10},
            {"alpha3": 0.10},
            {"alpha4": 0.10},
            {"alpha5": 0.10},
            {"set_initial_pose": True},
            {"initial_pose.x": 3.575},
            {"initial_pose.y": 3.402},
            {"initial_pose.z": 0.0},
            {"initial_pose.yaw": -0.040},
            {"scan_topic": "scan"},
            {"map_topic": "map"},
            {"odom_frame_id": "odom"},
            {"base_frame_id": "base_link"},
            {"scan_frame_id": "laser"},
        ],
    )

    # lifecycle manager to auto-configure and activate both nodes
    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {"node_names": ["map_server", "amcl"]},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("map_file", default_value=""),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            map_server_node,
            amcl_node,
            lifecycle_manager_node,
        ]
    )
