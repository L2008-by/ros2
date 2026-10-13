# -*- coding: utf-8 -*-
"""双足步态仿真与遥控 launch 文件。"""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    return LaunchDescription([
        Node(
            package="humanoid_gait_control",
            executable="mujoco_sim_node",
            name="mujoco_sim_node",
            parameters=[{
                "step_len": 0.22,
                "step_period": 0.8,
                "n_steps": 8,
                "com_height": 0.745,
                "sim_rate": 500.0,
            }],
        ),
        Node(
            package="humanoid_gait_control",
            executable="gait_teleop_node",
            name="gait_teleop_node",
        ),
    ])
