"""
肌肉骨骼人手臂仿真与键盘遥控 Launch 文件
========================================
使用方式::

    ros2 launch humanoid_muscle_arm_control main.launch.py

启动节点
--------
1. ``musculoskeletal_sim_node``  —— MuJoCo 仿真（订阅 ``/cmd_vel``，发布关节/末端/肌肉状态）
2. ``keyboard_teleop_node``       —— 键盘遥控（发布 ``/cmd_vel``）

参数来源：``config/sim_params.yaml``（通过 ``parameters=[LaunchConfiguration('config')]``
注入节点，因此配置文件修改后立即生效），下面只覆盖命令行可改的少量参数。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('humanoid_muscle_arm_control')
    default_model = os.path.join(pkg_share, 'models', 'musculoskeletal_arm.xml')
    default_config = os.path.join(pkg_share, 'config', 'sim_params.yaml')

    return LaunchDescription([
        # ==================== 参数声明 ====================
        DeclareLaunchArgument('config', default_value=default_config,
                              description='仿真与控制器参数文件 (YAML)'),
        DeclareLaunchArgument('model_path', default_value=default_model,
                              description='肌肉骨骼人 MJCF 模型路径'),
        DeclareLaunchArgument('sim_rate', default_value='500.0',
                              description='物理仿真频率 (Hz)'),
        DeclareLaunchArgument('publish_rate', default_value='30.0',
                              description='话题发布频率 (Hz)'),
        DeclareLaunchArgument('hand_speed', default_value='0.18',
                              description='键盘控制末端目标点速度 (m/s)'),

        # ==================== 仿真节点 ====================
        Node(
            package='humanoid_muscle_arm_control',
            executable='musculoskeletal_sim_node',
            name='musculoskeletal_sim_node',
            output='screen',
            parameters=[LaunchConfiguration('config'), {          # 先 YAML，后覆盖
                'model_path': LaunchConfiguration('model_path'),
                'config_path': LaunchConfiguration('config'),
                'sim_rate': LaunchConfiguration('sim_rate'),
                'publish_rate': LaunchConfiguration('publish_rate'),
            }],
        ),

        # ==================== 键盘遥控节点 ====================
        Node(
            package='humanoid_muscle_arm_control',
            executable='keyboard_teleop_node',
            name='keyboard_teleop_node',
            output='screen',
            emulate_tty=True,
            parameters=[LaunchConfiguration('config'), {
                'hand_speed': LaunchConfiguration('hand_speed'),
            }],
        ),
    ])
