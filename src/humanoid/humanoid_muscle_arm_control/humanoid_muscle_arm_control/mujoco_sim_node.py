# -*- coding: utf-8 -*-
"""
肌肉骨骼人手臂 MuJoCo 仿真节点
==============================

职责
----
* 加载自建肌肉骨骼人上肢模型（3 自由度 + 6 块 Hill 型肌肉）；
* 按 ``/cmd_vel`` 的速度指令平移末端目标点，逐物理步推进仿真；
* 发布 ``/joint_states``、``/musculoskeletal/hand_pose``、``/musculoskeletal/activation``、
  ``/musculoskeletal/muscle_force``，并用 **tf2_ros.TransformBroadcaster** 广播 ``world → hand``
  （``/tf`` 的类型必须是 ``tf2_msgs/TFMessage``，不能用 geometry_msgs 直接发布）。

参数全部通过 ``declare_parameter`` 声明，``launch/main.launch.py`` 会把
``config/sim_params.yaml`` 注入进来，因此配置文件是真正生效的。
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import tf2_ros

try:                                                     # 安装后（colcon）与源码直跑都兼容
    from humanoid_muscle_arm_control.sim_core import (
        MusculoskeletalArm, ArmParams, JOINT_NAMES, MUSCLE_NAMES)
except ImportError:                                      # pragma: no cover
    from sim_core import MusculoskeletalArm, ArmParams, JOINT_NAMES, MUSCLE_NAMES


class MusculoskeletalSimNode(Node):
    """肌肉骨骼人手臂仿真节点。"""

    def __init__(self):
        super().__init__('musculoskeletal_sim_node')

        # ---------------- 参数（由 launch 注入 sim_params.yaml） ----------------
        defaults = ArmParams()
        self.declare_parameter('model_path', '')
        self.declare_parameter('config_path', '')
        self.declare_parameter('sim_rate', defaults.sim_rate)
        self.declare_parameter('publish_rate', defaults.publish_rate)
        self.declare_parameter('kp_task', defaults.kp_task)
        self.declare_parameter('kd_task', defaults.kd_task)
        self.declare_parameter('kp_joint', defaults.kp_joint)
        self.declare_parameter('kd_joint', defaults.kd_joint)
        self.declare_parameter('control_mode', defaults.control_mode)
        self.declare_parameter('f_task_max', defaults.f_task_max)
        self.declare_parameter('tau_task_max', defaults.tau_task_max)
        self.declare_parameter('bias_compensation', defaults.bias_compensation)
        self.declare_parameter('cocontraction', defaults.cocontraction)
        self.declare_parameter('tau_act', defaults.tau_act)
        self.declare_parameter('tau_deact', defaults.tau_deact)
        self.declare_parameter('moment_refresh_steps', defaults.moment_refresh_steps)
        self.declare_parameter('hand_speed', defaults.hand_speed)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        sim_rate = self.get_parameter('sim_rate').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        params = ArmParams.from_dict({
            'sim_rate': sim_rate,
            'publish_rate': publish_rate,
            'kp_task': self.get_parameter('kp_task').get_parameter_value().double_value,
            'kd_task': self.get_parameter('kd_task').get_parameter_value().double_value,
            'kp_joint': self.get_parameter('kp_joint').get_parameter_value().double_value,
            'kd_joint': self.get_parameter('kd_joint').get_parameter_value().double_value,
            'control_mode': self.get_parameter('control_mode').get_parameter_value().string_value,
            'f_task_max': self.get_parameter('f_task_max').get_parameter_value().double_value,
            'tau_task_max': self.get_parameter('tau_task_max').get_parameter_value().double_value,
            'bias_compensation': self.get_parameter('bias_compensation').get_parameter_value().bool_value,
            'cocontraction': self.get_parameter('cocontraction').get_parameter_value().double_value,
            'tau_act': self.get_parameter('tau_act').get_parameter_value().double_value,
            'tau_deact': self.get_parameter('tau_deact').get_parameter_value().double_value,
            'moment_refresh_steps': self.get_parameter('moment_refresh_steps').get_parameter_value().integer_value,
            'hand_speed': self.get_parameter('hand_speed').get_parameter_value().double_value,
        })

        # ---------------- 仿真核心 ----------------
        self.arm = MusculoskeletalArm(model_path=model_path or None, params=params)
        self.hand_speed = params.hand_speed
        self.cmd = np.zeros(3)

        # ---------------- 订阅 / 发布 ----------------
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self._cmd_callback, 10)
        self.pub_joint = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_hand = self.create_publisher(PoseStamped, '/musculoskeletal/hand_pose', 10)
        self.pub_activation = self.create_publisher(
            Float64MultiArray, '/musculoskeletal/activation', 10)
        self.pub_force = self.create_publisher(Float64MultiArray, '/musculoskeletal/muscle_force', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)          # ★ 正确的 /tf 广播方式

        # ---------------- 定时器 ----------------
        self.sim_timer = self.create_timer(1.0 / max(1.0, sim_rate), self._sim_step)
        self.pub_timer = self.create_timer(1.0 / max(1.0, publish_rate), self._publish)

        # ---------------- 启动自检（力矩臂 + 主动肌/拮抗肌配对） ----------------
        self.get_logger().info(
            f'肌肉骨骼人仿真节点已启动 | {sim_rate:.0f} Hz | 关节 {len(JOINT_NAMES)} | '
            f'肌肉 {len(MUSCLE_NAMES)} | 具体模型: {self.arm.model_path}')
        self.get_logger().info('\n' + self.arm.moment_arm_table())
        for item in self.arm.antagonist_check():
            self.get_logger().info(
                f"  关节 {item['joint']:<16} 主动肌 {len(item['positive'])} 块 / "
                f"拮抗肌 {len(item['negative'])} 块 | 配对 {'OK' if item['paired'] else '缺失'}")

    # ------------------------------------------------------------------
    def _cmd_callback(self, msg: Twist):
        """速度指令 → 末端目标点平移速度（世界系 x 前 / y 左 / z 上）。"""
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=float)

    def _sim_step(self):
        if np.any(self.cmd != 0.0):                       # 速度指令 → 目标点平移
            delta = self.cmd * self.hand_speed * self.arm.model.opt.timestep
            self.arm.set_target(self.arm.target() + delta)
        self.arm.step()

    def _publish(self):
        now = self.get_clock().now().to_msg()
        state = self.arm.state()

        joint = JointState()
        joint.header.stamp = now
        joint.header.frame_id = 'world'
        joint.name = list(JOINT_NAMES)
        joint.position = [float(v) for v in state['q']]
        self.pub_joint.publish(joint)

        hand = PoseStamped()
        hand.header.stamp = now
        hand.header.frame_id = 'world'
        hand.pose.position.x = float(state['hand'][0])
        hand.pose.position.y = float(state['hand'][1])
        hand.pose.position.z = float(state['hand'][2])
        hand.pose.orientation.w = 1.0
        self.pub_hand.publish(hand)

        activation = Float64MultiArray()
        activation.data = [float(v) for v in state['activation']]
        self.pub_activation.publish(activation)

        force = Float64MultiArray()
        force.data = [float(v) for v in state['force']]
        self.pub_force.publish(force)

        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'musculoskeletal/hand'
        transform.transform.translation.x = float(state['hand'][0])
        transform.transform.translation.y = float(state['hand'][1])
        transform.transform.translation.z = float(state['hand'][2])
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)                       # ★ 走 /tf（TFMessage）


def main(args=None):
    rclpy.init(args=args)
    node = MusculoskeletalSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.arm.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
