# -*- coding: utf-8 -*-
"""
mujoco_sim_node.py —— 双足步态仿真 ROS2 节点

职责：加载 ``GaitPlanner`` + ``GaitSim``，订阅步态指令（cmd_vel 的 vx 映射
为目标步速），发布关节状态（sensor_msgs/JointState）与里程计
（nav_msgs/Odometry）。与 standalone 入口（main.py）共用同一套仿真逻辑。
"""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

from humanoid_gait_control.gait_planner import GaitPlanner
from humanoid_gait_control.sim_core import GaitSim

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "models", "humanoid_gait.xml")


class GaitSimNode(Node):
    def __init__(self):
        super().__init__("mujoco_sim_node")

        # 参数（launch 可注入）
        self.declare_parameter("step_len", 0.22)
        self.declare_parameter("step_period", 0.8)
        self.declare_parameter("n_steps", 8)
        self.declare_parameter("com_height", 0.745)
        self.declare_parameter("sim_rate", 500.0)       # Hz（dt=0.002 s）

        step_len = self.get_parameter("step_len").value
        period = self.get_parameter("step_period").value
        n_steps = int(self.get_parameter("n_steps").value)
        com_h = self.get_parameter("com_height").value
        rate = self.get_parameter("sim_rate").value

        self.planner = GaitPlanner(step_len=step_len, step_period=period,
                                   com_height=com_h, n_steps=n_steps)
        self.sim = GaitSim(self.planner, model_path=_DEFAULT_MODEL)
        self._cmd_v = 0.0
        self._started = False

        self._sub = self.create_subscription(Twist, "cmd_vel", self.cb_cmd, 10)
        self._pub_js = self.create_publisher(JointState, "joint_states", 10)
        self._pub_odom = self.create_publisher(Odometry, "odom", 10)
        self._timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"双足步态仿真节点启动 | 步长 {step_len} m 周期 {period} s "
            f"{n_steps} 步 | 站立高度 {self.sim.read_odom()['z']:.3f} m")

    def cb_cmd(self, msg: Twist):
        # vx → 目标步速（m/s），映射为步长×步频；正值前进，负值原地停
        self._cmd_v = float(msg.linear.x)
        if abs(self._cmd_v) > 1e-3 and not self._started:
            self._started = True
            self.get_logger().info(f"收到步速指令 {self._cmd_v:.2f} m/s，开始行走")

    def tick(self):
        if self._started:
            self.sim.step()
        else:
            # 待命：维持站姿（跑一小步让 PD 稳定）
            self.sim.step()
        self._publish()

    def _publish(self):
        od = self.sim.read_odom()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = "base"
        for nm in self.sim.leg_joints + self.sim.arm_joints + self.sim.neck_joints:
            js.name.append(nm)
            js.position.append(float(self.sim.data.qpos[self.sim.qadr[nm]]))
            js.velocity.append(float(self.sim.data.qvel[self.sim.dadr[nm]]))
        self._pub_js.publish(js)

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
        odom.pose.pose.position.x = od["x"]
        odom.pose.pose.position.y = od["y"]
        odom.pose.pose.position.z = od["z"]
        odom.twist.twist.linear.x = od["vx"]
        self._pub_odom.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = GaitSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
