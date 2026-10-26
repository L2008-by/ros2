# -*- coding: utf-8 -*-
"""ROS 2 节点：按参数跑一遍爬台阶，并把关节角与基座位姿发出去。

发布：
    /stair_climb/joint_states   sensor_msgs/JointState   六个关节角
    /stair_climb/odom           nav_msgs/Odometry        基座位姿
    /stair_climb/status         std_msgs/String          进度与最终指标
"""

import math
import os

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import humanoid_stair_climb.config as config
from humanoid_stair_climb.gait import StairGait
from humanoid_stair_climb.metrics import RunRecorder, format_summary
from humanoid_stair_climb.sim_core import StairSim

JOINTS = ("hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r")


class StairClimbNode(Node):
    def __init__(self):
        super().__init__("stair_climb_node")
        self.declare_parameter("control_hz", config.CONTROL_HZ)
        self.declare_parameter("time_limit", config.TIME_LIMIT)
        self.declare_parameter("cycle_time", config.CYCLE_TIME)
        self.declare_parameter("swing_ratio", config.SWING_RATIO)
        self.declare_parameter("lift_height", config.LIFT_HEIGHT)
        self.declare_parameter("scene", "")

        self.hz = float(self.get_parameter("control_hz").value)
        self.time_limit = float(self.get_parameter("time_limit").value)
        scene = self.get_parameter("scene").value or None

        self.gait = StairGait(
            cycle_time=float(self.get_parameter("cycle_time").value),
            swing_ratio=float(self.get_parameter("swing_ratio").value),
            lift_height=float(self.get_parameter("lift_height").value),
        )
        self.sim = StairSim(scene)
        self.rec = RunRecorder()

        self.pub_js = self.create_publisher(JointState, "stair_climb/joint_states", 10)
        self.pub_odom = self.create_publisher(Odometry, "stair_climb/odom", 10)
        self.pub_status = self.create_publisher(String, "stair_climb/status", 10)

        self.t = 0.0
        self.base_z = config.STAND_Z
        self.dt = 1.0 / self.hz
        self.done = False
        self.create_timer(self.dt, self.step)
        self.get_logger().info("stair_climb_node 启动：%.1f Hz，场景 %s" % (self.hz, self.sim.model))

    def step(self):
        if self.done:
            return
        self.t += self.dt
        base_x = self.gait.base_x(self.t)
        target_z = config.STAND_Z + self.gait.support_height(self.t)
        self.base_z += (target_z - self.base_z) * min(1.0, self.dt / config.BASE_Z_TAU)
        joints = self.gait.joint_targets(self.t, base_x, self.base_z)
        self.sim.set_state(base_x, self.base_z, joints)
        self.rec.add(self.t, self.sim.base_pos(), joints,
                     self.sim.clearance(), self.gait.support_height(self.t))

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(JOINTS)
        js.position = list(joints)
        self.pub_js.publish(js)

        odom = Odometry()
        odom.header.stamp = js.header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = base_x
        odom.pose.pose.position.z = self.base_z
        odom.pose.pose.orientation = Quaternion(w=1.0)
        self.pub_odom.publish(odom)

        if self.sim.reached() or self.t >= self.time_limit:
            self.done = True
            summary = self.rec.summary()
            self.pub_status.publish(String(data=format_summary(summary)))
            self.get_logger().info(format_summary(summary))


def main(args=None):
    rclpy.init(args=args)
    node = StairClimbNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
