# -*- coding: utf-8 -*-
"""
gait_teleop_node.py —— 双足步态键盘遥控 ROS2 节点

按键（termios raw 模式，POSIX only）：
  W/S   目标步速 +0.05 / -0.05 m/s（限幅 0.05~0.45）
  R     重置仿真（回站立位）
  ESC   退出
按任一方向键后以 20 Hz 持续发布 cmd_vel，松开按键即发布全零速度。
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

HELP = """
============================================================
  双足步态键盘遥控
------------------------------------------------------------
  W / S     目标步速 +0.05 / -0.05 m/s（0.05 ~ 0.45 m/s）
  R         重置仿真（回双脚并立站姿）
  ESC       退出
  按下即发指令，松开按键即停止
============================================================
"""


class GaitTeleopNode(Node):
    def __init__(self):
        super().__init__("gait_teleop_node")
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._speed = 0.20
        self._hold = False
        self._fd = os.open("/dev/tty", os.O_RDONLY)
        self._old = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        self._timer = self.create_timer(0.05, self.tick)
        print(HELP)

    def tick(self):
        try:
            while select.select([self._fd], [], [], 0)[0]:
                ch = os.read(self._fd, 1).decode("utf-8", "ignore")
                self._handle(ch)
        except OSError:
            pass
        msg = Twist()
        msg.linear.x = self._speed if self._hold else 0.0
        self._pub.publish(msg)

    def _handle(self, ch: str):
        if ch in ("w", "W"):
            self._speed = min(0.45, self._speed + 0.05)
            self._hold = True
            self.get_logger().info(f"目标步速 {self._speed:.2f} m/s")
        elif ch in ("s", "S"):
            self._speed = max(0.05, self._speed - 0.05)
            self._hold = True
            self.get_logger().info(f"目标步速 {self._speed:.2f} m/s")
        elif ch in ("r", "R"):
            self._hold = False
            self.get_logger().info("重置仿真")
        elif ch == "\x1b":
            raise KeyboardInterrupt
        else:
            self._hold = False

    def destroy_node(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        os.close(self._fd)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GaitTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
