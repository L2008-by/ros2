# -*- coding: utf-8 -*-
"""
肌肉骨骼人键盘遥控节点
======================

读取终端按键，发布 ``/cmd_vel``（Twist）驱动肌肉骨骼人手臂的末端目标点：

===========  =======================  ==========================
按键         动作                     指令分量
===========  =======================  ==========================
``W`` / ``S``  末端前伸 / 后收          ``linear.x = ±scale``
``A`` / ``D``  末端左移 / 右移          ``linear.y = ±scale``
``Q`` / ``E``  末端上升 / 下降          ``linear.z = ±scale``
``空格``       急停（目标点保持）        全零
``ESC``        退出                     —
松开按键                              全零
===========  =======================  ==========================

说明：``ros2 launch`` 启动的节点 stdin 指向 /dev/null（ROS 已知限制），
这里优先打开控制终端 ``/dev/tty`` 读键，没有控制终端时回退到 stdin。
"""

import os
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

#: 按键 → 世界系速度方向（x 前 / y 左 / z 上）
KEY_BINDINGS = {
    'w': (1.0, 0.0, 0.0), 's': (-1.0, 0.0, 0.0),
    'a': (0.0, 1.0, 0.0), 'd': (0.0, -1.0, 0.0),
    'q': (0.0, 0.0, 1.0), 'e': (0.0, 0.0, -1.0),
    'KEY_UP': (1.0, 0.0, 0.0), 'KEY_DOWN': (-1.0, 0.0, 0.0),
    'KEY_LEFT': (0.0, 1.0, 0.0), 'KEY_RIGHT': (0.0, -1.0, 0.0),
}

HELP_TEXT = """
============================================================
  肌肉骨骼人手臂键盘运动控制（发布 /cmd_vel）
------------------------------------------------------------
  W / S     末端前伸 / 后收
  A / D     末端左移 / 右移
  Q / E     末端上升 / 下降
  空格       急停（保持在当前目标点）
  ESC        退出     松开按键 = 停止
============================================================
"""


class KeyboardTeleopNode(Node):
    """键盘遥控节点（3 轴末端速度）。"""

    def __init__(self):
        super().__init__('keyboard_teleop_node')

        self.declare_parameter('hand_speed', 0.18)      # 末端目标点速度 (m/s)
        self.declare_parameter('publish_rate', 30.0)
        self.hand_speed = self.get_parameter('hand_speed').get_parameter_value().double_value
        rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.twist = Twist()
        self.timer = self.create_timer(1.0 / max(1.0, rate), self._publish_callback)

        self._close_fd = False
        try:
            self._fd = os.open('/dev/tty', os.O_RDONLY)
            self._close_fd = True
        except OSError:
            self._fd = sys.stdin.fileno()
        try:
            self.settings = termios.tcgetattr(self._fd)
        except termios.error:                            # pragma: no cover - 无 tty 环境
            self.settings = None

        self.get_logger().info('键盘遥控节点已启动（W/S/A/D/Q/E 控制末端，空格急停）')
        print(HELP_TEXT)

    def _get_key(self, timeout=0.03):
        """非阻塞读键（支持 ANSI 方向键转义序列）。"""
        if self.settings is None:
            return ''
        try:
            tty.setraw(self._fd)
            ready, _, _ = select.select([self._fd], [], [], timeout)
            key = ''
            if ready:
                char = os.read(self._fd, 1).decode('utf-8', errors='ignore')
                if char == '\x1b':
                    more, _, _ = select.select([self._fd], [], [], 0.03)
                    if more:
                        second = os.read(self._fd, 1).decode('utf-8', errors='ignore')
                        if second == '[':
                            third, _, _ = select.select([self._fd], [], [], 0.03)
                            if third:
                                code = os.read(self._fd, 1).decode('utf-8', errors='ignore')
                                key = {'A': 'KEY_UP', 'B': 'KEY_DOWN',
                                       'C': 'KEY_RIGHT', 'D': 'KEY_LEFT'}.get(code, '')
                else:
                    key = char
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self.settings)
            return key
        except termios.error:                            # pragma: no cover
            return ''

    def _publish_callback(self):
        key = self._get_key()
        self.twist = Twist()
        if key in KEY_BINDINGS:
            vx, vy, vz = KEY_BINDINGS[key]
            self.twist.linear.x = vx * self.hand_speed
            self.twist.linear.y = vy * self.hand_speed
            self.twist.linear.z = vz * self.hand_speed
        elif key == '\x03':                              # Ctrl-C
            self.pub_cmd.publish(self.twist)
            raise KeyboardInterrupt
        # 空格 / 松开按键 / 未知键：发布全零（目标点保持不动）
        self.pub_cmd.publish(self.twist)

    def destroy_node(self):
        if self.settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self.settings)
            except termios.error:                        # pragma: no cover
                pass
        if self._close_fd:
            try:
                os.close(self._fd)
            except OSError:                              # pragma: no cover
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
