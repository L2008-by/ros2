"""
人形机器人 MuJoCo 仿真节点
========================
核心 ROS2 节点：加载自建人形 MJCF 模型，按 /cmd_vel 速度指令推进物理仿真，
并发布里程计 / 关节状态 / TF。

控制策略（详见 sim_core.py）：
  * 前进/后退 —— 速度级控制（覆盖底盘 freejoint 世界系线速度）；
  * 转向       —— 姿态级控制（直接旋转底盘四元数），避免双足原地旋转时
                  脚底摩擦滑移导致的侧倾失稳；
  * 站姿       —— 全关节 PD 伺服回直立位（髋/踝 roll 增益单独加强）。
"""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

try:
    from humanoid_keyboard_control.sim_core import HumanoidSim
except ImportError:
    from sim_core import HumanoidSim


class MujocoSimNode(Node):
    """人形机器人 MuJoCo 仿真节点"""

    def __init__(self):
        super().__init__('mujoco_sim_node')

        # ========== 参数 ==========
        self.declare_parameter('model_path', '')
        self.declare_parameter('sim_rate', 500.0)        # 物理步进频率 (Hz)
        self.declare_parameter('publish_rate', 30.0)     # 话题发布频率 (Hz)
        self.declare_parameter('kp', 120.0)
        self.declare_parameter('kv', 6.0)
        self.declare_parameter('kp_roll', 400.0)
        self.declare_parameter('v_max', 0.3)
        self.declare_parameter('w_max', 0.6)
        self.declare_parameter('cmd_timeout', 0.5)       # 指令超时停车 (s)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        sim_rate = self.get_parameter('sim_rate').get_parameter_value().double_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.cmd_timeout = self.get_parameter('cmd_timeout').get_parameter_value().double_value

        # ========== 仿真核心 ==========
        self.sim = HumanoidSim(model_path=model_path or None)
        self.sim.KP = self.get_parameter('kp').get_parameter_value().double_value
        self.sim.KV = self.get_parameter('kv').get_parameter_value().double_value
        self.sim.KP_ROLL = self.get_parameter('kp_roll').get_parameter_value().double_value
        self.sim.V_MAX = self.get_parameter('v_max').get_parameter_value().double_value
        self.sim.W_MAX = self.get_parameter('w_max').get_parameter_value().double_value

        # ========== 控制指令缓存 ==========
        self.cmd_vel = Twist()
        self.last_cmd_time = None

        # ========== 订阅者 ==========
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10)

        # ========== 发布者 ==========
        self.pub_odom = self.create_publisher(Odometry, '/humanoid/odom', 10)
        self.pub_joint_states = self.create_publisher(JointState, '/joint_states', 10)
        # TF 通过 tf2_ros.TransformBroadcaster 发布（内部封装 tf2_msgs/TFMessage）
        self.tf_broadcaster = TransformBroadcaster(self)

        # ========== 定时器 ==========
        sim_period = 1.0 / sim_rate
        publish_period = 1.0 / publish_rate
        self.steps_per_publish = max(1, int(sim_rate / publish_rate))
        self.step_count = 0

        self.sim_timer = self.create_timer(sim_period, self._sim_step_callback)
        self.pub_timer = self.create_timer(publish_period, self._publish_callback)

        self.get_logger().info(
            f'人形机器人 MuJoCo 仿真节点已启动 | 物理频率: {sim_rate:.0f}Hz | '
            f'发布频率: {publish_rate:.0f}Hz | 模型: {model_path or "(包内默认)"}'
        )

    # ------------------------------------------------------------------
    def _cmd_vel_callback(self, msg: Twist):
        """接收速度指令（键盘遥操作 / 其他控制器下发），并刷新指令时间戳。"""
        self.cmd_vel = msg
        self.last_cmd_time = self.get_clock().now()

    def _sim_step_callback(self):
        """物理步进：站姿 PD + 运动指令 → mj_step。

        指令超时停车：若超过 cmd_timeout 未收到新 /cmd_vel，将速度清零，
        避免遥控节点停止后仿真继续执行最后一条非零指令。
        """
        if self.last_cmd_time is not None:
            age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
            if age > self.cmd_timeout:
                self.cmd_vel = Twist()
        self.sim.step(v_lin=self.cmd_vel.linear.x, w_yaw=self.cmd_vel.angular.z)
        self.step_count += 1

    def _publish_callback(self):
        """发布里程计 / 关节状态 / TF。"""
        now = self.get_clock().now().to_msg()
        od = self.sim.read_odom()
        quat = self.sim.data.qpos[self.sim.root_qadr + 3: self.sim.root_qadr + 7]

        # --- 里程计 ---
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'humanoid/base_link'
        odom.pose.pose.position.x = od['x']
        odom.pose.pose.position.y = od['y']
        odom.pose.pose.position.z = od['z']
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        odom.twist.twist.linear.x = od['vx']
        odom.twist.twist.linear.y = od['vy']
        odom.twist.twist.angular.z = od['wz']
        self.pub_odom.publish(odom)

        # --- TF: world -> humanoid/base_link（标准 TransformBroadcaster） ---
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'humanoid/base_link'
        tf_msg.transform.translation.x = od['x']
        tf_msg.transform.translation.y = od['y']
        tf_msg.transform.translation.z = od['z']
        tf_msg.transform.rotation.w = float(quat[0])
        tf_msg.transform.rotation.x = float(quat[1])
        tf_msg.transform.rotation.y = float(quat[2])
        tf_msg.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(tf_msg)

        # --- 关节状态 ---
        jp = self.sim.read_joint_positions()
        jstate = JointState()
        jstate.header.stamp = now
        jstate.name = list(jp.keys())
        jstate.position = list(jp.values())
        self.pub_joint_states.publish(jstate)

        # --- 状态打印（1Hz） ---
        if self.step_count % (self.steps_per_publish * 30) == 0:
            self.get_logger().info(
                f'pos=({od["x"]:.2f}, {od["y"]:.2f}, {od["z"]:.2f}) '
                f'yaw={np.degrees(od["yaw"]):6.1f}° '
                f'vx={od["vx"]:+.2f} m/s 站立={self.sim.standing()}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
