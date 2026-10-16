# -*- coding: utf-8 -*-
"""
balance.py —— 平衡判据与抗扰恢复控制器（无 ROS 依赖）

内容：
  1. 几何工具：支撑多边形（脚底 8 角点的凸包）、点是否在凸包内、到边界的稳定裕度；
  2. LIPM 判据：ZMP（零矩点）、捕获点（Capture Point）与稳定裕度；
  3. BalanceController：踝策略（主）+ 髋策略（增强）的**关节力矩**修正。

为什么输出的是力矩而不是关节目标
--------------------------------
踝 pitch 关节的实际活动范围只有 ±0.9 rad（见 models/humanoid_balance.xml）。若用
「关节目标」表达平衡修正，为获得足够的恢复力矩就需要下发远超关节行程的目标
（例如 3 rad ≈ 172°），既不合理、又会被关节限位截断，物理含义模糊。
因此本模块的平衡控制器直接输出**关节力矩修正量**（N·m），由 sim_core 叠加到站姿
PD 力矩上；限幅也改为力矩量纲（``ankle_torque_limit`` 等），其上限由脚掌几何
与关节力矩权限决定（见文档 3.6 节）。

符号约定（由 test_standalone.py 的符号自检项实测标定）
------------------------------------------------------
  * 踝 pitch **正力矩 → 质心向后移动（−x）**；
  * 踝 roll  **正力矩 → 质心向 +y（左）移动**；
  * 髋 pitch **正力矩 → 质心向前移动（+x）**；
  * 髋 roll  **正力矩 → 质心向 −y（右）移动**。
因此把质心拉回参考位置所需的力矩符号为：踝 pitch 取正、踝 roll 取负、
髋 pitch 取负、髋 roll 取正（与下式一致）。符号写反会形成正反馈直接推倒机器人。
"""

import numpy as np

from .sim_core import JOINT_NAMES


# ----------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------
def convex_hull(points):
    """二维凸包（Andrew 单调链），返回逆时针顶点序列。"""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def point_in_polygon(point, polygon):
    """点是否在凸多边形内（含边界）。"""
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return False
    sign = 0
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def distance_to_boundary(point, polygon):
    """点到凸多边形边界的最近距离（内部为正，外部为负）。"""
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return float("nan")
    best = float("inf")
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length < 1e-12:
            continue
        # 点到线段的距离
        t = float(np.clip(np.dot(point - a, ab) / (length ** 2), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (a + t * ab))))
    return best if point_in_polygon(point, poly) else -best


# ----------------------------------------------------------------------
# LIPM 判据
# ----------------------------------------------------------------------
def capture_point(com, com_vel, height):
    """捕获点（Capture Point）水平位置：ξ = x + v/ω，ω = √(g/h)。"""
    omega = np.sqrt(9.81 / max(1e-6, height))
    return np.asarray(com[:2], dtype=float) + np.asarray(com_vel[:2], dtype=float) / omega


def stability_margin(com, com_vel, height, polygon=None, use_capture_point=True):
    """稳定裕度 (m)：**捕获点（或质心投影）到支撑多边形边界的最近距离**。

    参数:
        com, com_vel: 质心位置与速度
        height:       质心高度 (m)，用于捕获点的 ω = √(g/h)
        polygon:      支撑多边形顶点 (N×2)。为 None 时函数退化为不可用，
                      返回 NaN（调用方应显式传入 ``sim.foot_polygon()``）
        use_capture_point:
                      True  —— 用捕获点 ξ 计算裕度（动态判据：裕度 > 0 表示
                               仅靠踝/髋策略仍可停下，< 0 表示必须跨步）；
                      False —— 用质心投影计算裕度（静态判据）。

    返回值 > 0 表示仍在支撑多边形内，< 0 表示已越界。
    """
    if polygon is None or len(polygon) < 3:
        return float("nan")
    point = capture_point(com, com_vel, height) if use_capture_point \
        else np.asarray(com[:2], dtype=float)
    return distance_to_boundary(point, polygon)


# ----------------------------------------------------------------------
# 平衡控制器（输出关节力矩修正量）
# ----------------------------------------------------------------------
class BalanceController:
    """踝策略 + 髋策略的抗扰恢复控制器。

    ``compute`` 返回 15 维**关节力矩修正量**（N·m），由 ``sim_core.step()``
    叠加到站姿 PD 力矩上。
    """

    def __init__(self, params=None):
        p = params or {}
        # 踝策略（矢状面）：力矩增益 N·m/m 与 N·m·s/m
        self.kp_ankle = float(p.get("kp_ankle", 5400.0))
        self.kd_ankle = float(p.get("kd_ankle", 1440.0))
        self.ankle_torque_limit = float(p.get("ankle_torque_limit", 300.0))
        # 踝策略（侧向）
        self.kp_ankle_roll = float(p.get("kp_ankle_roll", 4800.0))
        self.kd_ankle_roll = float(p.get("kd_ankle_roll", 600.0))
        self.ankle_roll_torque_limit = float(p.get("ankle_roll_torque_limit", 180.0))
        # 髋策略
        self.hip_enable = bool(p.get("hip_enable", True))
        self.kp_hip = float(p.get("kp_hip", 480.0))
        self.kd_hip = float(p.get("kd_hip", 72.0))
        self.hip_torque_limit = float(p.get("hip_torque_limit", 54.0))
        self.kp_hip_roll = float(p.get("kp_hip_roll", 1600.0))
        self.kd_hip_roll = float(p.get("kd_hip_roll", 240.0))
        self.hip_roll_torque_limit = float(p.get("hip_roll_torque_limit", 180.0))
        # 目标质心水平位置；None 表示「以复位时的质心为参考」（推荐）
        self.com_ref = None if p.get("com_ref") is None else np.asarray(p["com_ref"], dtype=float)
        # 诊断量
        self.last = {}

    def reset(self, sim=None):
        """复位控制器状态；传入 sim 时把当前质心位置作为目标参考。"""
        self.last = {}
        if sim is not None and self.com_ref is None:
            self.com_ref = sim.com_position()[:2].copy()

    # ------------------------------------------------------------------
    def compute(self, sim):
        """按当前状态计算关节力矩修正量（N·m）。"""
        state = sim.read_state()
        com = state["com"]
        vel = state["com_vel"]
        ref = self.com_ref if self.com_ref is not None else np.zeros(2)
        ex = float(com[0] - ref[0])
        ey = float(com[1] - ref[1])
        vx = float(vel[0])
        vy = float(vel[1])

        # --- 踝策略（矢状面）：正力矩把质心向后推 ---
        ankle_tau = self.kp_ankle * ex + self.kd_ankle * vx
        ankle_tau = float(np.clip(ankle_tau,
                                  -self.ankle_torque_limit, self.ankle_torque_limit))

        # --- 踝策略（侧向）：正力矩把质心向 +y 推，故取负号 ---
        ankle_roll_tau = -(self.kp_ankle_roll * ey + self.kd_ankle_roll * vy)
        ankle_roll_tau = float(np.clip(ankle_roll_tau,
                                       -self.ankle_roll_torque_limit,
                                       self.ankle_roll_torque_limit))

        # --- 髋策略：踝饱和后靠上半身角动量继续把质心拉回 ---
        # 正髋 pitch 力矩把质心向前推，正髋 roll 力矩把质心向 −y 推
        hip_tau = hip_roll_tau = 0.0
        if self.hip_enable:
            hip_tau = float(np.clip(-(self.kp_hip * ex + self.kd_hip * vx),
                                    -self.hip_torque_limit, self.hip_torque_limit))
            hip_roll_tau = float(np.clip(self.kp_hip_roll * ey + self.kd_hip_roll * vy,
                                         -self.hip_roll_torque_limit,
                                         self.hip_roll_torque_limit))

        torque = np.zeros(len(JOINT_NAMES))
        for name, value in (("ankle_pitch_l", ankle_tau), ("ankle_pitch_r", ankle_tau),
                            ("ankle_roll_l", ankle_roll_tau),
                            ("ankle_roll_r", ankle_roll_tau),
                            ("hip_pitch_l", hip_tau), ("hip_pitch_r", hip_tau),
                            ("hip_roll_l", hip_roll_tau), ("hip_roll_r", hip_roll_tau)):
            torque[JOINT_NAMES.index(name)] = value

        self.last = {
            "ex": ex, "ey": ey, "vx": vx, "vy": vy,
            "ankle_tau": ankle_tau, "ankle_roll_tau": ankle_roll_tau,
            "hip_tau": hip_tau, "hip_roll_tau": hip_roll_tau,
            "ankle_saturated": abs(ankle_tau) >= self.ankle_torque_limit - 1e-9
                               or abs(ankle_roll_tau) >= self.ankle_roll_torque_limit - 1e-9,
            "zmp": state["zmp"], "tilt": state["tilt"],
        }
        return torque
