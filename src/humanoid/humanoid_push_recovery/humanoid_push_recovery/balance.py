# -*- coding: utf-8 -*-
"""
balance.py —— 平衡判据与抗扰恢复控制器（无 ROS 依赖）

内容：
  1. 几何工具：支撑多边形（脚底 8 角点的凸包）、点是否在凸包内、到边界的稳定裕度；
  2. LIPM 判据：ZMP（零矩点）与捕获点（Capture Point）计算；
  3. BalanceController：踝策略（主）+ 髋策略（增强）的关节目标修正。

控制律说明（矢状面 x 方向为例）
--------------------------------
实测符号约定（见 test_standalone.py 的符号自检项）：踝关节目标增大 → 质心向后移动。
因此「质心偏前（x>0）→ 需要把质心拉回」对应踝目标增大，控制律取

    ankle_pitch_cmd = kp * (x_com - x_ref) + kd * vx_com

并以 ankle_limit 限幅（踝关节力矩权限受脚掌尺寸物理限制）。
侧向（y 方向）roll 关节同构。

髋策略通过摆动上半身产生角动量，在踝策略饱和后提供额外的恢复能力；
是否启用由参数 ``hip_enable`` 控制，便于做「踝策略 vs 踝+髋策略」的对照实验。
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
    """捕获点（Capture Point）水平位置：ξ = x + v·√(h/g)。"""
    omega = np.sqrt(9.81 / max(1e-6, height))
    return np.asarray(com[:2], dtype=float) + np.asarray(com_vel[:2], dtype=float) / omega


def stability_margin(zmp, com, com_vel, height):
    """基于捕获点的稳定裕度 (m)：捕获点到支撑多边形边界的距离。

    裕度 > 0 表示「以当前速度，仅靠踝/髋策略仍可停下」；< 0 表示必须跨步。
    """
    xi = capture_point(com, com_vel, height)
    return xi


# ----------------------------------------------------------------------
# 平衡控制器
# ----------------------------------------------------------------------
class BalanceController:
    """踝策略 + 髋策略的抗扰恢复控制器。

    每步输出 15 个关节的目标角度（直立位 + 平衡修正），交由 sim_core 的
    站姿 PD 转成关节力矩。
    """

    def __init__(self, params=None):
        p = params or {}
        # 踝策略增益（矢状面 / 侧向）
        self.kp_ankle = float(p.get("kp_ankle", 3.0))
        self.kd_ankle = float(p.get("kd_ankle", 0.35))
        self.kp_ankle_roll = float(p.get("kp_ankle_roll", 8.0))
        self.kd_ankle_roll = float(p.get("kd_ankle_roll", 1.0))
        self.ankle_limit = float(p.get("ankle_limit", 0.35))
        #: 侧向踝限幅单列：roll 关节力矩权限大，过大指令会把机器人「抛起来」，需单独限幅
        self.ankle_roll_limit = float(p.get("ankle_roll_limit", 0.20))
        # 髋策略增益（矢状面 / 侧向）
        self.hip_enable = bool(p.get("hip_enable", True))
        self.kp_hip = float(p.get("kp_hip", 4.0))
        self.kd_hip = float(p.get("kd_hip", 0.6))
        self.kp_hip_roll = float(p.get("kp_hip_roll", 4.0))
        self.kd_hip_roll = float(p.get("kd_hip_roll", 0.6))
        self.hip_limit = float(p.get("hip_limit", 0.45))
        # 目标质心水平位置；``com_ref=None`` 表示「以复位时的质心为参考」（推荐）
        self.com_ref = None if p.get("com_ref") is None else np.asarray(p["com_ref"], dtype=float)
        # 诊断量
        self.last = {}
        self.history = []

    def reset(self, sim=None):
        """复位控制器状态；传入 sim 时把当前质心位置作为目标参考。"""
        self.last = {}
        self.history = []
        if sim is not None and self.com_ref is None:
            self.com_ref = sim.com_position()[:2].copy()

    # ------------------------------------------------------------------
    def compute(self, sim):
        """按当前状态计算关节目标向量。"""
        state = sim.read_state()
        com = state["com"]
        vel = state["com_vel"]
        ref = self.com_ref if self.com_ref is not None else np.zeros(2)
        # 误差：以「质心相对目标位置的偏移」为反馈量
        ex = float(com[0] - ref[0])
        ey = float(com[1] - ref[1])
        vx = float(vel[0])
        vy = float(vel[1])

        # --- 踝策略（矢状面）：实测踝关节目标增大 → 质心向后移动 ---
        ankle = self.kp_ankle * ex + self.kd_ankle * vx
        ankle = float(np.clip(ankle, -self.ankle_limit, self.ankle_limit))

        # --- 踝策略（侧向）：实测踝 roll 目标增大 → 质心向 +y 移动，故取负号 ---
        ankle_roll = -(self.kp_ankle_roll * ey + self.kd_ankle_roll * vy)
        ankle_roll = float(np.clip(ankle_roll, -self.ankle_roll_limit, self.ankle_roll_limit))

        # --- 髋策略（踝饱和后靠上半身角动量继续拉回）---
        # 实测：髋 pitch 目标增大 → 质心向 +x 移动；髋 roll 目标增大 → 质心向 −y 移动
        hip = hip_roll = 0.0
        if self.hip_enable:
            hip = float(np.clip(-self.kp_hip * ex - self.kd_hip * vx,
                                -self.hip_limit, self.hip_limit))
            hip_roll = float(np.clip(self.kp_hip_roll * ey + self.kd_hip_roll * vy,
                                     -self.hip_limit, self.hip_limit))

        target = np.zeros(len(JOINT_NAMES))
        for name, value in (("ankle_pitch_l", ankle), ("ankle_pitch_r", ankle),
                            ("ankle_roll_l", ankle_roll), ("ankle_roll_r", ankle_roll),
                            ("hip_pitch_l", hip), ("hip_pitch_r", hip),
                            ("hip_roll_l", hip_roll), ("hip_roll_r", hip_roll)):
            target[JOINT_NAMES.index(name)] = value

        self.last = {
            "ex": ex, "ey": ey, "vx": vx, "vy": vy,
            "ankle_cmd": ankle, "ankle_roll_cmd": ankle_roll,
            "hip_cmd": hip, "hip_roll_cmd": hip_roll,
            "ankle_saturated": abs(ankle) >= self.ankle_limit - 1e-9
                               or abs(ankle_roll) >= self.ankle_roll_limit - 1e-9,
            "zmp": state["zmp"], "tilt": state["tilt"],
        }
        return target
