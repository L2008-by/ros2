# -*- coding: utf-8 -*-
"""
leg_ik.py —— 腿部逆运动学（平面 2D 三连杆解析解 + 正向校验）
================================================================

针对模型 ``humanoid_gait.xml`` 的腿部拓扑（大腿 0.34 m、小腿 0.30 m，
髋 pitch / 膝 pitch / 踝 pitch 均绕 y 轴），**全部关节角使用与模型
MuJoCo 定义一致的符号**：

* 髋 pitch ``θh``：绕 y 轴右手定则，**正值 = 大腿向后摆**（腿部末端向 -x）；
* 膝 pitch ``θk``：**正值 = 膝向后弯**（人类式屈膝，模型限位 [-2.5, 1.5]）；
* 踝 pitch ``θa``：补偿使脚底水平（脚向量世界方向角保持模型默认 0.503）。

在 x-z 平面（x 前进、z 向上）内，给定髋位置与踝目标位置，解析求解三个
关节角。侧向（髋 roll / 踝 roll）单独由 3D 入口给出，保持骨盆水平、
脚底着地。

正解（用于测试往返误差）::

    ankle_x = hip_x - sin(θh)·L1 - sin(θh+θk)·L2
    ankle_z = hip_z - cos(θh)·L1 - cos(θh+θk)·L2
"""

from __future__ import annotations

import math

import numpy as np

#: 大腿/小腿长度 (m)——髋→膝 0.52（thigh 偏移 0.18 + 几何 0.34）、
#: 膝→踝 0.46（shin 偏移 0.16 + 几何 0.30），与 humanoid_gait.xml 一致
THIGH_LEN = 0.52
SHIN_LEN = 0.46


def leg_ik_2d(hip: np.ndarray, ankle: np.ndarray, foot_angle: float = 0.0):
    """平面三连杆腿部 IK（模型符号）。

    参数
    ----
    hip:    髋关节位置 (x, z)
    ankle:  踝关节目标位置 (x, z)
    foot_angle: 期望脚底相对水平面的角度（脚尖上翘为正，默认 0 = 水平）

    返回
    ----
    (hip_pitch, knee_pitch, ankle_pitch) 或 None（目标不可达）
    """
    dx = float(ankle[0] - hip[0])
    dz = float(ankle[1] - hip[1])
    r = math.hypot(dx, dz)

    l1, l2 = THIGH_LEN, SHIN_LEN
    if r < 1e-6 or r > l1 + l2 + 1e-6 or r < abs(l1 - l2) - 1e-6:
        return None

    # 目标方向角 γ：相对「向下」(-z) 轴，向前为正
    gamma = math.atan2(dx, -dz)

    # 膝角：余弦定理，取「后弯」解（人类式屈膝，模型限位 [-2.5, 1.5]）
    cos_k = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_k = min(max(cos_k, -1.0), 1.0)
    knee = math.acos(cos_k)                     # ∈ [0, π]，正 = 后弯

    # 髋角（模型符号）：大腿前倾角 φ = γ + β（前倾构型），
    # 模型 hip_pitch = -φ（模型正值 = 向后摆）
    sin_k = math.sin(knee)
    beta = math.atan2(l2 * sin_k, l1 + l2 * math.cos(knee))
    phi = gamma + beta
    hip_pitch = -phi

    # 踝角：补偿使脚底水平（模型踝角 = -(hip_pitch + knee)）
    ankle_pitch = -(hip_pitch + knee) + foot_angle
    return (hip_pitch, knee, ankle_pitch)


def forward_leg_2d(hip: np.ndarray, hip_pitch: float, knee: float,
                   ankle_pitch: float) -> tuple:
    """正解（模型符号）：由关节角求踝位置与脚底方向角（测试往返误差用）。"""
    l1, l2 = THIGH_LEN, SHIN_LEN
    c1, s1 = math.cos(hip_pitch), math.sin(hip_pitch)
    c12, s12 = math.cos(hip_pitch + knee), math.sin(hip_pitch + knee)
    ankle = np.array([
        hip[0] - s1 * l1 - s12 * l2,
        hip[1] - c1 * l1 - c12 * l2,
    ])
    foot_dir = hip_pitch + knee + ankle_pitch       # 相对 -z 轴，0 = 水平
    return ankle, foot_dir


def leg_ik_3d(hip: np.ndarray, ankle: np.ndarray, foot_yaw_lean: float = 0.0):
    """3D 腿部 IK：在 2D 平面 IK 基础上叠加侧向髋/踝 roll。

    参数
    ----
    hip:    髋关节位置 (x, y, z)
    ankle:  踝关节目标位置 (x, y, z)
    foot_yaw_lean: 脚底侧倾目标（默认 0 = 水平）

    返回
    ----
    (hip_pitch, knee, ankle_pitch, hip_roll, ankle_roll) 或 None
    """
    dx = float(ankle[0] - hip[0])
    dz = float(ankle[2] - hip[2])
    dy = float(ankle[1] - hip[1])
    planar = leg_ik_2d(np.array([hip[0], hip[2]]), np.array([ankle[0], ankle[2]]), 0.0)
    if planar is None:
        return None
    hip_pitch, knee, ankle_pitch = planar

    # 侧向：髋 roll 把骨盆摆到脚上方（腿竖直长度近似），踝 roll 补偿脚底水平
    leg_len = abs(dz) if abs(dz) > 1e-3 else THIGH_LEN + SHIN_LEN
    hip_roll = math.atan2(dy, leg_len) if abs(dy) > 1e-4 else 0.0
    ankle_roll = -hip_roll + foot_yaw_lean
    return (hip_pitch, knee, ankle_pitch, hip_roll, ankle_roll)


def clamp_joint(q, lo, hi):
    """关节角限幅。"""
    return min(max(q, lo), hi)
