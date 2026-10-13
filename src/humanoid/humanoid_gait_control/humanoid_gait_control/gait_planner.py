# -*- coding: utf-8 -*-
"""
gait_planner.py —— 双足步态生成（LIPM 线性倒立摆 + 落脚点序列 + 摆动腿轨迹）
================================================================================

职责
----
在 MuJoCo 之外生成「可解释、可复算」的双足行走参考轨迹，供关节 PD 伺服跟踪：

1. **LIPM 质心轨迹**（Linear Inverted Pendulum Model）：支撑脚为支点，
   质心高度恒定 ``h``，前向动力学为 ::

        x'' = (g / h) · x            (x 为质心相对支点的水平位移)

   单步周期对称边界条件（步长 ``s``、单步时长 ``T``、时间常数 ``Tc=sqrt(h/g)``）：:

        α = T / Tc
        x0 = -s / 2                                   （步初质心相对支点）
        v0 = s · sinh(α) / (2 · Tc · (cosh(α) - 1))    （步初质心速度）

   于是质心每步从 ``-s/2`` 走到 ``+s/2``，速度回到 ``v0``，形成周期步态，
   平均速度 = ``s / T``。

2. **落脚点序列**：左右脚沿前进方向交替，横向间距 ``step_width``。

3. **摆动腿轨迹**：三次 Hermite 插值（起止速度为零）+ 抛物线抬脚，
   脚跟着地/抬起瞬间无冲击：:

        x(t) = p0 + (p1 - p0) · (3τ² - 2τ³)          τ = t / T_swing
        z(t) = z_base + 4 · h_swing · τ(1 - τ)

4. **相位状态机**：准备段（双脚站立，重心前移）→ 周期段（左右交替单脚支撑）。

本文件不依赖 MuJoCo，纯 numpy 实现，便于单元测试与指标复算。
"""

from __future__ import annotations

import math

import numpy as np

#: 重力加速度 (m/s²)
G = 9.81


def _cosh(x: float) -> float:
    return math.cosh(x)


def _sinh(x: float) -> float:
    return math.sinh(x)


class LIPM:
    """线性倒立摆模型：给定初始相对位移/速度，闭式积分质心轨迹。"""

    def __init__(self, com_height: float, g: float = G):
        self.h = float(com_height)
        self.g = float(g)
        self.Tc = math.sqrt(self.h / self.g)

    # ------------------------------------------------------------------
    # 闭式解
    # ------------------------------------------------------------------
    def position(self, x0: float, v0: float, t: float) -> float:
        """质心相对支点的水平位移 x(t)。"""
        a = t / self.Tc
        return x0 * _cosh(a) + self.Tc * v0 * _sinh(a)

    def velocity(self, x0: float, v0: float, t: float) -> float:
        """质心相对支点的水平速度 v(t)。"""
        a = t / self.Tc
        return x0 / self.Tc * _sinh(a) + v0 * _cosh(a)

    def trajectory(self, x0: float, v0: float, t_end: float, n: int) -> tuple:
        """返回 (t 序列, x 序列, v 序列)。"""
        ts = np.linspace(0.0, t_end, max(2, n))
        xs = np.array([self.position(x0, v0, t) for t in ts])
        vs = np.array([self.velocity(x0, v0, t) for t in ts])
        return ts, xs, vs

    # ------------------------------------------------------------------
    # 周期步态边界条件
    # ------------------------------------------------------------------
    def cycle_state(self, step_len: float, period: float) -> tuple:
        """周期对称单步的初态 (x0, v0)（相对当前支撑脚）。

        要求：``x(T)=x0+s``（前进一个步长）且 ``v(T)=v0``（速度周期对称）。
        """
        alpha = period / self.Tc
        x0 = -step_len / 2.0
        v0 = step_len * _sinh(alpha) / (2.0 * self.Tc * (_cosh(alpha) - 1.0))
        return x0, v0


def _hermite_step(p0: np.ndarray, p1: np.ndarray, tau: float) -> np.ndarray:
    """三次 Hermite 插值（起止速度为零）。"""
    u = 3.0 * tau * tau - 2.0 * tau * tau * tau
    return p0 + (p1 - p0) * u


def _swing_height(tau: float, h_swing: float) -> float:
    """抛物线抬脚曲线（带双脚支撑期）：步初/步末 20% 摆动脚着地，
    中段 60% 抬起（τ=0.5 时最大 h_swing），换脚前后双脚同时支撑。
    """
    if tau < 0.2 or tau > 0.8:
        return 0.0
    t = (tau - 0.2) / 0.6          # 0→1 归一化到抬脚段
    return 4.0 * h_swing * t * (1.0 - t)


class GaitPlanner:
    """双足周期步态规划器（直线前进，质心高度恒定）。

    参数
    ----
    step_len:    单步前进距离 (m)
    step_period: 单步时长 (s)
    step_width:  左右脚横向间距 (m)
    swing_height: 摆动脚抬离地面最大高度 (m)
    com_height:  质心高度 (m)
    n_steps:     计划行走步数
    prepare_time: 准备段时长（双脚站立、重心前移）
    start_foot:  'left' 起始支撑脚（第一步摆动另一只脚）
    """

    def __init__(self, step_len: float = 0.10, step_period: float = 1.2,
                 step_width: float = 0.05, swing_height: float = 0.05,
                 com_height: float = 1.05, n_steps: int = 12,
                 prepare_time: float = 0.4, start_foot: str = "left"):
        self.step_len = float(step_len)
        self.step_period = float(step_period)
        self.step_width = float(step_width)
        self.swing_height = float(swing_height)
        self.com_height = float(com_height)
        self.n_steps = int(n_steps)
        self.prepare_time = float(prepare_time)
        self.start_foot = start_foot

        self.lipm = LIPM(com_height)
        self.Tc = self.lipm.Tc
        self.total_time = self.prepare_time + self.n_steps * self.step_period

        # 周期步态初态（相对支点）
        self._x0, self._v0 = self.lipm.cycle_state(step_len, step_period)
        # 单步终点质心相对支点 = +s/2（hold 段平滑收回用）
        self._x0_swing_end = step_len / 2.0

        # 落脚点序列：共 n_steps+1 个落点（第 0 个为初始双脚位置）
        self._footprints = self._plan_footprints()

    # ------------------------------------------------------------------
    # 落脚点
    # ------------------------------------------------------------------
    def _plan_footprints(self) -> list:
        """左右交替落脚点（世界系 x 前进、y 横向）。"""
        # 起始脚：准备段结束时支撑脚站于 (0, +w/2)（start_foot=left 时左脚在 y 正侧，
        # 与模型 hip_l 的 y=+0.16 对齐）
        w = self.step_width
        left_y, right_y = w / 2.0, -w / 2.0
        pts = []
        support_y = left_y if self.start_foot == "left" else right_y
        # 初始（准备段）双脚并立：支撑脚在 (0, support_y)，摆动脚在 (0, -support_y)
        pts.append((0.0, support_y))
        for k in range(1, self.n_steps + 1):
            x = k * self.step_len
            y = support_y if k % 2 == 0 else -support_y
            pts.append((x, y))
        return [np.array(p, dtype=float) for p in pts]

    def footprints(self) -> list:
        return list(self._footprints)

    # ------------------------------------------------------------------
    # 相位
    # ------------------------------------------------------------------
    def phase(self, t: float) -> tuple:
        """返回 (段类型, 步索引, 段内时间 tau)。段类型: 'prepare' / 'step' / 'hold'。"""
        t = float(t)
        if t < self.prepare_time:
            return "prepare", -1, t / max(self.prepare_time, 1e-9)
        tc = t - self.prepare_time
        idx = int(tc // self.step_period)
        if idx >= self.n_steps:
            # 计划结束后保持最后姿态（质心平滑收回支撑脚上方，脚着地）
            th = (tc - self.n_steps * self.step_period) / max(self.step_period, 1e-9)
            return "hold", self.n_steps - 1, min(max(th, 0.0), 1.0)
        tau = (tc - idx * self.step_period) / self.step_period
        return "step", idx, tau

    def support_foot(self, t: float) -> np.ndarray:
        """当前支撑脚位置（世界系）。"""
        kind, idx, _ = self.phase(t)
        if kind == "prepare":
            return self._footprints[0]
        if kind == "hold":
            return self._footprints[max(0, idx)]
        return self._footprints[idx]

    def swing_foot_target(self, t: float) -> np.ndarray:
        """摆动脚目标位置（世界系，含抬脚高度）。"""
        kind, idx, tau = self.phase(t)
        if kind == "prepare" or (kind == "hold" and self.n_steps == 0):
            # 准备段/纯站立：双脚并立，摆动脚停在起始位置（与支撑脚横向对称）
            return np.array([0.0, -self._footprints[0][1], 0.0], dtype=float)
        if kind == "hold":
            # 计划结束：摆动脚停在最后一落点（着地）
            return np.array([self._footprints[-1][0], self._footprints[-1][1], 0.0],
                            dtype=float)
        # 摆动脚起点：第 0 步从初始并立位置起摆，之后接上一落点 footprints[idx-1]
        p0 = np.array([0.0, -self._footprints[0][1]], dtype=float) if idx == 0 \
            else self._footprints[idx - 1]
        p1 = self._footprints[idx + 1]       # 摆动脚下一落点
        pos = _hermite_step(p0, p1, tau)
        # 抬脚高度：抛物线拱（起落 z=0，中段最大 swing_height）
        z = _swing_height(tau, self.swing_height)
        return np.array([pos[0], pos[1], z], dtype=float)

    # ------------------------------------------------------------------
    # 质心轨迹
    # ------------------------------------------------------------------
    def com_position(self, t: float) -> np.ndarray:
        """质心位置（世界系，3D）。"""
        kind, idx, tau = self.phase(t)
        if kind == "prepare":
            # 准备段：质心稳定停在双脚中点（静态平衡点，腿微屈），
            # 不做侧移——LIPM 的 -s/2 起点是动态步态起点而非静态平衡点
            return np.array([0.0, 0.0, self.com_height], dtype=float)
        if kind == "hold":
            if self.n_steps == 0:
                # 纯站立（无步态）：质心停在双脚中点
                return np.array([0.0, 0.0, self.com_height], dtype=float)
            # 计划结束：质心从最后支撑脚前方 +s/2 平滑收回支撑脚上方（静态平衡）
            foot = self._footprints[max(0, idx)]
            u = tau * tau * (3.0 - 2.0 * tau)          # 0→1 平滑
            x_rel = self._x0_swing_end * (1.0 - u)
            return np.array([foot[0] + x_rel, 0.0, self.com_height], dtype=float)
        # 步进段：相对当前支撑脚做 LIPM
        foot = self._footprints[idx]
        if idx == 0:
            # 起步：质心从准备段终点（0）用 S 曲线均匀前进到 +s/2，
            # 与步末双支撑期衔接（摆动脚已落地），避免 LIPM 起点 -s/2
            # 造成的"先退后冲"
            u = tau * tau * (3.0 - 2.0 * tau)
            x_rel = self._x0_swing_end * u
        else:
            x_rel = self.lipm.position(self._x0, self._v0, tau * self.step_period)
        y_rel = self._lipm_lateral(tau, idx)
        return np.array([foot[0] + x_rel, y_rel, self.com_height], dtype=float)

    def _lipm_lateral(self, tau: float, idx: int) -> float:
        """侧向质心 y：配合双脚支撑期做分段过渡。

        步初 20%（换脚后双脚支撑）：质心从两脚中间平滑移到新支撑脚上方；
        步中 60%（单脚支撑）：质心恒在支撑脚正上方（侧向倒立摆平衡点）；
        步末 20%（摆动脚提前落地，双脚支撑）：质心平滑回到两脚中间，
        为下一步换脚做准备。
        """
        foot_y = float(self._footprints[idx][1])
        prev_y = 0.0 if idx == 0 else float(self._footprints[idx - 1][1])
        if tau < 0.2:
            u = (tau / 0.2) ** 2 * (3.0 - 2.0 * (tau / 0.2))
            return prev_y + (foot_y - prev_y) * u
        if tau > 0.8:
            u = ((tau - 0.8) / 0.2) ** 2 * (3.0 - 2.0 * ((tau - 0.8) / 0.2))
            return foot_y * (1.0 - u)
        return foot_y

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def com_velocity(self, t: float) -> np.ndarray:
        """质心速度（世界系，x/y 分量，供速度前馈使用）。"""
        kind, idx, tau = self.phase(t)
        if kind in ("prepare", "hold") or self.n_steps == 0:
            return np.zeros(3, dtype=float)
        foot = self._footprints[idx]
        if idx == 0:
            # d/dt[ s/2·u(τ) ] = s/2·6τ(1-τ)/T
            u_dot = 6.0 * tau * (1.0 - tau) / self.step_period
            vx = self._x0_swing_end * u_dot
            vy = 0.0
        else:
            vx = self.lipm.velocity(self._x0, self._v0, tau * self.step_period)
            vy = 0.0
        return np.array([float(vx), float(vy), 0.0], dtype=float)

    def pose(self, t: float) -> dict:
        """一步规划快照：段、支撑脚、摆动脚、质心、相位。"""
        kind, idx, tau = self.phase(t)
        return {
            "phase": kind,
            "step_idx": idx,
            "tau": tau,
            "support_foot": self.support_foot(t),
            "swing_foot": self.swing_foot_target(t),
            "com": self.com_position(t),
            "com_vel": self.com_velocity(t),
            "t": float(t),
        }

    # ------------------------------------------------------------------
    # 指标（供文档与测试引用）
    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        """步态参数与理论速度/步长。"""
        return {
            "step_len_m": self.step_len,
            "step_period_s": self.step_period,
            "step_width_m": self.step_width,
            "swing_height_m": self.swing_height,
            "com_height_m": self.com_height,
            "n_steps": self.n_steps,
            "avg_speed_mps": self.step_len / self.step_period,
            "total_distance_m": self.n_steps * self.step_len,
            "total_time_s": self.total_time,
            "lipm_time_constant_s": self.Tc,
            "lipm_x0_m": self._x0,
            "lipm_v0_mps": self._v0,
        }
