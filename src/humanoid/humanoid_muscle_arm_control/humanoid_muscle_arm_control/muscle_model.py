# -*- coding: utf-8 -*-
"""
muscle_model.py —— Hill 型肌肉模型（激活动力学 + 力-长度-速度关系）
==================================================================

职责
----
在 MuJoCo 之外显式计算每块肌肉的「激活 → 力」，让模型的行为可读、可标定：

1. **激活动力学**（Winters/Zajac 一阶模型）::

       da/dt = (u - a) · (1/τ_act) · (0.5 + 1.5a),   u ≥ a   （激活，快）
       da/dt = (u - a) / τ_deact,                    u < a   （放松，慢）

   其中 u ∈ [0, 1] 为神经激励（控制量），a ∈ [0, 1] 为肌肉激活度。

2. **力-长度关系**（高斯型主动曲线 + 指数型被动曲线）::

       f_L(L̃) = exp( -((L̃ - 1) / width)² )                      (主动)
       f_PE(L̃) = passive · (exp(k·(L̃ - 1)) - 1),  L̃ > 1         (被动)

3. **力-速度关系**（向心收缩 Hill 双曲线，离心收缩经验增强）::

       ṽ = v / v_max
       f_V = (1 + ṽ) / (1 - ṽ / 0.25),      ṽ < 0   （缩短：力下降，ṽ = -1 时归零）
       f_V = 1 + 0.4·(1 - exp(-3·ṽ)),       ṽ ≥ 0   （拉长：离心增强，饱和到 1.4）

4. **总力**::

       F = F_max · ( a · f_L · f_V + f_PE )        （限幅到 [0, 1.5·F_max]）

5. **反解**：给定期望肌力（控制器算出来的），反解对应的激励 u，用于控制分配。

单位：长度 m，速度 m/s，力 N，时间 s。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MuscleSpec:
    """单块肌肉的力学参数。"""

    name: str
    f_max: float                      # 最大等长收缩力 (N)
    l_opt: float = 0.25               # 最佳纤维长度 (m)，启动时按标称姿态腱长标定
    v_max_factor: float = 10.0        # v_max = v_max_factor · l_opt (1/s)
    width: float = 1.10               # 主动力-长度曲线宽度（本模型肌肉跨距大，需要较宽的平台）
    passive_scale: float = 0.015      # 被动力幅值系数（相对 F_max）
    passive_k: float = 4.0            # 被动力-长度曲线指数
    passive_max: float = 0.08         # 被动力上限（相对 F_max，防止指数项爆炸）
    eccentric_gain: float = 0.4       # 离心增强上限（1 + gain）
    hill_k: float = 0.25              # 向心 Hill 双曲线形状系数

    @property
    def v_max(self) -> float:
        return self.v_max_factor * self.l_opt


class HillMuscleModel:
    """一组肌肉的激活/力计算（与 MuJoCo 解耦，便于单元测试）。"""

    def __init__(self, specs, tau_act: float = 0.015, tau_deact: float = 0.060,
                 activation_min: float = 0.005, force_max_factor: float = 1.5):
        self.specs = list(specs)
        self.tau_act = float(tau_act)
        self.tau_deact = float(tau_deact)
        self.activation_min = float(activation_min)
        self.force_max_factor = float(force_max_factor)
        self.n = len(self.specs)
        self.activation = [0.0] * self.n          # 当前激活度 a
        self.excitation = [0.0] * self.n          # 当前激励 u
        self.force = [0.0] * self.n               # 当前肌力 (N)

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def calibrate_lengths(self, nominal_lengths) -> None:
        """按标称姿态下的肌肉长度设定 l_opt（使标称姿态处于 f_L 峰值的下降侧）。

        数值：``l_opt = nominal / 1.05``，即标称姿态下 L̃ ≈ 1.05，
        略偏长（肌力略低于峰值），转向时的长度变化不会立刻跌破峰值的一侧。
        """
        for spec, name in zip(self.specs, [s.name for s in self.specs]):
            length = float(nominal_lengths[name])
            spec.l_opt = max(0.05, length / 1.05)

    def reset(self) -> None:
        self.activation = [0.0] * self.n
        self.excitation = [0.0] * self.n
        self.force = [0.0] * self.n

    # ------------------------------------------------------------------
    # 静态曲线
    # ------------------------------------------------------------------
    def length_factor(self, i: int, length: float) -> float:
        spec = self.specs[i]
        l_tilde = length / spec.l_opt
        return math.exp(-((l_tilde - 1.0) / spec.width) ** 2)

    def velocity_factor(self, i: int, velocity: float) -> float:
        spec = self.specs[i]
        v_tilde = velocity / max(1e-6, spec.v_max)
        if v_tilde < 0.0:                      # 向心（缩短）
            denom = 1.0 - v_tilde / spec.hill_k
            return max(0.0, (1.0 + v_tilde) / max(1e-6, denom))
        # 离心（拉长）：经验增强，饱和到 1 + eccentric_gain
        return 1.0 + spec.eccentric_gain * (1.0 - math.exp(-3.0 * v_tilde))

    def passive_factor(self, i: int, length: float) -> float:
        """被动力-长度曲线（指数上升，上限 ``passive_max``，避免长位形下力爆炸）。"""
        spec = self.specs[i]
        l_tilde = length / spec.l_opt
        if l_tilde <= 1.0:
            return 0.0
        value = spec.passive_scale * (math.exp(spec.passive_k * (l_tilde - 1.0)) - 1.0)
        return float(min(value, spec.passive_max))

    def factors(self, i: int, length: float, velocity: float):
        """返回 (f_L, f_V, f_PE)。"""
        return (self.length_factor(i, length),
                self.velocity_factor(i, velocity),
                self.passive_factor(i, length))

    # ------------------------------------------------------------------
    # 力与反解
    # ------------------------------------------------------------------
    def force_from_activation(self, i: int, length: float, velocity: float,
                              activation: float) -> float:
        f_l, f_v, f_pe = self.factors(i, length, velocity)
        spec = self.specs[i]
        raw = spec.f_max * (activation * f_l * f_v + f_pe)
        return float(min(max(raw, 0.0), self.force_max_factor * spec.f_max))

    def activation_for_force(self, i: int, length: float, velocity: float,
                             force_desired: float) -> float:
        """反解激励 u：把期望肌力换算成激活度（考虑当前 f_L/f_V 与被动分量）。"""
        f_l, f_v, f_pe = self.factors(i, length, velocity)
        spec = self.specs[i]
        active_capacity = spec.f_max * max(1e-6, f_l * f_v)       # 当前主动力上限
        force_active = float(force_desired) - spec.f_max * f_pe   # 扣除被动分量
        u = force_active / active_capacity
        return float(min(max(u, 0.0), 1.0))

    def update_activation(self, excitation, dt: float) -> None:
        """按激活动力学推进激活度（内部按 ≤0.5 ms 子步积分，保证稳定）。"""
        self.excitation = [float(min(max(u, 0.0), 1.0)) for u in excitation]
        sub_dt = 0.0005
        steps = max(1, int(math.ceil(dt / sub_dt)))
        h = dt / steps
        for _ in range(steps):
            for i in range(self.n):
                a = self.activation[i]
                u = self.excitation[i]
                if u >= a:
                    da = (u - a) * (0.5 + 1.5 * a) / self.tau_act
                else:
                    da = (u - a) / self.tau_deact
                self.activation[i] = min(max(a + da * h, self.activation_min), 1.0)

    def compute_forces(self, lengths, velocities) -> list:
        """用当前激活度计算全部肌力并缓存。"""
        self.force = [self.force_from_activation(i, float(lengths[i]), float(velocities[i]),
                                                self.activation[i])
                      for i in range(self.n)]
        return self.force

    def step(self, lengths, velocities, excitation, dt: float) -> list:
        """一次完整更新：激励 → 激活 → 肌力。"""
        self.update_activation(excitation, dt)
        return self.compute_forces(lengths, velocities)

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"Hill 肌肉模型 | {self.n} 块肌肉 | τ_act={self.tau_act*1000:.0f} ms "
                 f"τ_deact={self.tau_deact*1000:.0f} ms"]
        for i, spec in enumerate(self.specs):
            lines.append(
                f"  {spec.name:<18} F_max={spec.f_max:6.1f} N  L_opt={spec.l_opt*100:5.1f} cm  "
                f"a={self.activation[i]:4.2f}  F={self.force[i]:6.1f} N")
        return "\n".join(lines)
