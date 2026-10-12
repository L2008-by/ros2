# -*- coding: utf-8 -*-
"""
sim_core.py —— 肌肉骨骼人手臂的 MuJoCo 仿真核心（无 ROS 依赖）
==============================================================

设计要点
--------
1. **肌肉驱动**：6 块肌肉（3 个自由度各一对主动肌/拮抗肌）通过空间腱跨越关节，
   MuJoCo 负责刚体动力学与腱长几何，肌力由 :mod:`muscle_model` 显式计算。
2. **力矩臂实测**：不手写「哪块肌肉管哪个关节、正负号是什么」，而是用有限差分
   从模型自身测出力矩臂矩阵 ``M[i, j] = ∂L_i/∂q_j``（``L_i`` 为腱长，``q_j`` 为关节角）。
   肌力对关节力矩的贡献为 ``dτ_j/dF_i = -M[i, j]``——符号来自模型，避免配错主动肌。
3. **控制分配**：末端（hand site）任务空间 PD → 雅可比转置得到期望关节力矩
   → 叠加 ``qfrc_bias`` 重力/科氏补偿 → 用最小范数分配反解 6 块肌肉的激活度，
   并保留共收缩基量 ``c0``（提供关节刚度）。
4. **可复现**：standalone（``main.py``）与 ROS2 节点共用本文件，行为完全一致。

坐标系：+x 前、+y 左、+z 上；右肩位于世界坐标 (0, -0.20, 1.40)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields

import numpy as np
import mujoco

from humanoid_muscle_arm_control.muscle_model import HillMuscleModel, MuscleSpec

try:                                                     # 协同降维（可选功能）
    from humanoid_muscle_arm_control import synergy as synergy_lib
except ImportError:                                      # pragma: no cover - 源码直跑
    import synergy as synergy_lib

# 模型文件位于包内 models/ 目录
_MODEL_REL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "musculoskeletal_arm.xml")

#: 受控关节（顺序即膝关节/力矩臂矩阵的列顺序）
JOINT_NAMES = ("shoulder_pitch", "shoulder_roll", "elbow_pitch")

#: 肌肉（顺序即执行器顺序、力矩臂矩阵的行顺序）
MUSCLE_NAMES = ("shoulder_flexor", "shoulder_extensor", "shoulder_abductor",
                "shoulder_adductor", "elbow_flexor", "elbow_extensor")

#: 各肌肉最大等长收缩力 (N)
#: 取值原则：让每个自由度的最大肌力矩约为自重重力力矩的 3~5 倍。
#: （若按解剖学真实值取数百牛，力矩臂 0.1 m 会得到 70~110 N·m 的极端权威，
#:   简单地 PD 控制会瞬间饱和振荡，因此本教学模型按"够用且可控"标定。）
MUSCLE_F_MAX = {"shoulder_flexor": 320.0, "shoulder_extensor": 320.0,
                "shoulder_abductor": 200.0, "shoulder_adductor": 200.0,
                "elbow_flexor": 220.0, "elbow_extensor": 300.0}

#: 初始关节角（肩 pitch / 肩 roll / 肘 pitch）——肘部保持屈曲，远离奇异位形
HOME_QPOS = np.array([-0.35, 0.00, -1.00])

#: 目标点允许范围（世界坐标，m）与可达半径（略小于 0.30+0.26 = 0.56 m）
TARGET_BOUNDS = (np.array([-0.20, -0.75, 0.80]), np.array([0.75, 0.10, 1.90]))
HAND_REACH = 0.52


@dataclass
class ArmParams:
    """控制器与肌肉参数（可由 ``config/sim_params.yaml`` 覆盖）。"""

    sim_rate: float = 500.0            # 物理仿真频率 (Hz)
    publish_rate: float = 30.0         # 话题发布频率 (Hz)
    kp_task: float = 60.0              # 任务空间比例增益 (N/m)（control_mode="task" 时使用）
    kd_task: float = 14.0              # 任务空间阻尼 (N·s/m)
    kp_joint: float = 12.0             # 关节空间比例增益 (N·m/rad)；过大会超出肌肉力矩能力而振荡
    kd_joint: float = 3.0              # 关节阻尼注入 (N·m·s/rad)，两种模式都生效
    control_mode: str = "ik"           # "ik"：DLS 逆解 + 关节 PD；"task"：雅可比转置任务空间 PD
    f_task_max: float = 10.0           # 任务力限幅 (N)
    tau_task_max: float = 10.0         # 任务力矩限幅 (N·m)，重力补偿不在此限
    bias_compensation: bool = True     # 是否补偿重力/科氏（关闭用于对照实验）
    cocontraction: float = 0.04        # 共收缩基量（相对 F_max）
    tau_act: float = 0.015             # 激活时间常数 (s)
    tau_deact: float = 0.060           # 放松时间常数 (s)
    moment_refresh_steps: int = 100    # 力矩臂重标定周期（步）
    hand_speed: float = 0.18           # 键盘控制目标点速度 (m/s)
    use_synergy: bool = False          # 是否把 6 维肌肉激励压到 2 维协同再合成
    synergy_scale: float = 1.0         # 协同增益（>1 会更容易超出主动力上限）

    @classmethod
    def from_dict(cls, config):
        """从 dict 构造（忽略未知键，支持 bool 字符串）。"""
        known = {f.name: f.type for f in fields(cls)}
        kwargs = {}
        for key, value in (config or {}).items():
            if key not in known or value is None:
                continue
            if isinstance(value, str) and value.lower() in ("true", "false"):
                value = value.lower() == "true"
            kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self):
        return asdict(self)


def load_params(config_path=None):
    """从 YAML 读取参数（ROS2 两段式 ``/**/ros__parameters`` 或扁平结构都能吃）。

    读取失败（文件不存在 / 未安装 PyYAML）时返回默认参数，保证 standalone 可用。
    """
    params = ArmParams()
    if not config_path or not os.path.exists(config_path):
        return params
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception:                                    # pragma: no cover - 环境缺 PyYAML
        return params
    section = raw
    if isinstance(raw, dict):
        wildcard = raw.get("/**")
        if isinstance(wildcard, dict) and isinstance(wildcard.get("ros__parameters"), dict):
            section = wildcard["ros__parameters"]
        elif isinstance(raw.get("ros__parameters"), dict):
            section = raw["ros__parameters"]
    if isinstance(section, dict):
        return ArmParams.from_dict(section)
    return params


class MusculoskeletalArm:
    """肌肉骨骼人手臂仿真封装（模型 + 肌肉 + 控制分配）。"""

    def __init__(self, model_path=None, params=None):
        self.params = params if params is not None else ArmParams()
        path = model_path or _MODEL_REL
        with open(path, encoding="utf-8") as handle:      # 先读字符串，兼容非 ASCII 路径
            xml = handle.read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model_path = path

        # ---- 索引 ----
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in JOINT_NAMES]
        self.qpos_adr = [int(self.model.jnt_qposadr[i]) for i in self.joint_ids]
        self.dof_adr = [int(self.model.jnt_dofadr[i]) for i in self.joint_ids]
        self.body_shoulder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "upper_arm")
        self.body_forearm = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "forearm")
        self.site_hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand")
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in MUSCLE_NAMES]

        # 模型里三个目标球的位置，用作演示/交互的目标点
        self.landmarks = {}
        for name in ("target_forward", "target_lateral", "target_low"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.landmarks[name] = np.array(self.model.body_pos[bid], dtype=float)

        # ---- 肌肉模型 ----
        self.f_max_vec = np.array([MUSCLE_F_MAX[n] for n in MUSCLE_NAMES], dtype=float)
        specs = [MuscleSpec(name=n, f_max=MUSCLE_F_MAX[n]) for n in MUSCLE_NAMES]
        self.muscles = HillMuscleModel(specs, tau_act=self.params.tau_act,
                                       tau_deact=self.params.tau_deact)

        # ---- 状态 ----
        self.step_count = 0
        self.moment_arms = np.zeros((len(MUSCLE_NAMES), len(JOINT_NAMES)))
        self._jacp = np.zeros((3, self.model.nv))
        self._renderer = None
        self._renderer_shape = None
        self._q_ik = None
        self.synergy_coefficients = np.zeros(synergy_lib.SYNERGY_BASIS.shape[0])
        self.last = {}

        self.reset()

    # ------------------------------------------------------------------
    # 初始化 / 标定
    # ------------------------------------------------------------------
    def reset(self):
        """回到初始姿态，清空激活与目标点。"""
        mujoco.mj_resetData(self.model, self.data)
        for adr, value in zip(self.qpos_adr, HOME_QPOS):
            self.data.qpos[adr] = value
        mujoco.mj_forward(self.model, self.data)
        self.muscles.reset()
        self.step_count = 0
        self._q_ik = self.joint_angles().copy()
        self._target = self.hand_position().copy()
        self.muscles.calibrate_lengths(self.nominal_lengths())
        self.calibrate_moment_arms()
        return self.state()

    def nominal_lengths(self):
        """标称姿态下各肌肉的腱长（用于标定最佳纤维长度）。"""
        return {name: float(self.data.actuator_length[self.act_ids[i]])
                for i, name in enumerate(MUSCLE_NAMES)}

    def calibrate_moment_arms(self, eps=1e-4):
        """有限差分实测力矩臂矩阵 ``M[i, j] = ∂L_i/∂q_j``（单位 m）。"""
        qpos0 = self.data.qpos.copy()
        for j, adr in enumerate(self.qpos_adr):
            self.data.qpos[adr] = qpos0[adr] + eps
            mujoco.mj_forward(self.model, self.data)
            length_plus = np.array(self.data.actuator_length, dtype=float)
            self.data.qpos[adr] = qpos0[adr] - eps
            mujoco.mj_forward(self.model, self.data)
            length_minus = np.array(self.data.actuator_length, dtype=float)
            self.moment_arms[:, j] = (length_plus - length_minus) / (2.0 * eps)
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)
        return self.moment_arms

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------
    def hand_position(self):
        return np.array(self.data.site_xpos[self.site_hand], dtype=float)

    def shoulder_position(self):
        return np.array(self.data.xpos[self.body_shoulder], dtype=float)

    def hand_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_hand)
        return jacp

    def hand_velocity(self):
        return self.hand_jacobian() @ np.array(self.data.qvel, dtype=float)

    def joint_angles(self):
        return np.array([self.data.qpos[adr] for adr in self.qpos_adr], dtype=float)

    def state(self):
        """一次控制步的全部可观测量（供日志、ROS2 话题与测试使用）。"""
        return {
            "time": float(self.data.time),
            "q": self.joint_angles().copy(),
            "hand": self.hand_position().copy(),
            "target": np.array(self._target, dtype=float).copy(),
            "hand_error": float(np.linalg.norm(self._target - self.hand_position())),
            "activation": np.array(self.muscles.activation, dtype=float),
            "force": np.array(self.muscles.force, dtype=float),
            "muscle_length": np.array(self.data.actuator_length, dtype=float),
            "moment_arms": self.moment_arms.copy(),
            "synergy": np.array(self.synergy_coefficients, dtype=float).copy(),
            "q_ik": (None if self._q_ik is None else np.array(self._q_ik, dtype=float).copy()),
            "control_mode": self.params.control_mode,
        }

    # ------------------------------------------------------------------
    # 目标点
    # ------------------------------------------------------------------
    def set_target(self, position, clip=True, solve=True):
        """设置末端目标点；``clip=True`` 时限制在工作空间内（含可达半径）。

        ``control_mode="ik"`` 时同时解一次 DLS 逆运动学得到关节目标 ``_q_ik``：
        目标跳变大（> 5 cm，例如演示路标）用完整迭代，键盘连续微动用热启动少迭代。
        """
        target = np.array(position, dtype=float)
        if clip:
            lo, hi = TARGET_BOUNDS
            target = np.clip(target, lo, hi)
            shoulder = self.shoulder_position()
            offset = target - shoulder
            distance = float(np.linalg.norm(offset))
            if distance > HAND_REACH:
                target = shoulder + offset * (HAND_REACH / distance)
        self._target = target
        if solve and self.params.control_mode == "ik":
            jump = float(np.linalg.norm(target - self.hand_position()))
            iterations = 400 if jump > 0.05 else 12
            self._q_ik = self.solve_ik(target, iterations=iterations,
                                       warm_start=self._q_ik)
        return self._target

    def target(self):
        return np.array(self._target, dtype=float)

    def move_target(self, direction, distance=None):
        """沿给定方向平移目标点（键盘控制用）。"""
        step = self.params.hand_speed * self.model.opt.timestep if distance is None else distance
        return self.set_target(self._target + np.array(direction, dtype=float) * step)

    # ------------------------------------------------------------------
    # 控制与仿真
    # ------------------------------------------------------------------
    def _task_force(self):
        error = self._target - self.hand_position()
        velocity = self._jacp @ np.array(self.data.qvel, dtype=float)
        force = self.params.kp_task * error - self.params.kd_task * velocity
        norm = float(np.linalg.norm(force))
        if norm > self.params.f_task_max:
            force = force * (self.params.f_task_max / norm)
        return force

    def _desired_torque(self, task_force):
        jac = self._jacp[:, self.dof_adr]                       # (3 关节, 3 任务)
        qvel = np.array(self.data.qvel[self.dof_adr], dtype=float)
        if self.params.control_mode == "ik" and self._q_ik is not None:
            # DLS 逆解给出可行关节目标 → 关节空间 PD（不受雅可比局部解/限位停滞影响）
            q_err = self._q_ik - self.joint_angles()
            task_torque = self.params.kp_joint * q_err - self.params.kd_joint * qvel
        else:
            task_torque = jac.T @ task_force
            if self.params.kd_joint > 0.0:                      # 关节阻尼注入
                task_torque = task_torque - self.params.kd_joint * qvel
        limit = float(self.params.tau_task_max)                 # 任务力矩限幅（防饱和振荡）
        task_torque = np.clip(task_torque, -limit, limit)
        if self.params.bias_compensation:                       # 重力 + 科氏补偿（全额通过）
            task_torque = task_torque + np.array(self.data.qfrc_bias[self.dof_adr], dtype=float)
        return task_torque

    # ------------------------------------------------------------------
    # 数值逆运动学（阻尼最小二乘）
    # ------------------------------------------------------------------
    def _ik_iterate(self, target, q, iterations, tol, damping, max_step, lower, upper):
        """单次 DLS 迭代（从给定起点），返回 (q, 末端残差)。"""
        for _ in range(int(iterations)):
            for adr, value in zip(self.qpos_adr, q):
                self.data.qpos[adr] = value
            mujoco.mj_forward(self.model, self.data)
            error = np.asarray(target, dtype=float) - self.hand_position()
            if float(np.linalg.norm(error)) < tol:
                return q, float(np.linalg.norm(error))
            jac = self.hand_jacobian()[:, self.dof_adr]                      # 3×3
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), error)
            q = np.clip(q + np.clip(step, -max_step, max_step), lower, upper)
        for adr, value in zip(self.qpos_adr, q):
            self.data.qpos[adr] = value
        mujoco.mj_forward(self.model, self.data)
        return q, float(np.linalg.norm(np.asarray(target, dtype=float) - self.hand_position()))

    def solve_ik(self, target, iterations=300, tol=1e-4, damping=1e-3, max_step=0.15,
                 warm_start=None, extra_starts=6):
        """阻尼最小二乘（DLS）数值逆解：求末端到达 ``target`` 的关节角。

        单起点很容易卡在关节限位上（实测"外展/上举"两个路标正是如此），因此采用
        **多起点重试**：热启动 → 当前位形 → 初始位形 → 若干关节范围内均匀试探，取残差最小者。
        求解过程临时写入试探位形（``mj_forward``），结束后恢复原状态；残差记在
        ``self.last_ik_error`` 里便于诊断。返回关节角数组。
        """
        qpos_backup = self.data.qpos.copy()
        lower = np.array([self.model.jnt_range[i][0] for i in self.joint_ids], dtype=float)
        upper = np.array([self.model.jnt_range[i][1] for i in self.joint_ids], dtype=float)

        starts = []
        if warm_start is not None:
            starts.append(np.array(warm_start, dtype=float))
        starts.append(self.joint_angles().copy())
        starts.append(np.array(HOME_QPOS, dtype=float))
        if extra_starts:
            rng = np.random.default_rng(0)
            for _ in range(int(extra_starts)):
                starts.append(rng.uniform(lower, upper))

        best_q, best_error = None, float("inf")
        for q0 in starts:
            q, error = self._ik_iterate(target, np.clip(q0, lower, upper), iterations,
                                        tol, damping, max_step, lower, upper)
            if error < best_error:
                best_q, best_error = q, error
            if best_error < tol:
                break

        self.data.qpos[:] = qpos_backup
        mujoco.mj_forward(self.model, self.data)
        self.last_ik_error = best_error
        return best_q

    def _allocate_forces(self, torque_desired, force_base=None, tol=1e-3):
        """**非负约束**下的最小范数肌肉力分配（字典序目标）。

        肌肉只能产生拉力（``F_i ≥ 0``），而无约束的最小范数解往往要求某些肌肉"推"。
        这里枚举激活子集求严格非负解，并按**字典序**挑选：

        1. 先要求**能复现期望力矩**（``‖M_Sᵀ ΔF_S − residual‖ ≤ tol``，``tol`` 单位 N·m）；
        2. 再在满足力矩的子集里取 ``‖ΔF‖²`` 最小者（即最小肌肉募集）；
        3. 若没有任何子集能精确复现（真实的肌肉力量饱和），退而取残差最小者。

        ``force_base`` 是"不受控的基准肌力"（共收缩基量 + 被动弹性力）。它也会产生关节力矩，
        且因主动肌/拮抗肌力矩臂不等不能自动抵消，必须从期望力矩里扣除——
        否则会形成恒定偏置，表现为几十毫米的稳态跟踪误差甚至振荡。
        """
        gain = self.moment_arms
        base = (self.params.cocontraction * self.f_max_vec
                if force_base is None else np.asarray(force_base, dtype=float))
        residual = np.asarray(torque_desired, dtype=float) - gain.T @ base
        n = gain.shape[0]

        delta, *_ = np.linalg.lstsq(gain.T, residual, rcond=None)          # 快路径
        if np.all(delta >= -base - 1e-9) and np.all(delta <= self.f_max_vec - base + 1e-9):
            return np.clip(base + delta, 0.0, self.f_max_vec)

        candidates = [i for i in range(n) if float(np.dot(gain[i], residual)) > 0.0]
        best_key, best_delta = None, None
        for pool in (candidates, list(range(n))):                          # 预筛 → 全枚举
            for mask in range(1, 1 << len(pool)):
                idx = [pool[k] for k in range(len(pool)) if (mask >> k) & 1]
                sub = gain[idx, :].T                                       # (3, k)
                part, *_ = np.linalg.lstsq(sub, residual, rcond=None)
                if np.any(part < -base[idx] - 1e-9) or \
                   np.any(part > self.f_max_vec[idx] - base[idx] + 1e-9):
                    continue
                error = float(np.linalg.norm(sub @ part - residual))
                key = (0 if error <= tol else 1, error, float(np.dot(part, part)))
                if best_key is None or key < best_key:
                    full = np.zeros(n)
                    full[idx] = part
                    best_key, best_delta = key, full
            if best_key is not None and best_key[0] == 0:                  # 已找到满足力矩的解
                break
        if best_delta is None:
            return base                                                    # 兜底：仅共收缩
        return np.clip(base + best_delta, 0.0, self.f_max_vec)

    def allocate(self, torque_desired):
        """期望关节力矩 → 6 块肌肉的力指令与激励（最小范数 + 共收缩基量）。

        肌肉力对关节力矩的贡献矩阵 ``G[i, j] = -M[i, j]``；
        对每个关节的最小范数分配为 ``ΔF_i = G[i, j] · τ_j / Σ_k G[k, j]²``。

        ⚠ 两个必须注意的点（否则会出现恒定偏置、交叉耦合误差或**正反馈振荡**）：

        1. **力矩贡献的符号**：本模块用「单块肌肉施加恒力、观察关节运动」的实测标定确认，
           MuJoCo 的 tendon 执行器对关节的力矩贡献为 ``τ_j = +M[i,j]·F_i``
           （``M[i,j] = ∂L_i/∂q_j``）。按教科书写成 ``-M`` 会让分配器命令反向力矩，
           整条控制回路变成正反馈、手臂被推飞——这一点只能靠真仿真实验发现。
        2. **共收缩基量要扣除**：``c0·F_max`` 本身产生关节力矩，且因主动肌/拮抗肌力矩臂
           不等而不能自动抵消；同时力矩臂矩阵存在交叉耦合（屈肘肌跨肩肘、外展肌同时影响
           pitch/roll），因此必须用最小范数解而不是"按关节除以列范数²"。

        于是 ``G = M``，取最小范数解：

        .. math::  \\Delta F = G (G^T G + \\varepsilon I)^{-1} (\\tau^* - M^T F_{base})

        当力指令超出 ``F_max`` 时会被截断，此时实际力矩小于期望（肌肉力量饱和），
        属于真实存在的限制。

        ``params.use_synergy=True`` 时，把 6 维激励先投影到 2 维肌肉协同再合成回来。
        """
        gain = self.moment_arms                                   # (6 肌肉, 3 关节)
        base = self.params.cocontraction * self.f_max_vec
        lengths = np.array(self.data.actuator_length, dtype=float)
        velocities = np.array(self.data.actuator_velocity, dtype=float)
        # 不受控的基准肌力 = 共收缩基量 + 被动弹性力（后者随长度变化，是主要的稳态偏置来源）
        passive = np.array([self.muscles.specs[i].f_max * self.muscles.passive_factor(i, lengths[i])
                            for i in range(len(MUSCLE_NAMES))], dtype=float)
        force_cmd = self._allocate_forces(torque_desired, force_base=base + passive)
        excitation = np.array([
            self.muscles.activation_for_force(i, lengths[i], velocities[i], force_cmd[i])
            for i in range(len(MUSCLE_NAMES))], dtype=float)

        self.synergy_coefficients = synergy_lib.project(excitation)
        if self.params.use_synergy:                          # 2 维协同控制模式
            excitation = synergy_lib.synthesis(
                self.synergy_coefficients * self.params.synergy_scale)
        return force_cmd, excitation

    def step(self, external_force=None):
        """推进一个物理步： kinematics → 任务力 → 关节力矩 → 肌肉分配 → mj_step。

        ``external_force`` 不为空时，在末端（hand site）施加该世界系外力，
        用于刚度实验（``measure_stiffness``）。
        """
        mujoco.mj_forward(self.model, self.data)                # 同步 site / 腱长 / 雅可比
        period = max(1, int(self.params.moment_refresh_steps))
        if self.step_count % period == 0:                      # 力矩臂随姿态缓慢刷新
            self.calibrate_moment_arms()

        self._jacp = self.hand_jacobian()
        task_force = self._task_force()
        torque = self._desired_torque(task_force)
        force_cmd, excitation = self.allocate(torque)

        lengths = np.array(self.data.actuator_length, dtype=float)
        velocities = np.array(self.data.actuator_velocity, dtype=float)
        forces = self.muscles.step(lengths, velocities, excitation, self.model.opt.timestep)
        self.data.ctrl[:] = np.array(forces, dtype=float)

        self.data.qfrc_applied[:] = 0.0
        if external_force is not None:                          # 末端探针力（刚度实验）
            mujoco.mj_applyFT(self.model, self.data,
                              np.array(external_force, dtype=float), np.zeros(3),
                              self.hand_position(), self.body_forearm, self.data.qfrc_applied)

        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.last = {
            "task_force": task_force,
            "torque": torque,
            "force_cmd": force_cmd,
            "excitation": excitation,
            "force": np.array(forces, dtype=float),
        }
        return self.last

    # ------------------------------------------------------------------
    # 实验：共收缩 → 末端刚度
    # ------------------------------------------------------------------
    def measure_stiffness(self, direction=(1.0, 0.0, 0.0), force=8.0, cocontraction=None,
                          settle_time=1.2, recovery_time=0.4):
        """施加已知探针力，实测末端稳态位移，得到沿力方向的等效刚度 (N/m)。

        步骤：先让控制器收敛到当前目标点（无外力）→ 记录自由位置 → 施加恒定外力
        → 记录稳态位置 → 卸载并恢复。刚度 = 力 / 沿力方向的位移投影。
        """
        saved = self.params.cocontraction
        # 刚度实验期间关闭位置反馈：否则控制器会把探针力顶回去，测到的是控制器刚度
        saved_gains = (self.params.kp_joint, self.params.kp_task)
        self.params.kp_joint, self.params.kp_task = 0.0, 0.0
        if cocontraction is not None:
            self.params.cocontraction = float(cocontraction)
        unit = np.array(direction, dtype=float)
        unit = unit / max(1e-9, float(np.linalg.norm(unit)))
        steps = max(1, int(settle_time / self.model.opt.timestep))

        for _ in range(steps):
            self.step()
        free_position = self.hand_position().copy()

        for _ in range(steps):
            self.step(external_force=unit * force)
        loaded_position = self.hand_position().copy()

        for _ in range(max(1, int(recovery_time / self.model.opt.timestep))):
            self.step()

        self.params.cocontraction = saved
        self.params.kp_joint, self.params.kp_task = saved_gains
        displacement = loaded_position - free_position
        projected = float(np.dot(displacement, unit))
        stiffness = float(force / projected) if abs(projected) > 1e-6 else float("inf")
        return {
            "cocontraction": float(cocontraction if cocontraction is not None else saved),
            "force": float(force),
            "direction": unit,
            "displacement": displacement,
            "projected_displacement": projected,
            "displacement_norm": float(np.linalg.norm(displacement)),
            "stiffness": stiffness,
        }

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self, camera="overview", height=480, width=640):
        """渲染一帧（复用 Renderer，避免每帧新建上下文）。"""
        if self._renderer is None or self._renderer_shape != (height, width):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_shape = (height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def moment_arm_table(self):
        lines = ["力矩臂矩阵 M[i,j] = ∂L_i/∂q_j  (单位: m)"]
        header = "  {:<20}".format("肌肉 \\ 关节") + "".join(f"{n:>18}" for n in JOINT_NAMES)
        lines.append(header)
        for i, name in enumerate(MUSCLE_NAMES):
            row = "  {:<20}".format(name) + "".join(f"{self.moment_arms[i, j]:>18.4f}"
                                                    for j in range(len(JOINT_NAMES)))
            lines.append(row)
        return "\n".join(lines)

    def antagonist_check(self):
        """检查每个关节是否都有正/负力矩臂（即主动肌-拮抗肌配对）。"""
        report = []
        for j, name in enumerate(JOINT_NAMES):
            column = self.moment_arms[:, j]
            positive = [MUSCLE_NAMES[i] for i in range(len(MUSCLE_NAMES)) if column[i] > 1e-4]
            negative = [MUSCLE_NAMES[i] for i in range(len(MUSCLE_NAMES)) if column[i] < -1e-4]
            report.append({"joint": name, "positive": positive, "negative": negative,
                           "paired": bool(positive) and bool(negative),
                           "max_abs": float(np.max(np.abs(column)))})
        return report

    def summary(self):
        lines = [
            f"模型: {os.path.basename(self.model_path)} | nq={self.model.nq} nv={self.model.nv} "
            f"nu={self.model.nu} | 时间步 {self.model.opt.timestep*1000:.1f} ms",
            f"关节: {', '.join(JOINT_NAMES)}",
            self.muscles.summary(),
        ]
        return "\n".join(lines)
