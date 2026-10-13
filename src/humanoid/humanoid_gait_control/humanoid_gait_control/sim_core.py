# -*- coding: utf-8 -*-
"""
sim_core.py —— 双足步态行走 MuJoCo 仿真核心（无 ROS 依赖）
================================================================

职责
----
1. 加载自建双足人形模型 ``models/humanoid_gait.xml``；
2. 按 ``gait_planner.GaitPlanner`` 的参考轨迹，每步：
   * 支撑腿：髋/膝/踝 由腿部 IK 解到规划位形，保持脚底水平；
   * 摆动腿：IK 跟踪三次 Hermite 摆动轨迹（含抬脚高度）；
   * 双臂：与支撑腿反相摆动（抵消角动量）；
   * 关节空间 PD 伺服（力矩 = Kp·(q_des - q) - Kd·q̇）；
3. 输出：里程计、关节状态、躯干倾角、是否跌倒、前视/侧视渲染。

站立平衡说明
------------
模型全关节直立时，脚底距骨盆中心 0.745 m（髋 0.05 + 大腿 0.34 + 小腿 0.30
+ 踝下 0.055），因此复位时把 freejoint 高度设为 0.745 m 使双脚恰好着地，
质心（骨盆中心近似）高度即 ``planner.com_height``。

本文件不依赖 ROS，standalone 入口（main.py）与 ROS 节点共用同一套逻辑。
"""

from __future__ import annotations

import math
import os

import numpy as np
import mujoco

from .gait_planner import GaitPlanner
from .leg_ik import leg_ik_3d, clamp_joint

#: 模型文件位于包内 models/ 目录
_MODEL_REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "models", "humanoid_gait.xml")

#: 站立平衡高度（骨盆中心 = root 高度，脚底恰好着地）
#: 髋(0.05) + 大腿(0.18+0.34) + 小腿(0.16+0.30) + 踝下(0.055) = 1.085
STAND_Z = 1.085
#: 踝关节中心相对脚底中心：踝 = 脚底 + (-0.03, 0, +0.055)
ANKLE_FROM_FOOT = np.array([-0.03, 0.0, 0.055])
#: 髋关节相对骨盆中心（COM）：(-0.02, ±0.16, -0.05)
HIP_L_OFF = np.array([-0.02, 0.16, -0.05])
HIP_R_OFF = np.array([-0.02, -0.16, -0.05])


class GaitSim:
    """双足步态 MuJoCo 仿真封装。"""

    #: 关节 PD 增益
    KP = 250.0
    KV = 8.0
    #: 侧向（roll）关节加强增益
    KP_ROLL = 600.0
    KV_ROLL = 12.0
    #: 躯干姿态反馈（踝策略）：踝角 += -K·(倾角) - D·(角速度)
    #: 默认关闭：本模型脚掌仅 0.12 m，踝角微扰易致脚跟/脚尖离地，接触非线性
    #: 破坏平衡；稳定性由 LIPM 质心规划 + 落脚点序列 + 关节 PD 保证
    KP_TILT = 0.0
    KD_TILT = 0.0
    KP_TILT_ROLL = 0.0
    KD_TILT_ROLL = 0.0
    #: 姿态反馈幅度上限（rad）≈ 7°，避免踝角突变引起弹跳
    TILT_BIAS_LIMIT = 0.12
    #: 骨盆位置闭环（P-D）：目标修正 = -KP_POS·(位置误差) - KD_POS·(线速度)
    #: 纵向（前进方向）增益适中、限幅较宽（腿工作空间纵向富余）；
    #: 侧向（左右）增益加大并带速度阻尼——侧向偏离会被倒立摆放大，
    #: 但限幅收紧（侧向髋偏移余量小，过大修正会使 IK 超出腿工作空间）
    KP_POS_X = 0.7
    KD_POS_X = 0.5
    POS_BIAS_LIMIT_X = 0.12
    #: LIPM 目标速度前馈系数（量纲 s）：把规划质心速度直接叠加到目标，
    #: 补偿位置伺服固有的滞后，提升巡航跟踪
    K_VEL_FF_X = 0.15
    KP_POS_Y = 2.0
    KD_POS_Y = 0.6
    POS_BIAS_LIMIT_Y = 0.08
    #: 控制力矩限幅
    CTRL_LIMIT = 300.0
    #: 摆臂幅度 / 肘关节固定屈曲
    ARM_SWING = 0.30
    ELBOW_BEND = -0.25

    def __init__(self, planner: GaitPlanner, model_path: str = None,
                 realtime: bool = False):
        self.planner = planner
        path = model_path or _MODEL_REL
        with open(path, encoding="utf-8") as f:
            xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # 定位 root freejoint 与全部人形关节
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qadr = self.model.jnt_qposadr[self.root_id]
        self.root_dadr = self.model.jnt_dofadr[self.root_id]
        self.qadr = {}
        self.dadr = {}
        for i in range(self.model.njnt):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if nm == "root":
                continue
            self.qadr[nm] = self.model.jnt_qposadr[i]
            self.dadr[nm] = self.model.jnt_dofadr[i]

        # 腿/臂/颈关节清单（顺序固定，供控制向量索引）
        self.leg_joints = (
            "hip_roll_l", "hip_pitch_l", "knee_l", "ankle_pitch_l", "ankle_roll_l",
            "hip_roll_r", "hip_pitch_r", "knee_r", "ankle_pitch_r", "ankle_roll_r",
        )
        self.arm_joints = ("shoulder_pitch_l", "elbow_l", "shoulder_pitch_r", "elbow_r")
        self.neck_joints = ("neck",)

        self.realtime = realtime
        self.dt = self.model.opt.timestep
        self.t = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 双脚并立着地：root 高度 = LIPM 质心高度，双腿初始化为静态平衡 IK 解
        # （微屈膝、踝脚着地），使起始帧 PD 力矩 ≈ 0，避免初始阶跃弹跳
        self.data.qpos[self.root_qadr + 2] = float(self.planner.com_height)
        pose0 = self.planner.pose(0.0)
        legs = self._leg_targets(pose0)
        qadr = self.qadr
        for side in ("l", "r"):
            tg = legs[side]
            if tg is None:
                continue
            self.data.qpos[qadr[f"hip_roll_{side}"]] = tg["hip_roll"]
            self.data.qpos[qadr[f"hip_pitch_{side}"]] = tg["hip_pitch"]
            self.data.qpos[qadr[f"knee_{side}"]] = tg["knee"]
            self.data.qpos[qadr[f"ankle_pitch_{side}"]] = tg["ankle_pitch"]
            self.data.qpos[qadr[f"ankle_roll_{side}"]] = tg["ankle_roll"]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0

    def step(self):
        """执行一步仿真：规划 → IK → PD → mj_step。"""
        pose = self.planner.pose(self.t)
        ctrl = self.compute_ctrl(pose)
        self.data.ctrl[:] = np.clip(ctrl, -self.CTRL_LIMIT, self.CTRL_LIMIT)
        mujoco.mj_step(self.model, self.data)
        self.t += self.dt

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------
    def _hip_world(self, side: str, com: np.ndarray) -> np.ndarray:
        off = HIP_L_OFF if side == "l" else HIP_R_OFF
        return com + off

    def _leg_targets(self, pose: dict, com_used: np.ndarray | None = None):
        """计算左右腿的目标关节角（世界系 → 腿部 IK）。

        com_used：支撑腿采用的质心（默认 = 规划质心；传入反馈修正值可闭环
        抑制骨盆漂移）；摆动腿始终用规划质心（按规划轨迹迈步）。
        """
        com_plan = pose["com"]                      # (x, y, z)
        com_sup = com_plan if com_used is None else com_used
        support = pose["support_foot"]         # (x, y) 地面投影
        swing = pose["swing_foot"]             # (x, y, z) 含抬脚高度
        # 支撑腿 = 支撑脚 y 所在一侧（左腿 hip 在 y 正侧）
        support_left = support[1] > 0.0
        targets = {}
        for side, hip_off in (("l", HIP_L_OFF), ("r", HIP_R_OFF)):
            is_support = (side == "l") == support_left
            hip = (com_sup if is_support else com_plan) + hip_off
            is_support = (side == "l") == support_left
            if is_support:
                foot = np.array([support[0], support[1], 0.0])
            else:
                foot = swing
            ankle = foot + ANKLE_FROM_FOOT
            sol = leg_ik_3d(hip, ankle, 0.0)
            if sol is None:
                # 不可达：保持当前关节角（限幅处理）
                targets[side] = None
                continue
            hp, kn, ap, hr, ar = sol
            targets[side] = {
                "hip_pitch": clamp_joint(hp, -1.7, 1.7),
                "knee": clamp_joint(kn, -2.5, 1.5),
                "ankle_pitch": clamp_joint(ap, -0.9, 0.9),
                "hip_roll": clamp_joint(hr, -0.5, 0.5),
                "ankle_roll": clamp_joint(ar, -0.6, 0.6),
            }
        return targets

    def compute_ctrl(self, pose: dict) -> np.ndarray:
        """由规划快照生成关节力矩（PD 伺服 + 摆臂 + 姿态/位置反馈）。"""
        # 0) 骨盆位置闭环：实际位置相对规划的偏差用 P-D 修正（限幅），
        #    反馈到支撑腿目标，抑制倒立摆漂移；纵向/侧向独立增益与限幅
        act = np.array([self.data.qpos[self.root_qadr],
                        self.data.qpos[self.root_qadr + 1]])
        vel = np.array([self.data.qvel[self.root_dadr],
                        self.data.qvel[self.root_dadr + 1]])
        e = act - pose["com"][:2]
        corr_x = np.clip(-(self.KP_POS_X * e[0] + self.KD_POS_X * vel[0]),
                         -self.POS_BIAS_LIMIT_X, self.POS_BIAS_LIMIT_X)
        corr_y = np.clip(-(self.KP_POS_Y * e[1] + self.KD_POS_Y * vel[1]),
                         -self.POS_BIAS_LIMIT_Y, self.POS_BIAS_LIMIT_Y)
        com_used = pose["com"].copy()
        com_used[0] += corr_x + self.K_VEL_FF_X * float(pose["com_vel"][0])
        com_used[1] += corr_y

        # 1) 腿部目标
        leg_targets = self._leg_targets(pose, com_used)
        q_des = np.zeros(len(self.leg_joints))
        for i, jn in enumerate(self.leg_joints):
            side = "l" if jn.endswith("_l") else "r"
            target = leg_targets[side]
            if target is None:
                q_des[i] = 0.0
                continue
            base = jn.rsplit("_", 1)[0]          # 去掉 l/r 后缀
            q_des[i] = target[base]

        # 2) 双臂反相摆动（与支撑腿相位同步：支撑左腿 → 右臂前摆）
        phase = pose["phase"]
        tau = pose["tau"]
        if phase == "prepare":
            arm_amp = 0.0
        else:
            step_phase = math.sin(2.0 * math.pi * tau)
            support_left = pose["support_foot"][1] > 0.0
            # 支撑左腿时右臂前摆
            arm_amp = self.ARM_SWING * step_phase * (1.0 if support_left else -1.0)
        arm = np.array([arm_amp, self.ELBOW_BEND, -arm_amp, self.ELBOW_BEND])
        neck = np.array([0.0])

        # 3) 躯干姿态反馈（踝策略）：前倾(pitch>0) → 踝 pitch 负偏（脚尖上翘），
        #    侧倾(roll>0) → 踝 roll 负偏，把躯干拉回竖直
        roll, pitch, wr, wp = self.torso_pose()
        bp = -self.KP_TILT * pitch - self.KD_TILT * wp
        br = -self.KP_TILT_ROLL * roll - self.KD_TILT_ROLL * wr
        bp = min(max(bp, -self.TILT_BIAS_LIMIT), self.TILT_BIAS_LIMIT)
        br = min(max(br, -self.TILT_BIAS_LIMIT), self.TILT_BIAS_LIMIT)
        q_des[3] += bp        # ankle_pitch_l
        q_des[8] += bp        # ankle_pitch_r
        q_des[4] += br        # ankle_roll_l
        q_des[9] += br        # ankle_roll_r

        # 4) PD 力矩
        q_cur = np.array([self.data.qpos[self.qadr[j]] for j in self.leg_joints])
        qd_cur = np.array([self.data.qvel[self.dadr[j]] for j in self.leg_joints])
        kp = np.full(len(self.leg_joints), self.KP)
        kv = np.full(len(self.leg_joints), self.KV)
        for i, jn in enumerate(self.leg_joints):
            if jn in ("hip_roll_l", "hip_roll_r", "ankle_roll_l", "ankle_roll_r"):
                kp[i] = self.KP_ROLL
                kv[i] = self.KV_ROLL
        ctrl_leg = kp * (q_des - q_cur) - kv * qd_cur

        arm_q = np.array([self.data.qpos[self.qadr[j]] for j in self.arm_joints])
        arm_qd = np.array([self.data.qvel[self.dadr[j]] for j in self.arm_joints])
        ctrl_arm = self.KP * (arm - arm_q) - self.KV * arm_qd

        neck_q = np.array([self.data.qpos[self.qadr[j]] for j in self.neck_joints])
        neck_qd = np.array([self.data.qvel[self.dadr[j]] for j in self.neck_joints])
        ctrl_neck = self.KP * (neck - neck_q) - self.KV * neck_qd

        return np.concatenate([ctrl_leg, ctrl_arm, ctrl_neck])

    # ------------------------------------------------------------------
    # 感知输出
    # ------------------------------------------------------------------
    def read_odom(self) -> dict:
        """里程计：位置、偏航角、速度。"""
        q = self.data.qpos
        r = self.root_qadr
        qw, qx, qy, qz = q[r + 3: r + 7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return {
            "x": float(q[r]), "y": float(q[r + 1]), "z": float(q[r + 2]),
            "yaw": float(yaw),
            "vx": float(self.data.qvel[self.root_dadr]),
            "t": self.t,
        }

    def torso_pose(self):
        """躯干姿态与角速度：返回 (roll, pitch, roll_rate, pitch_rate)。

        roll 正 = 右侧下沉（绕 x），pitch 正 = 前倾（绕 y）。
        """
        r = self.root_qadr
        qw, qx, qy, qz = self.data.qpos[r + 3: r + 7]
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(min(max(sinp, -1.0), 1.0))
        w = self.data.qvel[self.root_dadr + 3: self.root_dadr + 6]
        return roll, pitch, float(w[0]), float(w[1])

    def torso_tilt(self) -> float:
        """躯干倾角（rad，绕 x 轴与绕 y 轴的合成，衡量前后/左右倾倒）。"""
        roll, pitch, _, _ = self.torso_pose()
        return math.hypot(roll, pitch)

    def standing(self) -> bool:
        """是否仍保持站立（根高 > 0.5 m，脚底脱离地面即视为跌倒）。"""
        return self.data.qpos[self.root_qadr + 2] > 0.5

    def render(self, camera: str = "overview", height: int = 480, width: int = 640):
        """渲染相机图像（RGB ndarray）。"""
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        img = renderer.render()
        del renderer
        return img

    # ------------------------------------------------------------------
    # 行走执行（供 main.py / 测试调用）
    # ------------------------------------------------------------------
    def walk(self, on_step=None):
        """执行完整步态直至结束；on_step(t, odom, tilt) 回调用于采样。

        返回 (是否未跌倒, 里程计序列, 躯干倾角序列)。
        """
        odoms, tilts = [], []
        total = int(round(self.planner.total_time / self.dt))
        fell = False
        for _ in range(total):
            self.step()
            od = self.read_odom()
            tilt = self.torso_tilt()
            odoms.append(od)
            tilts.append(tilt)
            if on_step is not None:
                on_step(self.t, od, tilt)
            if not self.standing():
                fell = True
                break
        return (not fell), odoms, tilts
