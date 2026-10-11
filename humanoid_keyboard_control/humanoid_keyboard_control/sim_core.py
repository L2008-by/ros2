# -*- coding: utf-8 -*-
"""
sim_core.py —— 人形机器人 MuJoCo 仿真核心（无 ROS 依赖）

职责：
  1. 加载自建简化人形 MJCF 模型（models/humanoid_simple.xml）；
  2. 站姿控制：所有关节 PD 伺服回直立位（髋/踝 roll 单独加强增益）；
  3. 运动控制（键盘指令 v/w）：
       - 前进/后退：速度级控制（覆盖底盘 freejoint 的世界系线速度）；
       - 转向：姿态级控制（直接对底盘四元数做微小旋转，避免双足原地
         旋转时脚底摩擦滑移导致的侧倾失稳）；
  4. 输出：里程计（x, y, yaw）、关节状态、前视相机渲染。

本文件不依赖 ROS，standalone 入口（main.py）与 ROS 节点（mujoco_sim_node.py）
共用同一套仿真逻辑。
"""

import math
import os

import numpy as np
import mujoco

# 模型文件位于包内 models/ 目录
_MODEL_REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "humanoid_simple.xml")


class HumanoidSim:
    """人形机器人 MuJoCo 仿真封装。"""

    #: 站姿 PD 增益（关节力矩 = kp * (目标-当前) - kv * 角速度）
    KP = 120.0
    KV = 6.0
    #: 髋/踝 roll 关节单独加强，抗侧倾
    KP_ROLL = 400.0
    #: 运动指令限幅
    V_MAX = 0.3      # m/s
    W_MAX = 0.6      # rad/s

    def __init__(self, model_path=None, realtime=False):
        # 中文/非 ASCII 路径兼容：先读文件内容再从字符串加载
        path = model_path or _MODEL_REL
        with open(path, encoding="utf-8") as f:
            xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # 定位 root freejoint 与人形关节
        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qadr = self.model.jnt_qposadr[self.root_id]
        self.root_dadr = self.model.jnt_dofadr[self.root_id]
        self.human_qadr = [self.model.jnt_qposadr[i]
                           for i in range(self.model.njnt)
                           if self.model.jnt_qposadr[i] > self.root_qadr]
        self.human_dadr = [self.model.jnt_dofadr[i]
                           for i in range(self.model.njnt)
                           if self.model.jnt_qposadr[i] > self.root_qadr]
        # roll 关节（抗侧倾加强）
        self.roll_idx = []
        for nm in ("hip_roll_l", "hip_roll_r", "ankle_roll_l", "ankle_roll_r"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            self.roll_idx.append(self.human_qadr.index(self.model.jnt_qposadr[jid]))

        self.realtime = realtime
        self.dt = self.model.opt.timestep
        self.reset()

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, v_lin=0.0, w_yaw=0.0):
        """执行一步仿真。

        参数:
            v_lin: 线速度指令（世界系 +x，前进为正），速度级控制
            w_yaw: 偏航角速度指令（rad/s），姿态级控制
        """
        v_lin = float(np.clip(v_lin, -self.V_MAX, self.V_MAX))
        w_yaw = float(np.clip(w_yaw, -self.W_MAX, self.W_MAX))

        # 1) 站姿 PD 力矩（目标姿态：全部关节直立 0）
        ctrl = self.KP * (-self.data.qpos[self.human_qadr]) - self.KV * self.data.qvel[self.human_dadr]
        for qi in self.roll_idx:
            ctrl[qi] = self.KP_ROLL * (-self.data.qpos[self.human_qadr[qi]]) \
                       - self.KV * self.data.qvel[self.human_dadr[qi]]
        self.data.ctrl[:] = np.clip(ctrl, -250.0, 250.0)

        # 2) 前进/后退：速度级控制（覆盖 freejoint 世界系线速度）
        self.data.qvel[self.root_dadr] = v_lin
        # 3) 转向：姿态级控制（清除角速度，直接旋转四元数）
        self.data.qvel[self.root_dadr + 3: self.root_dadr + 6] = 0.0
        if w_yaw != 0.0:
            d = w_yaw * self.dt
            qw, qx, qy, qz = self.data.qpos[self.root_qadr + 3: self.root_qadr + 7]
            dw, dx, dy, dz = math.cos(d / 2), 0.0, 0.0, math.sin(d / 2)
            nw = qw*dw - qx*dx - qy*dy - qz*dz
            nx = qw*dx + qx*dw + qy*dz - qz*dy
            ny = qw*dy - qx*dz + qy*dw + qz*dx
            nz = qw*dz + qx*dy - qy*dx + qz*dw
            self.data.qpos[self.root_qadr + 3: self.root_qadr + 7] = np.array([nw, nx, ny, nz])

        mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    # 感知输出
    # ------------------------------------------------------------------
    def read_odom(self):
        """里程计：位置 (x, y)、偏航角、速度。"""
        q = self.data.qpos
        r = self.root_qadr
        qw, qx, qy, qz = q[r + 3: r + 7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return {
            "x": float(q[r]), "y": float(q[r + 1]), "z": float(q[r + 2]),
            "yaw": float(yaw),
            "vx": float(self.data.qvel[self.root_dadr]),
            "wz": float(self.data.qvel[self.root_dadr + 5]),
        }

    def read_joint_positions(self):
        """关节位置（rad），键为关节名。"""
        out = {}
        for i in range(self.model.njnt):
            if self.model.jnt_qposadr[i] <= self.root_qadr:
                continue
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            out[nm] = float(self.data.qpos[self.model.jnt_qposadr[i]])
        return out

    def render(self, camera="front_camera", height=480, width=640):
        """渲染相机图像（RGB ndarray）。"""
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        img = renderer.render()
        del renderer
        return img

    def standing(self):
        """是否仍保持站立（根高 > 0.9 m）。"""
        return self.data.qpos[self.root_qadr + 2] > 0.9
