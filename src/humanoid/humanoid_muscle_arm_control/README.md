# humanoid_muscle_arm_control

肌肉骨骼人（右上肢）**肌肉驱动**的 MuJoCo 物理仿真与键盘运动控制 ROS2 功能包。

对应课程任务：**① 人形机器人 / 肌肉骨骼人 / 车辆 / 水下机器人 / 无人机物理仿真 + 键盘运动控制**（本模块为其中的**肌肉驱动上肢**，按仓库分类归档于「人形机器人」）。

## 功能概述

- 自建简化肌肉骨骼人上肢 MJCF 模型：固定躯干 + 右臂 3 自由度（肩 pitch / 肩 roll / 肘 pitch），
  末端带 `hand` site，场景内置三个目标球（前伸 / 外展 / 上举）；
- **肌肉驱动**：6 块肌肉（每个自由度一对主动肌/拮抗肌，其中屈肘肌为跨肩肘双关节肌）通过
  **空间腱**跨越关节，执行器输出的是「肌肉力」；
- **Hill 型肌肉模型**（`muscle_model.py`）：一阶激活动力学（`τ_act` / `τ_deact`）、
  力-长度曲线（高斯主动 + 指数被动）、力-速度曲线（向心 Hill 双曲线 + 离心增强）；
- **力矩臂实测**（`sim_core.calibrate_moment_arms`）：由模型自身用有限差分求
  `M[i,j] = ∂L_i/∂q_j`，肌力对关节力矩的贡献 `dτ_j/dF_i = -M[i,j]`——**符号来自模型，不手写**，
  避免主动肌/拮抗肌配错；
- **控制分配**：末端任务空间 PD → `mj_jacSite` 雅可比转置 → `qfrc_bias` 重力/科氏补偿
  → 最小范数反解 6 块肌肉激活度，并保留**共收缩基量** `c0`（提供关节刚度）；
- **2 维肌肉协同控制**（`synergy.py`）：把 6 维激活投影到 2 个协同基底（并支持数据驱动 NMF 提取），
  演示「中枢只发少量协同指令」的低维控制；
- **共收缩刚度实验**：给末端施加已知探针力，实测稳态位移反推刚度（`measure_stiffness`）；
- 发布 `/joint_states`、`/musculoskeletal/hand_pose`、`/musculoskeletal/activation`、
  `/musculoskeletal/muscle_force`，并用 **`tf2_ros.TransformBroadcaster`** 正确广播 `/tf`
  （`world → musculoskeletal/hand`）；
- 无 ROS 独立运行（`main.py` 交互 / 演示 / 自检 / 素材采集），与 ROS2 节点共用同一仿真核心。

## 环境要求

- Ubuntu 22.04 + ROS2 Humble（或 Windows / 任意 Python 环境直接跑 standalone）
- `pip3 install -r requirements.txt`（`mujoco`、`numpy` 必需；`pyyaml`、`imageio`、`matplotlib` 可选）
- 无显示环境（纯 SSH / 容器）下渲染需 `export MUJOCO_GL=osmesa`

## 安装

```bash
# 方式一：源码直接运行（无需编译）
cd src/humanoid/humanoid_muscle_arm_control
python3 main.py --check

# 方式二：colcon 编译后用 ROS2 启动
cd ~/ros2_ws && colcon build --symlink-install --packages-select humanoid_muscle_arm_control
source install/setup.bash
```

## 运行方式

### 1. standalone（无需 ROS，最快验证）

```bash
python3 main.py              # 键盘交互控制末端目标点
python3 main.py --demo       # 自动演示：初始姿态 → 前伸 → 侧向 → 前下方 → 回位
python3 main.py --check      # 模型自检：力矩臂矩阵 / 主动肌-拮抗肌配对 / 控制回路
python3 main.py --test       # 自动化测试（10 项，含刚度与协同实验）
python3 main.py --capture DIR # 生成文档素材：关键帧、动图、肌肉激活曲线、metrics.json
python3 main.py --synergy    # 用 2 维肌肉协同模式控制
```

### 2. ROS2 launch（Humble）

```bash
ros2 launch humanoid_muscle_arm_control main.launch.py
```

启动后焦点保持在终端，用键盘控制末端；另开终端查看状态：

```bash
ros2 topic echo /joint_states
ros2 topic echo /musculoskeletal/activation
ros2 run tf2_ros tf2_echo world musculoskeletal/hand
```

## 操作键位

| 按键 | 动作 | 说明 |
| --- | --- | --- |
| `W` / `S` | 末端前伸 / 后收 | 世界系 +x / −x |
| `A` / `D` | 末端左移 / 右移 | 世界系 +y / −y |
| `Q` / `E` | 末端上升 / 下降 | 世界系 +z / −z |
| `空格` | 急停 | 目标点保持在当前位置 |
| `R` | 回位 | 目标点回到初始末端位置 |
| `ESC` | 退出 | — |
| 松开按键 | 停止 | 目标点不再移动 |

## ROS2 话题接口

| 话题 | 类型 | 方向 | 说明 |
| --- | --- | --- | --- |
| `/cmd_vel` | geometry_msgs/Twist | 订阅 | 末端目标点速度（linear.x/y/z，m/s） |
| `/joint_states` | sensor_msgs/JointState | 发布 | 肩 pitch / 肩 roll / 肘 pitch 关节角 (rad) |
| `/musculoskeletal/hand_pose` | geometry_msgs/PoseStamped | 发布 | 末端位置（world 系） |
| `/musculoskeletal/activation` | std_msgs/Float64MultiArray | 发布 | 6 块肌肉的激活度 a ∈ [0,1] |
| `/musculoskeletal/muscle_force` | std_msgs/Float64MultiArray | 发布 | 6 块肌肉的肌力 (N) |
| `/tf` | tf2_msgs/TFMessage | 发布 | `world → musculoskeletal/hand`（tf2_ros 广播） |

## 目录结构

```
humanoid_muscle_arm_control/
├── humanoid_muscle_arm_control/
│   ├── sim_core.py              # 仿真核心：模型/力矩臂标定/控制分配/刚度实验（无 ROS）
│   ├── muscle_model.py          # Hill 型肌肉：激活动力学 + 力-长度-速度
│   ├── synergy.py               # 肌肉协同：设计基底 + 投影 + NMF 提取
│   ├── mujoco_sim_node.py       # ROS2 仿真节点（tf2_ros 广播 /tf）
│   └── keyboard_teleop_node.py  # ROS2 键盘遥控节点（3 轴末端速度）
├── models/musculoskeletal_arm.xml   # 自建肌肉骨骼人上肢 MJCF 模型
├── launch/main.launch.py        # 启动文件（注入 config/sim_params.yaml）
├── config/sim_params.yaml       # 仿真/控制器/肌肉/键盘参数（standalone 与 ROS2 共用）
├── test/test_standalone.py      # 10 项自动化测试
├── main.py / main.sh            # 模块主入口与一键脚本
└── README.md
```

## 与已有模块的区别

| 对比项 | `humanoid_keyboard_control`（人形机器人） | 本模块（肌肉骨骼人） |
| --- | --- | --- |
| 驱动方式 | 关节力矩 PD 伺服 | **Hill 型肌肉**（激活动力学 + 力-长度-速度曲线） |
| 控制量 | 关节力矩 | 末端任务空间 → **力矩臂矩阵反解肌肉激活** |
| 主动肌/拮抗肌 | 无此概念 | 每个自由度一对，含跨关节双关节肌 |
| 可调柔顺性 | 无 | **共收缩 c0 → 关节/末端刚度**（可实测） |
| 控制维度 | 3 个关节 | 可进一步压到 **2 维肌肉协同** |
| 参考模型 | MuJoCo 官方 humanoid 设计思路 | 人体上肢肌肉-骨骼几何（Hill 肌肉模型） |

## AI 声明

- 模型几何、Hill 肌肉模型、力矩臂标定与控制分配、协同分析与刚度实验均为本作业原创实现；
- 参考了 MuJoCo 官方文档（`spatial` 腱 / `mj_jacSite` / `mj_applyFT`）与仓库内已有模块
  （`humanoid_keyboard_control`、`rov_mujoco`）的 ROS2 包结构惯例；
- 代码与文档在 AI 助手协助下完成，实测数据均由 `main.py --capture` / `--test` 在本机仿真中生成。
