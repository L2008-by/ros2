# humanoid_gait_control —— 双足步态生成与行走控制

基于 MuJoCo 的人形机器人双足步行模块：**LIPM 线性倒立摆步态规划 + 腿部逆运动学
+ 关节空间 PD 伺服**，让自建双足模型在仿真中稳定交替迈步行走。

## 快速开始

```bash
pip install -r requirements.txt

python3 main.py --check    # 模型与步态自检
python3 main.py --test     # 自动化测试
python3 main.py --demo     # 自动行走 8 步并打印实测指标
python3 main.py --capture docs_img  # 采集关键帧/动图/metrics.json
```

## 模块结构

```
humanoid_gait_control/
├── models/humanoid_gait.xml      # 自建双足人形 MJCF（16 驱动关节）
├── humanoid_gait_control/
│   ├── gait_planner.py           # LIPM 步态规划（质心/落脚点/摆动腿轨迹）
│   ├── leg_ik.py                 # 腿部平面三连杆解析 IK（+ 侧向 roll）
│   ├── sim_core.py               # MuJoCo 仿真核心（IK→PD→mj_step）
│   ├── mujoco_sim_node.py        # ROS2 仿真节点（JointState/Odometry）
│   └── gait_teleop_node.py       # ROS2 键盘遥控节点（W/S 调步速）
├── config/sim_params.yaml        # 步态与控制参数
├── launch/main.launch.py         # ros2 launch 入口
├── test/test_standalone.py       # 自动化测试（无 ROS）
└── main.py                       # standalone 入口
```

## 方法

1. **LIPM 质心轨迹**：支撑脚为倒立摆支点，质心高度恒定 `h`，动力学
   `x'' = (g/h)·x`；单步周期对称边界条件给出初态 `(x0, v0)`，质心每步
   从 `-s/2` 走到 `+s/2`，速度回归初值，形成周期步态，平均速度 `s/T`。
2. **落脚点序列**：左右脚沿前进方向交替，横向间距 `step_width`。
3. **摆动腿轨迹**：三次 Hermite 插值（起止速度为零）+ 抛物线抬脚，
   脚跟着地无冲击。
4. **腿部 IK**：髋/膝/踝平面三连杆解析解，踝角补偿使脚底水平；
   侧向由髋 roll / 踝 roll 保持骨盆水平、脚底着地。
5. **关节 PD**：力矩 `τ = Kp·(q_des − q) − Kd·q̇`，侧向 roll 关节加强增益；
   双臂与支撑腿反相摆动抵消角动量。

## 实测指标（8 步默认参数）

| 指标 | 结果 |
|---|---|
| 行走 8 步 | 不跌倒 |
| 总位移 | ≈ 1.76 m（计划 1.76 m） |
| 平均速度 | ≈ 0.275 m/s（理论 0.275 m/s） |
| 躯干倾角 | 全程 < 12° |
| 仿真性能 | 见 main.py --demo 实测 |

## ROS2

```bash
ros2 launch humanoid_gait_control main.launch.py
# 话题：cmd_vel (Twist, vx=步速)、joint_states (JointState)、odom (Odometry)
```

## 参考

* [人形机器人基础](../../docs/humanoid/humanoid_basics.md)
* [人形机器人肌肉驱动上肢仿真与键盘运动控制](../../docs/humanoid/humanoid_muscle_arm_control.md)
* MuJoCo 官方 humanoid 模型结构、ROBOTIS OP3 关节配置（github.com/google-deepmind/mujoco_menagerie）
