# 人形机器人双足步态行走控制（humanoid_gait_control）

任务「人形机器人双足步态行走」的 MuJoCo 实现：在自建双足人形 MJCF 模型上实现
**LIPM（线性倒立摆）质心步态规划 + 腿部解析 IK + 关节 PD 伺服**的完整行走控制链，
机器人可在仿真中稳定站立、起步并完成连续步态行走，同时发布里程计与关节状态话题。

## 1. 运行架构

```text
步态规划器 ──LIPM 质心轨迹/落脚点──► 腿部解析 IK ──关节角目标──► 关节 PD 伺服
 (gait_planner)   (2D/3D 解析解)     (leg_ik)        (力矩 = Kp·Δq − Kd·q̇)
                                                                  │
                                          ┌───────────────────────┘
                                     MuJoCo 物理引擎（人形机器人模型）
                                          │
                                    odom / joint_states / 渲染
```

- **gait_planner**：LIPM 闭式解规划质心轨迹与落脚点序列（起步 S 曲线、双支撑过渡、hold 收步）。
- **leg_ik**：腿部二维 IK（髋/膝）+ 三维侧向 IK（髋/踝 roll + 脚平约束），解析求解。
- **sim_core**：关节 PD 力矩伺服、骨盆位置闭环（纵向/侧向独立增益与限幅）、
  LIPM 目标速度前馈、摆动腿足端轨迹跟踪、摆臂与躯干姿态。
- **mujoco_sim_node / gait_teleop_node**：ROS2 节点，将步态控制接入 ROS2 话题。

## 2. 控制框图

| 模块 | 职责 | 关键实现 |
|---|---|---|
| LIPM 规划 | 质心 x/y 轨迹 + 落脚点 | `x(t)=x0·cosh(t/Tc)+Tc·v0·sinh(t/Tc)`，`Tc=√(h/g)` |
| 腿部 IK | 踝→髋的关节角解析解 | 2D：余弦定理求髋/膝；3D：roll 由侧向偏移求 |
| PD 伺服 | 关节力矩 | `τ = Kp·(q_des − q) − Kd·q̇`，限幅 300 N·m |
| 位置闭环 | 骨盆偏差修正 | 纵向 KP=0.7/限幅 0.12 m，侧向 KP=2.0/限幅 0.08 m |
| 速度前馈 | 补偿滞后 | `com += K·v_lipm`（K=0.15 s） |

## 3. 编译与运行

### 3.1 standalone（无需 ROS，最快验证）

```shell
cd src/humanoid/humanoid_gait_control
python3 main.py --demo     # 自动演示「准备 → 行走 4 步」
python3 main.py --check    # 模型/步态自检
python3 main.py --test     # 自动化测试（20 项）
python3 main.py --capture ../img_dir   # 生成关键帧/动图/metrics
```

### 3.2 ROS2 Humble launch

```shell
cd ~/ros2_ws && colcon build --symlink-install --packages-select humanoid_gait_control
source install/setup.bash
ros2 launch humanoid_gait_control main.launch.py
```

另开终端可查看状态：

```shell
ros2 topic echo /humanoid/odom
ros2 topic echo /joint_states
```

## 4. 实现要点

### 4.1 自建双足模型（models/humanoid_gait.xml）

双腿总长 0.98 m（大腿 0.52 m、小腿 0.46 m），站立高约 1.05 m，髋部带 ±0.16 m 侧向偏移，
踝中心相对脚底 0.055 m 抬高；关节限位：髋 pitch ±1.7、膝 [-2.5, 1.5]、踝 pitch ±0.9、
roll ±0.5/0.6 rad。全部关节由 `motor` 驱动。

### 4.2 LIPM 质心步态规划

- 单步周期 1.2 s、步长 0.10 m（平均速度 0.083 m/s），周期对称初态
  `x0 = −s/2`、`v0 = s/(2·Tc·sinh(T/Tc))`，一步后质心前进 s 且速度回归初值。
- **起步**：第 0 步用 S 曲线把质心从 0 平滑推进到 +s/2，避免直接落入 LIPM 起点 −s/2 造成的"先退后冲"。
- **双支撑过渡**：步初/步末各 20% 双脚支撑期，质心侧向在双脚间平滑过渡；
- **hold 收步**：计划走完后质心平滑收回最后支撑脚上方，回到静态平衡。

### 4.3 腿部解析 IK 与 PD 伺服

- 二维 IK：已知髋与踝目标，用余弦定理解析求髋 pitch 与膝（取人类式"后弯"解），踝 pitch 由脚平约束补齐。
- 三维侧向：髋/踝 roll 由踝相对髋的侧向偏移解析求得，脚底保持水平。
- 关节 PD：`τ = Kp·(q_des − q) − Kd·q̇`，Kp=400、Kd=15，侧向 roll 加强（Kp=800、Kd=18）。
- 骨盆位置闭环抑制倒立摆漂移：纵向/侧向独立增益，限幅保证 IK 不越出腿工作空间。

## 5. 仿真演示

| 场景 | 说明 | 渲染 |
|---|---|---|
| 准备段 | 双脚并立、重心稳定 | ![准备段](../../img/humanoid/humanoid_gait_control/scene_ready.png) |
| 第 1 步 | 摆动腿抬脚离地 | ![第 1 步](../../img/humanoid/humanoid_gait_control/scene_step1.png) |
| 第 3 步 | 交替迈步中 | ![第 3 步](../../img/humanoid/humanoid_gait_control/scene_step3.png) |
| 周期步态 | 连续行走 | ![周期步态](../../img/humanoid/humanoid_gait_control/scene_walk.png) |

完整演示动图：![演示动图](../../img/humanoid/humanoid_gait_control/gait_demo.gif)

## 6. 实测指标（standalone --demo，本地 MuJoCo 渲染）

```text
行走结果：成功（准备 → 4 步 → 收步）
总位移 0.22 m（计划 0.40 m） | 平均速度 0.043 m/s
躯干倾角 均值 6.79° 最大 15.10° 标准差 2.96°
自动化测试：20 / 20 通过（LIPM 解析/IK 往返/站姿/4 步行走/位移/速度/倾角）
```

> 注：当前默认演示为 4 步稳定行走；更长步数（>4 步）下质心跟踪滞后会放大，
> 表现为步态中后段躯干前倾加剧，属已知限制，控制链正在持续改进中。

## 7. 已知限制与改进方向

- 长距离行走的质心跟踪滞后：位置闭环限幅受腿工作空间约束，超过 4 步后滞后累积、
  躯干前倾增大。改进方向：落脚点随质心状态在线重规划（步态反馈）、
  踝力矩 ZMP 控制替代纯位置伺服。
- 静态站姿存在 <7 cm/2 s 的轻微漂移（接触数值噪声 + 姿态微偏量级）。
- 当前步速较低（0.083 m/s），快速步态需引入质心高度波动与弹性储能建模。
