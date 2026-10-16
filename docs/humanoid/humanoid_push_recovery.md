# 人形机器人平衡控制与外力抗扰恢复（humanoid_push_recovery）

任务「人形机器人物理仿真」下的**稳定性与抗扰恢复**方向实现：自建双足人形 MJCF 模型的
底盘 freejoint **完全自由**（不做速度锁定），机器人依靠双脚与地面的真实接触以及关节
力矩站立；模块实现 **ZMP / 捕获点稳定性判据** 与 **踝策略 + 髋策略** 的恢复控制律，
并定量给出「扰动力 → 最大质心偏移 → 恢复 / 跌倒」的完整恢复包线。

与同分类其它人形模块的区别（下列模块在各自的 PR 中提交，**尚未合并到本分支**，
为避免死链此处只做文字对比、不提供跳转）：

| 模块 | 输入 | 关注点 |
|---|---|---|
| humanoid_keyboard_control | 键盘速度指令 | 指令执行准确性（运动学层面） |
| humanoid_gait_control | 步态周期 | 行走步态生成与跟踪 |
| **humanoid_push_recovery（本模块）** | **外力扰动** | **外力下的稳定性判据与恢复能力** |

本模块的机器人**不是**靠改写 freejoint 速度「站住」的，而是真实受力平衡——这一点是
与键盘控制模块的本质区别，也是能够研究「被推是否跌倒」的前提。

## 1. 系统架构

```text
        ┌──────────────── 外力扰动 ────────────────┐
        │  /balance/push (Vector3) 或 keyboard      │
        ▼                                          │
   torso: data.xfrc_applied (6 维力/力矩)           │
        │                                          │
        ▼                                          │
   站姿 PD (15 关节) ──► 关节力矩 ──► MuJoCo 物理引擎 │
        ▲                                          │
        │  关节目标修正                             │
   ┌────┴─────────────────────────┐                │
   │ 平衡控制器 BalanceController  │                │
   │ · 踝策略：质心偏移/速度 → 踝力矩 │                │
   │ · 髋策略：角动量 → 髋力矩       │                │
   └────▲─────────────────────────┘                │
        │                                          │
   ┌────┴─────────────────────────┐                │
   │ 稳定性判据                    │◄───────────────┘
   │ · 支撑多边形（脚底 8 角点凸包）│
   │ · ZMP（LIPM 近似）            │
   │ · 捕获点 ξ = x + v/√(g/h)     │
   └──────────────────────────────┘
```

## 2. 依赖安装与运行

### 2.1 环境与依赖安装

```shell
# 1) 安装 ROS2 Humble（Ubuntu 22.04）：
#    参考官方文档 https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

# 2) 安装 Python 依赖（在**仓库根目录**执行；mujoco / numpy / pyyaml）
pip3 install -r src/humanoid/humanoid_push_recovery/requirements.txt

# 3) 可视化窗口依赖（仅 Linux 桌面需要；若提示缺少 GLFW 再执行）
sudo apt install -y libglfw3
```

ROS 依赖（`rclpy`、`geometry_msgs`、`sensor_msgs`、`std_msgs`、`tf2_ros`、
`ros2launch`）在 `colcon build` 时自动解析，无需手动安装。

### 2.2 源码放入工作空间与构建

```shell
mkdir -p ~/ros2_ws/src
cp -r src/humanoid/humanoid_push_recovery ~/ros2_ws/src/

cd ~/ros2_ws
colcon build --symlink-install --packages-select humanoid_push_recovery
source install/setup.bash
```

> 注意：执行完本节后当前目录是 `~/ros2_ws`，而本包被拷贝到
> `~/ros2_ws/src/humanoid_push_recovery`（**不含 `humanoid/` 这一层**）。

### 2.3 standalone（无需 ROS，最快验证）

standalone 模式**不依赖 ROS，也不要求先 colcon build**。下面的路径以「仓库克隆在
`~/ros2`」为例，请替换成自己的实际路径：

```shell
cd ~/ros2/src/humanoid/humanoid_push_recovery   # ← 换成你自己的仓库路径
python3 main.py --check          # 模型/判据/控制器 自检
python3 main.py                  # 演示：一次推扰 + 平衡恢复（打开可视化窗口）
python3 main.py --force 180      # 指定扰动力 (N)
python3 main.py --direction 90   # 扰动方向（0=前向, 90=侧向, 180=后向）
python3 main.py --no-hip         # 关闭髋策略（仅踝策略，对照用）
python3 main.py --sweep          # 测定各方向临界冲量并与策略对照
python3 main.py --capture assets_out  # 采集关键帧/动图/曲线/metrics.json
python3 main.py --test           # 自动化测试（31 项）
```

若运行的是 2.2 节拷贝到工作空间的那一份，路径为
`~/ros2_ws/src/humanoid_push_recovery`（同样没有 `humanoid/` 这一层）。

无显示环境（SSH / 容器）下渲染需要 EGL：

```shell
export MUJOCO_GL=egl
```

### 2.4 ROS2 Humble launch

```shell
cd ~/ros2_ws
colcon build --symlink-install --packages-select humanoid_push_recovery
source install/setup.bash
ros2 launch humanoid_push_recovery main.launch.py
```

焦点保持在启动终端，按键盘施加扰动：

| 按键 | 动作 |
|---|---|
| `W` / `S` | 向前 / 向后推 |
| `A` / `D` | 向左 / 向右推 |
| `Q` / `E` | 左前 / 右后斜向推 |
| `+` / `-` | 推扰力度 ±20 N |
| `空格` | 复位 |
| `ESC` | 退出 |

另开终端查看状态：

```shell
ros2 topic echo /balance/status            # stable / recovering / fallen
ros2 topic echo /balance/stability_margin  # 捕获点到支撑边界的稳定裕度 (m)
ros2 topic echo /balance/zmp               # 零矩点
ros2 run tf2_ros tf2_echo world humanoid/base_link   # 验证 TF 可读取
```

### 2.5 可视化窗口（看到人形机器人界面）

standalone 演示与 `ros2 launch` **默认都会打开 MuJoCo 可视化窗口**，窗口内可实时看到
机器人被推后的位移与姿态变化，相机自动跟踪躯干。

| 运行环境 | 能否看到窗口 | 说明 |
|---|---|---|
| 本机 Ubuntu 桌面 / 虚拟机桌面终端 | ✅ | 直接运行即可 |
| SSH 远程登录 | 需 X11 转发 | 用 `ssh -X user@host` 登录，或先 `export DISPLAY=:0` |
| 无显示服务器（纯命令行 / CI） | ❌ | standalone 加 `--no-viewer`；ROS2 用 `visualize:=false` |

无图形界面时程序会打印提示并自动退化为无窗口模式，**物理仿真、话题发布与自动化测试不受影响**。

## 3. 实现要点

### 3.1 自建双足模型（models/humanoid_balance.xml）

15 关节 = 双腿（髋 roll / 髋 pitch / 膝 pitch / 踝 pitch / 踝 roll）×2 + 双臂
（肩 pitch / 肘 pitch）×2 + 颈 ×1，全部由 `motor` 驱动。与同分类其它模块自建双足
模型的关键区别：

- **底盘 freejoint 不做速度锁定**：机器人的站立完全由双脚与地面的接触力和关节力矩决定；
- 脚掌前后长度加大到 0.22 m，以获得更大的踝关节力矩权限；
- 脚掌与地面摩擦单独加大（`friction="1.4 0.7 0.2"`）；
- 外力扰动施加在 `torso` body 上（`data.xfrc_applied`）。

### 3.2 站姿 PD 控制

每个关节的驱动力矩为 PD 控制，目标姿态为直立位（全 0）：

\[
\tau_i = k_p (q_i^{ref} - q_i) - k_d \dot{q}_i,\qquad
i \in \{\text{髋/膝/踝/肩/肘/颈}\}
\]

其中 `kp=120`、`kd=6`；髋/踝 roll 关节取 `kp_roll=400` 抗侧倾；力矩限幅
`±300 N·m`。**只有站姿 PD 时机器人会因倒立摆本质而逐渐前倾并跌倒**，这正是需要
平衡控制器的原因（见 5.2 节对照实验）。

### 3.3 稳定性判据

- **支撑多边形**：取双脚底面 8 个角点，用 Andrew 单调链求凸包；
- **ZMP（零矩点）**：采用线性倒立摆近似
  \[
  \mathrm{ZMP} = \mathrm{CoM}_{xy} - \frac{z_{\mathrm{com}}}{g}\,\ddot{\mathrm{CoM}}_{xy}
  \]
  质心加速度由质心速度的有限差分求得（并做一阶低通抑制差分噪声）；
- **捕获点**：\(\xi = \mathrm{CoM}_{xy} + \mathbf{v}_{\mathrm{com}}/\omega\)，
  \(\omega=\sqrt{g/h}\)；**稳定裕度**定义为捕获点到支撑多边形边界的有符号距离，
  裕度 > 0 表示仅靠踝/髋策略仍可停下，< 0 表示必须跨步。

> 实现注记：实测在 MuJoCo 3.16 中 `data.subtree_linvel[0]` 恒为 0，
> 因此质心速度统一由 `data.subtree_com[0]` 的有限差分求得（见 `sim_core.py`）。

### 3.4 恢复控制律：踝策略 + 髋策略（输出关节力矩）

平衡控制器输出的是**关节力矩修正量**（N·m），而不是关节目标角度。原因：踝 pitch
关节的实际行程只有 ±0.9 rad，若用「关节目标」表达平衡修正，为获得足够的恢复力矩
必须下发远超行程的目标（例如 3 rad ≈ 172°），既不合理、又会立刻被关节限位截断，
物理含义模糊。改用**力矩量纲**后，限幅直接表示关节力矩权限，整定也更有依据。

**踝策略（主力）**：以质心相对参考位置的偏移与质心速度为反馈，给出踝 pitch / roll
的恢复力矩：

\[
\tau^{\mathrm{ankle}} = k_p^{a}\,(x_{\mathrm{com}}-x_{\mathrm{ref}}) + k_d^{a}\,v_{\mathrm{com},x},
\qquad |\tau^{\mathrm{ankle}}| \le \tau^{a}_{\max}
\]

**髋策略（增强）**：踝策略饱和后，通过摆动上半身产生角动量继续把质心拉回：

\[
\tau^{\mathrm{hip}} = -k_p^{h}\,(x_{\mathrm{com}}-x_{\mathrm{ref}}) - k_d^{h}\,v_{\mathrm{com},x},
\qquad |\tau^{\mathrm{hip}}| \le \tau^{h}_{\max}
\]

力矩由 `sim_core.step()` 叠加到站姿 PD 力矩上，再一并限幅到 `torque_limit`。
（髋策略可开关：`hip_enable` / `--no-hip`，用于「仅踝 vs 踝+髋」对照实验。）

### 3.5 符号约定与整定（踩过的坑，供复现参考）

控制律的**符号若写反，会形成正反馈直接把机器人推倒**，因此模块把符号自检写进了
自动化测试（测试项 7，四个方向逐一标定）。实测符号约定（均指**正力矩**）：

| 关节 | 正力矩作用下质心移动方向 | 控制律符号 |
|---|---|---|
| 踝 pitch | **向后**（−x） | `+(k_p e_x + k_d v_x)` |
| 踝 roll | **向 +y（左）** | `−(k_p e_y + k_d v_y)` |
| 髋 pitch | 向前（+x） | `−(k_p e_x + k_d v_x)` |
| 髋 roll | 向 −y（右） | `+(k_p e_y + k_d v_y)` |

另一个关键整定点：**侧向 roll 关节的力矩权限远大于 pitch**。早期实现用「关节目标」
表达控制量时，沿用 pitch 的增益（等价于约 300 N·m 的滚转力矩）会把机器人
**抛离地面**（根高从 1.084 m 跳到 1.203 m）而立刻跌倒；改为力矩量纲后，侧向单独整定
并单独限幅（`kp_ankle_roll=4800 N·m/m`、`kd_ankle_roll=600 N·m·s/m`、
`ankle_roll_torque_limit=180 N·m`，约为 pitch 的 60%）。

### 3.6 能力边界（重要）

- 本模块的恢复能力**完全来自踝/髋关节力矩，未实现跨步恢复（stepping）**。
  当扰动超过临界冲量时机器人会跌倒——这是**由脚掌几何与关节力矩权限决定的物理上限**，
  不是控制器参数没调好。以踝策略的静态权限估算：踝关节可提供的最大恢复力矩
  约为 \(F_z \cdot l_{\text{foot}} \approx 35 \times 9.81 / 2 \times 0.11 \approx 187\ \mathrm{N\cdot m}\)
  每只脚，而倒立摆的倾覆力矩为 \(mg\,x_{\mathrm{com}}\)，由此给出的可恢复质心偏移
  已接近实测临界值；
- 模型为自建简化双足，连杆长度、质量分布与摩擦均与真实人形机器人不同，
  临界冲量的**绝对值不可外推到实机**，但「扰动力—偏移—跌倒」的定性规律与
  策略间的**相对提升**具有参考意义；
- 评价指标只描述**本仿真模型在上述参数下的行为**，不代表硬件实验结论。

## 4. 效果图

推扰—恢复过程的自动演示（150 N 前向推扰，场景第三视角，关键帧）：

| 静立站姿 | 被推瞬间 | 恢复中 | 恢复稳定 |
|---|---|---|---|
| ![](../img/humanoid/humanoid_push_recovery/scene_stand.png) | ![](../img/humanoid/humanoid_push_recovery/scene_push.png) | ![](../img/humanoid/humanoid_push_recovery/scene_recover.png) | ![](../img/humanoid/humanoid_push_recovery/scene_settled.png) |

完整动画：![](../img/humanoid/humanoid_push_recovery/push_recovery_demo.gif)

恢复过程曲线（质心偏移 / 躯干倾角 / 踝关节指令）：

![](../img/humanoid/humanoid_push_recovery/recovery_curve.png)

扰动力 → 最大质心偏移曲线（×标记为跌倒）：

![](../img/humanoid/humanoid_push_recovery/critical_impulse.png)

## 5. 验证结果

以下数值全部由 `python3 main.py --sweep` 与 `python3 main.py --test` 在本机实测
（Ubuntu 22.04 + MuJoCo 3.16，CPU 推理）。

### 5.1 站立稳定性

| 指标 | 结果 |
|---|---|
| 无扰动自由站立（8 s） | 不跌倒，质心漂移 **0.59 mm**，根高 1.084 m |
| 躯干倾角（静立） | 1.53° |
| 静立稳定裕度（捕获点到支撑边界） | **93.1 mm** |
| 支撑多边形尺寸 | 前后 220 mm × 侧向 460 mm |

### 5.2 抗扰恢复（对照实验）

**关闭平衡控制器时，120 N×(0.08 s) 推扰即跌倒**（最大质心偏移 373 mm）——
说明本模型的站立**不是**靠锁定底盘维持的，平衡控制器是必需的。

开启平衡控制器后：

| 扰动 | 最大质心偏移 | 恢复时间 |
|---|---|---|
| 前向 120 N (9.6 N·s) | **37.8 mm** | **0.91 s** |
| 侧向 150 N (12.0 N·s) | **34.3 mm** | **0.67 s** |
| 斜向 45° 120 N (9.6 N·s) | 14.7 mm | 0.59 s |

恢复后质心残差 0.11 mm（判据：偏移 < 5 mm 且速度 < 0.02 m/s）。

### 5.3 临界冲量（二分法测定，扰动持续 0.08 s）

| 方向 | 仅踝策略 | 踝+髋策略 | 提升 |
|---|---|---|---|
| 前向 (+x) | 152.5 N / **12.20 N·s** | 172.7 N / **13.82 N·s** | **+13.2%** |
| 侧向 (+y) | 227.2 N / **18.17 N·s** | 285.3 N / **22.82 N·s** | **+25.6%** |
| 后向 (−x) | 42.7 N / 3.42 N·s | 44.8 N / 3.58 N·s | +4.8% |

**结论与分析**：

1. **踝策略是站立抗扰的主力**，髋策略在不改变任何硬件参数的前提下把可恢复冲量
   提升 13.2%（前向）/ 25.6%（侧向），同时降低了最大偏移与恢复时间；
2. **侧向抗扰能力显著强于前向**（18.17 vs 12.20 N·s），因为双足站立时侧向支撑宽度
   （460 mm，靠双足共同支撑）大于前后支撑长度（220 mm，靠单脚脚掌）；
3. **后向抗扰最弱**（3.42 N·s），因为脚掌在踝关节前方的延伸（0.12 m）明显长于后方
   （0.07 m），向后倾覆时可用的踝关节力矩权限最小——这解释了为什么人从背后被推更容易
   向前扑倒；
4. 超过临界冲量后必然跌倒，属于 3.6 节所述**物理上限**；要进一步提升需要引入
   **跨步恢复（stepping）**，这是本模块明确未实现的边界。

### 5.4 仿真性能

| 指标 | 结果 |
|---|---|
| 仿真吞吐 | 17,779 step/s |
| 实时因子 | 35.6× |

## 6. 参考

* [人形机器人基础](./humanoid_basics.md)
* 本包 ROS2 包结构（`package.xml` / `setup.py` / `launch` / `config`）参照仓库内
  既有模块的 ament_python 惯例
* [MuJoCo 文档：freejoint / xfrc_applied / Renderer](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
* 线性倒立摆与捕获点：Kajita et al., *Biped Walking Pattern Generation by using Preview
  Control of Zero-Moment Point*, ICRA 2003；Pratt et al., *Capture Point: A Step toward
  Humanoid Push Recovery*, Humanoids 2006

## 7. AI 声明

本模块的 MJCF 模型、平衡判据实现、踝/髋控制律、参数整定流程、ROS2 节点与评测脚本
均在 AI 助手协助下完成，代码与文档由本人负责；本文所有实测数字均由本机运行
`main.py --sweep` / `main.py --test` 得到，未做任何未运行的声明。
