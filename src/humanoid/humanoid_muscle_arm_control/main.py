#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肌肉骨骼人（肌肉驱动）MuJoCo 仿真与键盘运动控制 —— 主入口
=========================================================

课程要求：入口为 ``main.py``，无需 ROS 即可运行整个模块。

运行模式
--------
1. 交互模式（默认）——键盘实时控制末端目标点::

       W/S 前伸/后收    A/D 左移/右移    Q/E 上升/下降    空格急停    ESC 退出

2. ``--demo``    ——自动演示「初始姿态 → 前伸 → 外展 → 前下方 → 回位」并录制素材；
3. ``--capture DIR`` ——在 ``--demo`` 基础上把关键帧、动图、肌肉激活分析与
   ``metrics.json``（实测指标）写入目录，用于生成文档配图；
4. ``--check``   ——模型自检：关节/肌肉/力矩臂/主动肌-拮抗肌配对/协同基底一致性；
5. ``--test``    ——运行 ``test/test_standalone.py``（站姿无关的肌肉-控制指标测试）；
6. ``--ros2``    ——通过 ``ros2 launch`` 启动仿真节点与键盘遥控节点。

示例::

    python3 main.py                      # 交互控制
    python3 main.py --capture ../docs_img # 生成文档素材
    python3 main.py --check              # 模型与控制自检
    python3 main.py --test               # 自动化测试
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from humanoid_muscle_arm_control.sim_core import (          # noqa: E402
    MusculoskeletalArm, ArmParams, load_params, JOINT_NAMES, MUSCLE_NAMES, HOME_QPOS)
from humanoid_muscle_arm_control import synergy as synergy_lib  # noqa: E402

#: 键盘按键 → 世界系速度方向（x 前 / y 左 / z 上）
KEY_VECTORS = {
    'w': (1, 0, 0), 'W': (1, 0, 0), 's': (-1, 0, 0), 'S': (-1, 0, 0),
    'a': (0, 1, 0), 'A': (0, 1, 0), 'd': (0, -1, 0), 'D': (0, -1, 0),
    'q': (0, 0, 1), 'Q': (0, 0, 1), 'e': (0, 0, -1), 'E': (0, 0, -1),
    'up': (1, 0, 0), 'down': (-1, 0, 0), 'left': (0, 1, 0), 'right': (0, -1, 0),
}

HELP_TEXT = """
============================================================
  肌肉骨骼人手臂（肌肉驱动）键盘运动控制
------------------------------------------------------------
  W / S     末端前伸 / 后收        A / D     末端左移 / 右移
  Q / E     末端上升 / 下降        空格       急停（保持目标点）
  R         目标点回到初始位置      ESC       退出
  松开按键 = 停止；末端由 6 块 Hill 型肌肉驱动
============================================================
"""

#: 演示序列：(帧名, 帧说明, 目标点来源)
DEMO_SEQUENCE = (
    ("scene_home", "初始姿态（肘屈曲，肌肉处于静息激活）", "home"),
    ("scene_forward", "前伸协同：末端抵近绿色目标球", "target_forward"),
    ("scene_lateral", "肩外展：末端抵近蓝色目标球", "target_lateral"),
    ("scene_low", "屈曲内收协同：末端下探至前下方橙色目标球", "target_low"),
)


# ----------------------------------------------------------------------
# 跨平台按键读取
# ----------------------------------------------------------------------
class KeyReader:
    """Linux 用 termios raw 模式；Windows 用 msvcrt 非阻塞读取。"""

    def __init__(self):
        self._platform = sys.platform
        self._fd = None
        self._old = None
        if self._platform.startswith("linux"):
            import termios
            import tty
            try:
                self._fd = os.open("/dev/tty", os.O_RDONLY)
            except OSError:
                self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)

    def get_key(self, timeout=0.03):
        if self._platform.startswith("linux"):
            import select
            import termios
            ready, _, _ = select.select([self._fd], [], [], timeout)
            if not ready:
                return None
            char = os.read(self._fd, 1)
            if not char:
                return None
            try:
                return char.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if self._platform.startswith("win"):
            import msvcrt
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ('\x00', '\xe0'):
                    second = msvcrt.getwch()
                    return {'H': 'up', 'P': 'down', 'K': 'left', 'M': 'right'}.get(second, second)
                return char
            return None
        return None

    def close(self):
        if self._fd is not None and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            if self._fd != sys.stdin.fileno():
                os.close(self._fd)


# ----------------------------------------------------------------------
# 素材输出
# ----------------------------------------------------------------------
def save_png(image, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, image)
        return True
    except ImportError:
        pass
    try:
        from PIL import Image
        Image.fromarray(image).save(path)
        return True
    except ImportError:
        print(f"[WARN] 未安装 imageio/PIL，无法保存 {path}")
        return False


def save_gif(frames, path, fps=10):
    if not frames:
        return False
    try:
        import imageio
    except ImportError:
        print(f"[WARN] 未安装 imageio，跳过动图: {path}")
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"[INFO] 演示动图已保存: {path} ({len(frames)} 帧, "
          f"{os.path.getsize(path) / 1024:.0f} KB)")
    return True


def save_activation_plot(history, coefficients, path):
    """画肌肉激活曲线 + 2 维协同系数曲线（缺 matplotlib 时跳过）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 未安装 matplotlib，跳过肌肉激活曲线图")
        return False
    data = np.asarray(history, dtype=float)                       # (T, 6)
    coef = np.asarray(coefficients, dtype=float)                  # (T, 2)
    time_axis = np.arange(data.shape[0]) * 0.01                   # 记录频率 100 Hz
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)
    for i, name in enumerate(MUSCLE_NAMES):
        axes[0].plot(time_axis, data[:, i], linewidth=1.2, label=name)
    axes[0].set_ylabel("muscle activation a")
    axes[0].set_title("Muscle activation (Hill model + activation dynamics)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    for k, name in enumerate(("synergy 1 (reach)", "synergy 2 (flex-adduct)")):
        axes[1].plot(time_axis, coef[:, k], linewidth=1.5, label=name)
    axes[1].set_ylabel("synergy coefficient")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Projection onto 2 muscle synergies")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[INFO] 肌肉激活/协同曲线已保存: {path}")
    return True


# ----------------------------------------------------------------------
# 模式 1：交互
# ----------------------------------------------------------------------
def run_interactive(arm, params):
    reader = KeyReader()
    print(HELP_TEXT)
    print(f"[INFO] 仿真已启动 | 物理步长 {arm.model.opt.timestep * 1000:.1f} ms | "
          f"肌肉 {len(MUSCLE_NAMES)} 块 | 末端速度 {params.hand_speed:.2f} m/s")
    if params.use_synergy:
        print("[INFO] 当前为 2 维肌肉协同控制模式")
    home = arm.hand_position().copy()
    last_print = 0.0
    try:
        while True:
            key = reader.get_key()
            if key is not None:
                if key == '\x1b':
                    break
                if key in ('r', 'R'):
                    arm.set_target(home)
                elif key in KEY_VECTORS:
                    arm.set_target(arm.target() +
                                   np.array(KEY_VECTORS[key], dtype=float) *
                                   params.hand_speed * arm.model.opt.timestep)
            arm.step()

            if arm.data.time - last_print >= 1.0:
                last_print = arm.data.time
                state = arm.state()
                active = np.argsort(state['activation'])[::-1][:3]
                top = ", ".join(f"{MUSCLE_NAMES[i]} {state['activation'][i]:.2f}" for i in active)
                print(f"\r[{state['time']:6.1f}s] 末端=({state['hand'][0]:+.3f}, "
                      f"{state['hand'][1]:+.3f}, {state['hand'][2]:+.3f}) "
                      f"误差={state['hand_error'] * 1000:5.1f} mm | 激活前列: {top}",
                      end='', flush=True)
            time.sleep(arm.model.opt.timestep)
    finally:
        reader.close()
    print("\n[INFO] 交互结束。")


# ----------------------------------------------------------------------
# 模式 2/3：演示与素材采集
# ----------------------------------------------------------------------
def run_demo(arm, params, capture_dir=None, gif_path=None, verbose=True):
    """自动演示动作序列；``capture_dir`` 不为空时输出关键帧、动图与分析。"""
    dt = arm.model.opt.timestep
    home = arm.hand_position().copy()
    frames = {}
    gif_frames = []
    activation_history = []
    coefficients = []
    metrics = {"sequence": [], "model": {
        "joints": list(JOINT_NAMES), "muscles": list(MUSCLE_NAMES),
        "timestep": dt, "home_qpos": HOME_QPOS.tolist()}}

    if verbose:
        print("=" * 60)
        print("  肌肉骨骼人手臂演示：初始姿态 → 前伸 → 外展 → 前下方 → 回位")
        print("=" * 60)

    for frame_name, description, source in DEMO_SEQUENCE:
        target = home if source == "home" else arm.landmarks[source].copy()
        arm.set_target(target)
        start_time = arm.data.time
        settle_start = None
        while arm.data.time - start_time < 3.0:                 # 最多 3 s 收敛
            arm.step()
            state = arm.state()
            if capture_dir and len(gif_frames) < 200 and arm.step_count % 32 == 0:
                # 降采样到 320×240，控制动图体积（文档里够用）
                gif_frames.append(arm.render(camera="overview")[::2, ::2])
            if state['hand_error'] < 0.010 and settle_start is None:
                settle_start = arm.data.time
            if settle_start is not None and arm.data.time - settle_start > 0.30:
                break                                                   # 稳定 0.3 s
        state = arm.state()
        metrics["sequence"].append({
            "frame": frame_name, "target": list(np.asarray(target, dtype=float)),
            "hand": list(state['hand']), "error_mm": state['hand_error'] * 1000.0,
            "settle_time_s": float(arm.data.time - start_time),
            "activation": list(state['activation'])})
        if capture_dir:
            frames[frame_name] = arm.render(camera="overview")
        if verbose:
            print(f"[{frame_name}] {description} | 末端 "
                  f"({state['hand'][0]:+.3f}, {state['hand'][1]:+.3f}, {state['hand'][2]:+.3f}) "
                  f"| 稳态误差 {state['hand_error'] * 1000:5.2f} mm "
                  f"| 用时 {arm.data.time - start_time:.2f} s")

    # 回位并记录一路的激活（用于协同分析）
    arm.set_target(home)
    for _ in range(int(1.5 / dt)):
        arm.step()
        if arm.step_count % 10 == 0:                            # 100 Hz 记录
            activation_history.append(arm.state()['activation'])
            coefficients.append(arm.state()['synergy'])

    if activation_history:
        rmse, explained = synergy_lib.reconstruction_error(activation_history)
        metrics["synergy"] = {"rmse": rmse, "explained_variance": explained,
                              "basis": synergy_lib.SYNERGY_BASIS.tolist(),
                              "names": list(synergy_lib.SYNERGY_NAMES)}
        if verbose:
            print(f"[协同分析] 2 维协同重构 6 维激活：RMSE = {rmse:.4f}，"
                  f"方差解释率 = {explained * 100:.2f}%")

    if capture_dir:
        for frame_name, _, _ in DEMO_SEQUENCE:
            if frame_name in frames:
                save_png(frames[frame_name], os.path.join(capture_dir, f"{frame_name}.png"))
        if gif_frames:
            save_gif(gif_frames, gif_path or os.path.join(capture_dir, "musculoskeletal_demo.gif"))
        if activation_history:
            save_activation_plot(activation_history, coefficients,
                                 os.path.join(capture_dir, "muscle_activation.png"))
        metrics["metrics_path"] = os.path.join(capture_dir, "metrics.json")
        with open(metrics["metrics_path"], "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"[INFO] 实测指标已写入 {metrics['metrics_path']}")

    return metrics


# ----------------------------------------------------------------------
# 模式 4：模型与控制自检
# ----------------------------------------------------------------------
def run_check(arm, params):
    print("=" * 60)
    print("  肌肉骨骼人模型自检")
    print("=" * 60)
    print(arm.summary())
    print()
    print(arm.moment_arm_table())
    print()
    paired_all = True
    for item in arm.antagonist_check():
        paired_all = paired_all and item["paired"]
        print(f"关节 {item['joint']:<16} 主动肌 {len(item['positive'])} 块 / "
              f"拮抗肌 {len(item['negative'])} 块 | 最大力矩臂 {item['max_abs'] * 100:5.2f} cm | "
              f"{'OK' if item['paired'] else '缺失配对!'}")
    print()
    same_order = tuple(synergy_lib.MUSCLE_ORDER) == tuple(MUSCLE_NAMES)
    print(f"协同基底肌肉顺序一致: {'OK' if same_order else '不一致!'}")
    print()

    # 控制回路 sanity check：向 target_forward 走 1.5 s，看误差收敛
    arm.reset()
    arm.set_target(arm.landmarks["target_forward"])
    steps = int(1.5 / arm.model.opt.timestep)
    for _ in range(steps):
        arm.step()
    state = arm.state()
    print(f"控制回路自检：1.5 s 后末端误差 {state['hand_error'] * 1000:.2f} mm | "
          f"关节角 (rad) " + ", ".join(f"{n}={q:+.3f}" for n, q in zip(JOINT_NAMES, state['q'])))
    print(f"肌肉激活: " + ", ".join(f"{n}={a:.2f}" for n, a in
                                    zip(MUSCLE_NAMES, state['activation'])))
    print(f"肌肉力  : " + ", ".join(f"{n}={f:.1f} N" for n, f in
                                    zip(MUSCLE_NAMES, state['force'])))
    print()
    print(f"结论：模型自检{'通过' if paired_all and same_order else '存在问题'}。")
    return 0 if (paired_all and same_order) else 1


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="肌肉骨骼人（肌肉驱动）MuJoCo 仿真与键盘运动控制")
    parser.add_argument("--interactive", action="store_true", help="键盘交互控制（默认模式）")
    parser.add_argument("--demo", action="store_true", help="自动演示动作序列")
    parser.add_argument("--capture", type=str, default=None,
                        help="采集文档素材到指定目录（关键帧/动图/激活曲线/metrics.json）")
    parser.add_argument("--gif", type=str, default=None, help="演示动图输出路径 (.gif)")
    parser.add_argument("--check", action="store_true", help="模型与控制自检")
    parser.add_argument("--test", action="store_true", help="运行自动化测试")
    parser.add_argument("--ros2", action="store_true", help="通过 ros2 launch 启动")
    parser.add_argument("--config", type=str, default=None, help="参数文件 (YAML)")
    parser.add_argument("--synergy", action="store_true", help="使用 2 维肌肉协同控制")
    parser.add_argument("--cocontraction", type=float, default=None, help="共收缩基量 (0~1)")
    parser.add_argument("--no-bias-comp", action="store_true", help="关闭重力/科氏补偿（对照实验）")
    args = parser.parse_args()

    if args.test:
        import subprocess
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "test", "test_standalone.py")
        sys.exit(subprocess.run([sys.executable, test_path]).returncode)
    if args.ros2:
        import subprocess
        print("[INFO] 通过 ros2 launch 启动肌肉骨骼人仿真与键盘遥控...")
        sys.exit(subprocess.run(["ros2", "launch", "humanoid_muscle_arm_control",
                                 "main.launch.py"]).returncode)

    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "config", "sim_params.yaml")
    params = load_params(args.config or default_config)
    if args.synergy:
        params.use_synergy = True
    if args.cocontraction is not None:
        params.cocontraction = float(args.cocontraction)
    if args.no_bias_comp:
        params.bias_compensation = False

    arm = MusculoskeletalArm(params=params)
    try:
        if args.check:
            sys.exit(run_check(arm, params))
        if args.capture:
            run_demo(arm, params, capture_dir=args.capture, gif_path=args.gif)
        elif args.demo:
            run_demo(arm, params)
        else:
            run_interactive(arm, params)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
