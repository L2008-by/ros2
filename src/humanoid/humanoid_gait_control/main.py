#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双足步态生成与行走控制（LIPM 步态 + 腿部 IK + 关节 PD）—— 主入口
================================================================

课程要求：入口为 ``main.py``，无需 ROS 即可运行整个模块。

运行模式
--------
1. ``--demo``     ——自动演示「准备 → 周期步态行走 N 步」，打印每步实测指标；
2. ``--capture DIR`` ——在 ``--demo`` 基础上把关键帧、动图与 ``metrics.json``
   （实测指标）写入目录，用于生成文档配图；
3. ``--check``    ——模型/步态自检：关节拓扑、腿部 IK 往返误差、LIPM 周期一致性；
4. ``--test``     ——运行 ``test/test_standalone.py``（自动化测试）；
5. ``--ros2``     ——通过 ``ros2 launch`` 启动仿真节点与步态遥控节点。

示例::

    python3 main.py --demo               # 自动行走 8 步
    python3 main.py --capture ../docs_img # 生成文档素材（关键帧/动图/metrics）
    python3 main.py --check              # 模型与步态自检
    python3 main.py --test               # 自动化测试
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from humanoid_gait_control.gait_planner import GaitPlanner  # noqa: E402
from humanoid_gait_control.sim_core import GaitSim  # noqa: E402


#: 素材关键帧时间点：(帧名, 说明, 时刻 s)
FRAME_TIMES = (
    ("scene_ready", "准备段：双脚并立，重心稳定", 0.30),
    ("scene_step1", "第 1 步：右腿摆动中（抬脚离地）", 0.40 + 0.30),
    ("scene_step3", "第 3 步：交替迈步中", 0.40 + 2.0 * 0.8 + 0.30),
    ("scene_walk", "周期步态：连续行走", 0.40 + 4.0 * 0.8 + 0.30),
)


def load_params(config_path: str | None) -> dict:
    """读取 YAML 参数（缺 yaml 依赖时回落默认值）。"""
    defaults = dict(
        com_height=1.05, step_len=0.10, step_period=1.2, step_width=0.05,
        swing_height=0.05, n_steps=4, prepare_time=0.4, start_foot="left",
        kp=400.0, kd=15.0, kp_roll=800.0, kd_roll=18.0, ctrl_limit=300.0,
        arm_swing=0.30, elbow_bend=-0.25,
        camera="overview", height=480, width=640,
    )
    if not config_path or not os.path.exists(config_path):
        return defaults
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 参数文件解析失败({exc})，使用默认参数")
        return defaults
    out = dict(defaults)
    for grp in ("gait", "control", "render"):
        for k, v in (raw.get(grp) or {}).items():
            out[k] = v
    return out


def build_planner(params: dict, overrides: dict) -> GaitPlanner:
    """按参数与命令行覆盖项构建步态规划器。"""
    p = dict(params)
    for key in ("step_len", "step_period", "step_width", "swing_height",
                "com_height", "n_steps", "prepare_time", "start_foot"):
        if overrides.get(key) is not None:
            p[key] = overrides[key]
    return GaitPlanner(
        step_len=float(p["step_len"]), step_period=float(p["step_period"]),
        step_width=float(p["step_width"]), swing_height=float(p["swing_height"]),
        com_height=float(p["com_height"]), n_steps=int(p["n_steps"]),
        prepare_time=float(p["prepare_time"]), start_foot=p["start_foot"],
    )


def apply_ctrl_params(sim: GaitSim, params: dict):
    """把控制增益写入仿真器。"""
    sim.KP = float(params["kp"])
    sim.KV = float(params["kd"])
    sim.KP_ROLL = float(params["kp_roll"])
    sim.KV_ROLL = float(params["kd_roll"])
    sim.CTRL_LIMIT = float(params["ctrl_limit"])
    sim.ARM_SWING = float(params["arm_swing"])
    sim.ELBOW_BEND = float(params["elbow_bend"])


# ----------------------------------------------------------------------
# 模式 1/2：演示与素材采集
# ----------------------------------------------------------------------
def run_demo(sim: GaitSim, planner: GaitPlanner, capture_dir: str | None = None,
             gif_path: str | None = None, verbose: bool = True) -> dict:
    """自动步态演示；``capture_dir`` 不为空时输出关键帧、动图与 metrics.json。"""
    frames = {}
    gif_frames = []
    metrics = {
        "gait": planner.metrics(),
        "sequence": [],
        "capture": [],
    }
    total = planner.total_time

    def sample(t, od, tilt):
        # 动图降采样（320×240 控制体积）
        if capture_dir and int(round(t / sim.dt)) % 64 == 0:
            gif_frames.append(sim.render(camera="overview")[::2, ::2])

    ok, odoms, tilts = sim.walk(on_step=sample)

    # 逐段统计（按 planner 相位）
    for k in range(planner.n_steps):
        t_start = planner.prepare_time + k * planner.step_period
        t_end = t_start + planner.step_period
        seg = [o for o in odoms if t_start <= o["t"] < t_end]
        if not seg:
            continue
        dx = seg[-1]["x"] - seg[0]["x"]
        metrics["sequence"].append({
            "step": k + 1,
            "t_start_s": round(t_start, 3),
            "displacement_m": round(dx, 4),
            "avg_speed_mps": round(dx / (seg[-1]["t"] - seg[0]["t"]), 4),
        })
        if verbose:
            print(f"[第 {k + 1:>2} 步] 位移 {dx:+.3f} m | "
                  f"平均速度 {dx / max(seg[-1]['t'] - seg[0]['t'], 1e-9):.3f} m/s")

    # 总体指标
    total_dx = odoms[-1]["x"] - odoms[0]["x"]
    total_dt = odoms[-1]["t"] - odoms[0]["t"]
    tilt_arr = np.asarray(tilts)
    metrics["result"] = {
        "success": ok,
        "fell_at_s": None if ok else round(odoms[-1]["t"], 3),
        "total_distance_m": round(total_dx, 4),
        "planned_distance_m": planner.n_steps * planner.step_len,
        "avg_speed_mps": round(total_dx / max(total_dt, 1e-9), 4),
        "theoretical_speed_mps": planner.step_len / planner.step_period,
        "speed_error_pct": round(
            100.0 * abs(total_dx / max(total_dt, 1e-9)
                       - planner.step_len / planner.step_period)
            / (planner.step_len / planner.step_period), 2),
        "torso_tilt_mean_deg": round(float(np.degrees(tilt_arr.mean())), 3),
        "torso_tilt_max_deg": round(float(np.degrees(tilt_arr.max())), 3),
        "torso_tilt_std_deg": round(float(np.degrees(tilt_arr.std())), 3),
    }
    if verbose:
        fell_msg = f"跌倒于 {odoms[-1]['t']:.2f} s" if not ok else "成功"
        print("=" * 60)
        print(f"  行走结果：{fell_msg}")
        print(f"  总位移 {total_dx:.3f} m（计划 {planner.n_steps * planner.step_len:.2f} m）"
              f" | 平均速度 {total_dx / max(total_dt, 1e-9):.3f} m/s "
              f"（理论 {planner.step_len / planner.step_period:.3f} m/s）")
        print(f"  躯干倾角 均值 {np.degrees(tilt_arr.mean()):.2f}° "
              f"最大 {np.degrees(tilt_arr.max()):.2f}° "
              f"标准差 {np.degrees(tilt_arr.std()):.2f}°")
        print("=" * 60)

    # 素材采集
    if capture_dir:
        for frame_name, desc, t_frame in FRAME_TIMES:
            sim.reset()
            # 快进到目标时刻
            for _ in range(int(round(t_frame / sim.dt))):
                sim.step()
            frames[frame_name] = sim.render(camera="overview")
            metrics["capture"].append({"frame": frame_name, "desc": desc,
                                       "t_s": t_frame})
            print(f"[{frame_name}] {desc} | 渲染完成")
        for name, img in frames.items():
            save_png(img, os.path.join(capture_dir, f"{name}.png"))
        if gif_frames:
            save_gif(gif_frames, gif_path or os.path.join(capture_dir, "gait_demo.gif"))
        metrics["metrics_path"] = os.path.join(capture_dir, "metrics.json")
        with open(metrics["metrics_path"], "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"[INFO] 实测指标已写入 {metrics['metrics_path']}")
    return metrics


# ----------------------------------------------------------------------
# 模式 3：模型与步态自检
# ----------------------------------------------------------------------
def run_check(planner: GaitPlanner, params: dict) -> int:
    from humanoid_gait_control.leg_ik import leg_ik_2d, forward_leg_2d
    print("=" * 60)
    print("  双足步态模块自检")
    print("=" * 60)

    # 1) LIPM 周期一致性：一步后质心位移 = 步长，速度回归初值
    lipm = planner.lipm
    x0, v0 = lipm.cycle_state(planner.step_len, planner.step_period)
    xT = lipm.position(x0, v0, planner.step_period)
    vT = lipm.velocity(x0, v0, planner.step_period)
    print(f"[LIPM] h={planner.com_height:.3f} m  Tc={lipm.Tc:.3f} s | "
          f"步长 {planner.step_len:.3f} m  周期 {planner.step_period:.3f} s")
    print(f"[LIPM] 初态 x0={x0:+.4f} m  v0={v0:+.4f} m/s | "
          f"一步后 x={xT:+.4f} m (Δ={xT - x0:+.4f}，期望 {planner.step_len:+.4f}) "
          f"v={vT:+.4f} m/s (期望 {v0:+.4f})")
    lipm_ok = abs((xT - x0) - planner.step_len) < 1e-6 and abs(vT - v0) < 1e-6

    # 2) 落脚点序列
    fp = planner.footprints()
    print(f"[步态] 落脚点 {len(fp)} 个：")
    for k, p in enumerate(fp):
        print(f"   {k}: ({p[0]:+.3f}, {p[1]:+.3f})")
    fp_ok = len(fp) == planner.n_steps + 1

    # 3) 腿部 IK 往返误差（多组随机目标）
    rng = np.random.default_rng(7)
    ik_errs = []
    for _ in range(200):
        hip = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.9)])
        # 随机可达踝目标（半径 0.50~0.95 内，腿长 0.52+0.46）
        r = rng.uniform(0.50, 0.95)
        ang = rng.uniform(-1.2, 1.2)
        ankle = hip + np.array([r * math.sin(ang), r * math.cos(ang)])
        sol = leg_ik_2d(hip, ankle, 0.0)
        if sol is None:
            continue
        ankle2, foot_dir = forward_leg_2d(hip, *sol)
        ik_errs.append(float(np.hypot(*(ankle2 - ankle))) + abs(foot_dir))
    ik_ok = len(ik_errs) > 0 and max(ik_errs) < 1e-8
    print(f"[IK] 往返误差 200 组最大 {max(ik_errs) if ik_errs else float('nan'):.2e} m+rad")

    ok = lipm_ok and fp_ok and ik_ok
    print()
    print(f"结论：自检{'通过' if ok else '存在问题'}。")
    return 0 if ok else 1


# ----------------------------------------------------------------------
# 素材输出
# ----------------------------------------------------------------------
def save_png(image, path: str) -> bool:
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


def save_gif(frames, path: str, fps: int = 15) -> bool:
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


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="双足步态生成与行走控制（LIPM 步态 + 腿部 IK + 关节 PD）")
    parser.add_argument("--demo", action="store_true", help="自动演示行走 N 步")
    parser.add_argument("--capture", type=str, default=None,
                        help="采集文档素材到指定目录（关键帧/动图/metrics.json）")
    parser.add_argument("--gif", type=str, default=None, help="演示动图输出路径 (.gif)")
    parser.add_argument("--check", action="store_true", help="模型与步态自检")
    parser.add_argument("--test", action="store_true", help="运行自动化测试")
    parser.add_argument("--ros2", action="store_true", help="通过 ros2 launch 启动")
    parser.add_argument("--config", type=str, default=None, help="参数文件 (YAML)")
    parser.add_argument("--steps", type=int, default=None, help="行走步数覆盖")
    parser.add_argument("--step-len", type=float, default=None, help="单步步长覆盖 (m)")
    parser.add_argument("--period", type=float, default=None, help="单步周期覆盖 (s)")
    args = parser.parse_args()

    if args.test:
        import subprocess
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "test", "test_standalone.py")
        sys.exit(subprocess.run([sys.executable, test_path]).returncode)
    if args.ros2:
        import subprocess
        print("[INFO] 通过 ros2 launch 启动双足步态仿真与遥控...")
        sys.exit(subprocess.run(["ros2", "launch", "humanoid_gait_control",
                                 "main.launch.py"]).returncode)

    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "config", "sim_params.yaml")
    params = load_params(args.config or default_config)
    planner = build_planner(params, {
        "step_len": args.step_len, "step_period": args.period,
        "n_steps": args.steps,
    })
    if args.check:
        sys.exit(run_check(planner, params))

    sim = GaitSim(planner)
    apply_ctrl_params(sim, params)
    try:
        if args.capture:
            run_demo(sim, planner, capture_dir=args.capture, gif_path=args.gif)
        else:
            run_demo(sim, planner)
    finally:
        sim.model = None  # 释放渲染资源


if __name__ == "__main__":
    main()
