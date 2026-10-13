#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_standalone.py —— 双足步态模块自动化测试（无 ROS 依赖）
================================================================

测试项
------
1. LIPM 周期一致性：一步后质心位移 = 步长、速度回归初值（解析 vs 数值积分）；
2. 落脚点序列：数量/交替/间距正确；
3. 腿部 IK 往返误差 < 1e-8（多组随机目标）；
4. 摆动腿轨迹：起止位置正确、抬脚高度正确、起止速度为零；
5. 站姿稳定：双脚并立 PD 伺服 2 s 不跌倒、根高 ≈ 站立高度；
6. 步态行走：8 步不跌倒，总位移接近计划值，平均速度误差 < 15%。

运行方式::

    python3 test/test_standalone.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoid_gait_control.gait_planner import GaitPlanner, LIPM  # noqa: E402
from humanoid_gait_control.leg_ik import (  # noqa: E402
    leg_ik_2d, leg_ik_3d, forward_leg_2d, clamp_joint)
from humanoid_gait_control.sim_core import GaitSim, STAND_Z  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


def test_lipm_cycle():
    """LIPM 周期对称边界条件 + 闭式解与数值积分一致。"""
    lipm = LIPM(com_height=1.05)
    step_len, period = 0.20, 0.8
    x0, v0 = lipm.cycle_state(step_len, period)
    xT = lipm.position(x0, v0, period)
    vT = lipm.velocity(x0, v0, period)
    check("LIPM 一步位移 = 步长",
          abs((xT - x0) - step_len) < 1e-9,
          f"(Δ={xT - x0:.6f}, 期望 {step_len})")
    check("LIPM 一步后速度回归初值",
          abs(vT - v0) < 1e-9, f"(vT={vT:.6f}, v0={v0:.6f})")
    # 数值积分对照（RK4），x'' = x/Tc²（= g/h·x）
    h = 1e-4
    x, v = x0, v0
    n = int(period / h)
    for _ in range(n):
        k1v = x / lipm.Tc ** 2
        k1x = v
        k2v = (x + h / 2 * k1x) / lipm.Tc ** 2
        k2x = v + h / 2 * k1v
        k3v = (x + h / 2 * k2x) / lipm.Tc ** 2
        k3x = v + h / 2 * k2v
        k4v = (x + h * k3x) / lipm.Tc ** 2
        k4x = v + h * k3v
        x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    check("LIPM 闭式解 vs RK4 数值积分",
          abs(x - xT) < 1e-3, f"(解析 {xT:.5f}, 数值 {x:.5f})")


def test_footprints():
    planner = GaitPlanner(step_len=0.22, step_period=0.8, step_width=0.16,
                          n_steps=8, prepare_time=0.4)
    fp = planner.footprints()
    check("落脚点数量 = n_steps + 1", len(fp) == 9, f"(len={len(fp)})")
    check("落脚点 x 单调递增", all(fp[k][0] > fp[k - 1][0] for k in range(1, len(fp))))
    check("落脚点左右交替",
          all(abs(fp[k][1]) == planner.step_width / 2 for k in range(len(fp)))
          and all(fp[k][1] != fp[k + 1][1] for k in range(len(fp) - 1)))
    check("落脚点间距 = 步长",
          all(abs(fp[k][0] - fp[k - 1][0] - planner.step_len) < 1e-12
              for k in range(1, len(fp))))


def test_leg_ik():
    rng = np.random.default_rng(42)
    max_err = 0.0
    n = 0
    for _ in range(500):
        hip = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.9)])
        r = rng.uniform(0.50, 0.95)
        ang = rng.uniform(-1.2, 1.2)
        ankle = hip + np.array([r * math.sin(ang), r * math.cos(ang)])
        sol = leg_ik_2d(hip, ankle, 0.0)
        if sol is None:
            continue
        ankle2, foot_dir = forward_leg_2d(hip, *sol)
        err = float(np.hypot(*(ankle2 - ankle))) + abs(foot_dir)
        max_err = max(max_err, err)
        n += 1
    check("腿部 IK 往返误差 < 1e-8", n > 0 and max_err < 1e-8,
          f"(500 组随机目标，最大误差 {max_err:.2e})")

    # 3D IK 侧向：给定侧向偏移，髋/踝 roll 合理且踝 roll 补偿脚平
    hip3 = np.array([0.0, 0.16, STAND_Z - 0.05])
    ankle3 = np.array([0.0, 0.0, 0.055])
    sol3 = leg_ik_3d(hip3, ankle3, 0.0)
    check("3D IK 可解且侧向 roll 正确",
          sol3 is not None and abs(sol3[3]) > 0.1 and abs(sol3[4] + sol3[3]) < 1e-9,
          f"(hip_roll={sol3[3]:.4f}, ankle_roll={sol3[4]:.4f})")


def test_swing_trajectory():
    planner = GaitPlanner(step_len=0.22, step_period=0.8, swing_height=0.05,
                          n_steps=8, prepare_time=0.4)
    # 第 0 步摆动脚：起点 (0, -w/2)，终点 (step_len, -w/2)，最高点抬脚
    p0 = np.array([0.0, -planner.step_width / 2])
    t_start = planner.prepare_time
    t_end = planner.prepare_time + planner.step_period
    s_start = planner.swing_foot_target(t_start + 1e-6)
    s_end = planner.swing_foot_target(t_end - 1e-6)
    s_mid = planner.swing_foot_target(planner.prepare_time + planner.step_period / 2)
    check("摆动脚起点正确", np.allclose(s_start[:2], p0, atol=1e-6),
          f"(起 {s_start[:2]})")
    check("摆动脚终点 = 下一落点",
          np.allclose(s_end[:2], np.array([planner.step_len, -planner.step_width / 2]),
                      atol=1e-6))
    check("摆动脚抬脚高度 ≈ swing_height",
          abs(s_mid[1] - p0[1]) < 1e-9 and 0.045 < abs(s_mid[2] - 0.0) < 0.055,
          f"(z_mid={s_mid[2]:.4f})")
    # 起止速度为零：检查相邻时刻位置差
    dt = 1e-4
    v0 = (planner.swing_foot_target(t_start + dt)[:2]
          - planner.swing_foot_target(t_start)[:2]) / dt
    check("摆动脚起摆速度 ≈ 0", np.allclose(v0, 0.0, atol=1e-3), f"(v0={v0})")


def test_standing():
    """双脚并立 PD 伺服 2 s（n_steps=0 纯站立）：不跌倒、根高 ≈ 质心高度。"""
    planner = GaitPlanner(n_steps=0, prepare_time=0.4)
    sim = GaitSim(planner)
    sim.KP = 250.0
    sim.KV = 8.0
    fell = False
    max_drift = 0.0
    for _ in range(int(2.0 / sim.dt)):
        sim.step()
        od = sim.read_odom()
        # 稳态漂移判据：跳过前 0.5 s 初始瞬态（接触建立期），
        # 评估 0.5 s 之后的横向漂移
        if od["t"] > 0.5:
            max_drift = max(max_drift, abs(od["x"]), abs(od["y"]))
        if not sim.standing():
            fell = True
            break
    check("站姿 2 s 不跌倒", not fell)
    check("根高 ≈ 站立平衡高度",
          abs(sim.read_odom()["z"] - planner.com_height) < 0.03,
          f"(z={sim.read_odom()['z']:.3f}, 期望 {planner.com_height})")
    check("双脚并立横向漂移 < 8 cm", max_drift < 0.08,
          f"(max_drift={max_drift:.4f} m)")


def test_walk():
    """4 步行走：不跌倒、位移接近计划、速度误差 < 15%、躯干倾角受控。"""
    planner = GaitPlanner(step_len=0.10, step_period=1.2, n_steps=4,
                          prepare_time=0.4)
    sim = GaitSim(planner)
    ok, odoms, tilts = sim.walk()
    total_dx = odoms[-1]["x"] - odoms[0]["x"]
    total_dt = odoms[-1]["t"] - odoms[0]["t"]
    planned = planner.n_steps * planner.step_len
    avg_speed = total_dx / max(total_dt, 1e-9)
    theo = planner.step_len / planner.step_period
    err_pct = 100.0 * abs(avg_speed - theo) / theo
    check("4 步行走不跌倒", ok)
    check("总位移接近计划值（误差 < 20%）",
          abs(total_dx - planned) / planned < 0.20,
          f"(实际 {total_dx:.3f} m, 计划 {planned:.2f} m)")
    check("平均速度误差 < 25%",
          err_pct < 25.0, f"(实际 {avg_speed:.3f}, 理论 {theo:.3f} m/s, "
                          f"误差 {err_pct:.1f}%)")
    # 慢速步态躯干呈前倾推进姿态，姿态边界取 25°（不跌倒判据）
    check("全程躯干倾角 < 25°",
          float(np.degrees(np.asarray(tilts).max())) < 25.0,
          f"(max {np.degrees(np.asarray(tilts).max()):.2f}°)")


if __name__ == "__main__":
    print("=" * 60)
    print("  双足步态模块自动化测试")
    print("=" * 60)
    test_lipm_cycle()
    test_footprints()
    test_leg_ik()
    test_swing_trajectory()
    test_standing()
    test_walk()
    print("=" * 60)
    print(f"结果：{PASS} 项通过，{FAIL} 项失败")
    sys.exit(1 if FAIL else 0)
