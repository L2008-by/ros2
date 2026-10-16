#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_standalone.py —— 平衡与抗扰恢复模块自动化测试（无 ROS 依赖）
====================================================================

测试项
------
1.  模型加载：自由度 / 执行器数量 / 关节数量；
2.  支撑多边形：脚底 8 角点凸包有效，静立时质心投影在支撑多边形内且有裕度；
3.  捕获点与 ZMP 计算有效；
4.  自由站立稳定性：无扰动 8 s 不跌倒且质心漂移 < 5 mm；
5.  踝策略符号自检：踝 pitch 目标增大 → 质心向后移动（控制律符号来源）；
6.  对照实验：关闭平衡控制器时同样扰动必然跌倒；
7.  前向抗扰恢复：120 N 推扰可恢复，最大偏移与恢复时间在上界内；
8.  侧向抗扰恢复：150 N 侧向推扰可恢复；
9.  策略对比：踝+髋策略的临界冲量不低于仅踝策略；
10. 恢复残差：恢复后质心残差 < 5 mm；
11. 跌倒判定：超大扰动下正确判为跌倒且根高/倾角越界；
12. 仿真性能：实时因子满足演示要求。

运行方式（包根目录）::

    python3 test/test_standalone.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from humanoid_push_recovery.balance import (                              # noqa: E402
    BalanceController, capture_point, convex_hull, distance_to_boundary)
from humanoid_push_recovery.evaluate import (                             # noqa: E402
    critical_force, run_trial)
from humanoid_push_recovery.sim_core import (                             # noqa: E402
    JOINT_NAMES, BalanceSim, load_params)

PARAMS = load_params()

PASS = 0
FAIL = 0
SKIP = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"[PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {detail}")


def skip(name, reason):
    global SKIP
    SKIP += 1
    print(f"[SKIP] {name} （{reason}）")


def make_sim(balance=True):
    sim = BalanceSim(params=PARAMS)
    if balance:
        sim.controller = BalanceController(PARAMS)
    sim.reset()
    return sim


# ----------------------------------------------------------------------
def test_model():
    """1. 模型与执行器。"""
    sim = make_sim()
    check("模型加载与执行器数量",
          sim.model.nq > 0 and sim.model.nu == len(JOINT_NAMES) and len(JOINT_NAMES) == 15,
          f"(nq={sim.model.nq}, nv={sim.model.nv}, nu={sim.model.nu}, 关节={len(JOINT_NAMES)})")
    check("底盘 freejoint 未被速度锁定",
          sim.model.jnt_type[sim.root_id] == 0,      # mjJNT_FREE == 0
          f"(jnt_type={int(sim.model.jnt_type[sim.root_id])} = free)")
    sim.close()


def test_support_polygon():
    """2. 支撑多边形与稳定裕度。"""
    sim = make_sim()
    corners = sim.foot_corners()
    hull = convex_hull(corners[:, :2])
    com = sim.com_position()
    margin = distance_to_boundary(com[:2], hull)
    check("脚底 8 角点构成有效凸包", corners.shape == (8, 3) and len(hull) >= 3,
          f"(角点 {corners.shape}, 凸包顶点 {len(hull)})")
    check("静立时质心投影落在支撑多边形内且有裕度", margin > 0.02,
          f"(稳定裕度 {margin * 1000:.1f} mm)")
    sx = corners[:, 0].max() - corners[:, 0].min()
    sy = corners[:, 1].max() - corners[:, 1].min()
    check("侧向支撑宽度大于前后支撑长度（双足站立特征）", sy > sx,
          f"(x 跨度 {sx * 1000:.0f} mm, y 跨度 {sy * 1000:.0f} mm)")
    sim.close()


def test_balance_criteria():
    """3. ZMP 与捕获点。"""
    sim = make_sim()
    com = sim.com_position()
    zmp = sim.zmp()
    xi = capture_point(com, sim.com_velocity(), com[2])
    check("静立时 ZMP 与质心投影接近（加速度≈0）",
          float(np.linalg.norm(zmp - com[:2])) < 0.02,
          f"(|ZMP−CoM_xy| = {float(np.linalg.norm(zmp - com[:2])) * 1000:.2f} mm)")
    check("捕获点计算有限且与质心量纲一致",
          np.all(np.isfinite(xi)) and abs(float(xi[0] - com[0])) < 0.2,
          f"(捕获点 {np.round(xi, 4)} m)")

    # 手动给一个水平速度，验证捕获点随速度前移
    xi_fast = capture_point(com, np.array([0.5, 0.0, 0.0]), com[2])
    check("捕获点随质心速度前移", float(xi_fast[0]) > float(xi[0]),
          f"(v=0.5 m/s 时捕获点 x 前移 {float(xi_fast[0] - xi[0]) * 1000:.0f} mm)")
    sim.close()


def test_free_standing():
    """4. 无扰动自由站立稳定性。"""
    sim = make_sim()
    ref = sim.com_position()
    max_drift = 0.0
    for _ in range(int(8.0 / sim.dt)):
        sim.step()
        max_drift = max(max_drift, float(np.linalg.norm(sim.com_position()[:2] - ref[:2])))
        if sim.fallen():
            break
    check("无扰动自由站立 8 s 不跌倒", not sim.fallen(),
          f"(根高 {sim.root_height():.3f} m, 倾角 {np.degrees(sim.torso_tilt()):.2f}°)")
    check("自由站立质心漂移 < 5 mm", max_drift < 0.005,
          f"(最大漂移 {max_drift * 1000:.2f} mm)")
    sim.close()


def test_ankle_sign():
    """5. 踝策略符号自检（控制律符号来源，防止符号写反导致正反馈）。"""
    sim = make_sim(balance=False)
    y0 = sim.com_position()[0]
    target = np.zeros(len(JOINT_NAMES))
    target[JOINT_NAMES.index("ankle_pitch_l")] = 0.06
    target[JOINT_NAMES.index("ankle_pitch_r")] = 0.06
    for _ in range(400):
        sim.data.ctrl[:] = 0.0
        ctrl = sim.KP * (target - sim.data.qpos[sim.joint_qadr]) \
            - sim.KV * sim.data.qvel[sim.joint_dadr]
        for qi in sim.roll_idx:
            ctrl[qi] = sim.KP_ROLL * (target[qi] - sim.data.qpos[sim.joint_qadr[qi]]) \
                - sim.KV * sim.data.qvel[sim.joint_dadr[qi]]
        sim.data.ctrl[:] = np.clip(ctrl, -sim.TORQUE_LIMIT, sim.TORQUE_LIMIT)
        import mujoco
        mujoco.mj_step(sim.model, sim.data)
        sim._update_com()
    dx = sim.com_position()[0] - y0
    check("踝 pitch 目标增大 → 质心向后移动（控制律符号正确）", dx < -0.01,
          f"(踝 +0.06 rad → 质心 Δx = {dx * 1000:+.1f} mm)")
    sim.close()


def test_no_balance_falls():
    """6. 对照实验：无平衡控制器必然跌倒。"""
    r = run_trial(PARAMS, force=120.0, direction_deg=0.0, balance=False)
    check("关闭平衡控制器时 120 N 推扰即跌倒（对照成立）", bool(r["fell"]),
          f"(无平衡：{'跌倒' if r['fell'] else '竟未跌倒'}, "
          f"最大偏移 {r['max_offset_m'] * 1000:.0f} mm)")


def test_forward_recovery():
    """7. 前向抗扰恢复。"""
    r = run_trial(PARAMS, force=120.0, direction_deg=0.0)
    check("前向 120 N 推扰可恢复", bool(r["recovered"]),
          f"(最大偏移 {r['max_offset_m'] * 1000:.0f} mm, "
          f"恢复 {r['recover_time_s']:.2f} s)" if r["recovered"] else "(未恢复)")
    check("前向 120 N 最大偏移 < 80 mm", r["max_offset_m"] < 0.080,
          f"(最大偏移 {r['max_offset_m'] * 1000:.1f} mm)")
    check("前向 120 N 恢复时间 < 2.5 s",
          bool(r["recover_time_s"] is not None and r["recover_time_s"] < 2.5),
          f"(恢复时间 {r['recover_time_s'] if r['recover_time_s'] else float('nan'):.2f} s)")


def test_lateral_recovery():
    """8. 侧向抗扰恢复。"""
    r = run_trial(PARAMS, force=150.0, direction_deg=90.0)
    check("侧向 150 N 推扰可恢复", bool(r["recovered"]),
          f"(最大偏移 {r['max_offset_m'] * 1000:.0f} mm, "
          f"恢复 {r['recover_time_s']:.2f} s)" if r["recovered"] else "(未恢复)")
    check("侧向 150 N 最大偏移 < 80 mm", r["max_offset_m"] < 0.080,
          f"(最大偏移 {r['max_offset_m'] * 1000:.1f} mm)")


def test_strategy_comparison():
    """9~10. 策略对比与恢复残差。"""
    f_ankle = critical_force(PARAMS, direction_deg=0.0, hip_enable=False,
                             iterations=8, total=6.0)
    f_hip = critical_force(PARAMS, direction_deg=0.0, hip_enable=True,
                           iterations=8, total=6.0)
    dur = float(PARAMS.get("push_duration", 0.08))
    check("仅踝策略临界冲量 > 5 N·s", f_ankle * dur > 5.0,
          f"(临界 {f_ankle:.1f} N = {f_ankle * dur:.2f} N·s)")
    check("踝+髋策略临界冲量不低于仅踝策略", f_hip >= f_ankle - 1e-6,
          f"(踝+髋 {f_hip:.1f} N = {f_hip * dur:.2f} N·s vs 仅踝 {f_ankle * dur:.2f} N·s, "
          f"提升 {(f_hip - f_ankle) / f_ankle * 100:.1f}%)")

    r = run_trial(PARAMS, force=100.0, direction_deg=0.0)
    resid = r["residual_offset_m"]
    check("恢复后质心残差 < 5 mm", bool(resid is not None and resid < 0.005),
          f"(残差 {resid * 1000:.2f} mm)" if resid is not None else "(未恢复)")


def test_fall_detection():
    """11. 跌倒判定。"""
    r = run_trial(PARAMS, force=500.0, direction_deg=0.0)
    check("超大扰动被正确判为跌倒", bool(r["fell"]),
          f"(500 N 推扰 → {'跌倒' if r['fell'] else '竟恢复'})")
    check("跌倒时根高或倾角越界",
          r["fell"] and (r["final_root_height"] < PARAMS.get("fall_height", 0.85)
                         or r["final_tilt_rad"] > PARAMS.get("fall_tilt", 0.55)),
          f"(根高 {r['final_root_height']:.3f} m, 倾角 {np.degrees(r['final_tilt_rad']):.1f}°)")


def test_performance():
    """12. 仿真吞吐。"""
    sim = make_sim()
    steps = 2000
    start = time.perf_counter()
    for _ in range(steps):
        sim.step()
    elapsed = time.perf_counter() - start
    sps = steps / max(1e-9, elapsed)
    check("仿真吞吐 > 2000 step/s（实时因子 > 4×）", sps > 2000.0,
          f"({sps:.0f} step/s, 实时因子 {sps * float(sim.model.opt.timestep):.2f}×)")
    sim.close()


def main():
    print("=" * 66)
    print("  人形机器人平衡与抗扰恢复模块自动化测试")
    print("=" * 66)
    test_model()
    test_support_polygon()
    test_balance_criteria()
    test_free_standing()
    test_ankle_sign()
    test_no_balance_falls()
    test_forward_recovery()
    test_lateral_recovery()
    test_strategy_comparison()
    test_fall_detection()
    test_performance()
    print("=" * 66)
    print(f"结果：{PASS} 项通过，{FAIL} 项失败，{SKIP} 项跳过")
    print("=" * 66)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
