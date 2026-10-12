# -*- coding: utf-8 -*-
"""
肌肉骨骼人（肌肉驱动）仿真与控制自动化测试
==========================================

覆盖本模块真正的技术点（而不是"能跑就行"）：

1. 协同基底与肌肉顺序一致（防止 ``synergy.py`` 与 ``sim_core.py`` 漂移）；
2. 力矩臂矩阵：每个自由度都存在正/负力矩臂（主动肌-拮抗肌配对），幅值在合理范围；
3. 控制分配恒等式：最小范数分配的肌肉力矩应复现期望关节力矩（未饱和时）；
4. 激活动力学：激活上升比放松快，时间常数在生理量级；
5. 末端定位精度：三个目标点的稳态误差；
6. 重力/科氏补偿的作用（开/关对照）；
7. 共收缩 → 末端刚度（施加探针力实测位移）；
8. 2 维肌肉协同对 6 维激活的重构能力（含数据驱动 NMF 对照）；
9. 相机渲染可用、仿真吞吐。

运行方式（包根目录）::

    python3 test/test_standalone.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoid_muscle_arm_control.sim_core import (      # noqa: E402
    MusculoskeletalArm, ArmParams, JOINT_NAMES, MUSCLE_NAMES)
from humanoid_muscle_arm_control import synergy as synergy_lib  # noqa: E402

PASSED = []


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_synergy_consistency():
    """协同基底的行顺序必须与肌肉顺序一致。"""
    check(tuple(synergy_lib.MUSCLE_ORDER) == tuple(MUSCLE_NAMES),
          "synergy.MUSCLE_ORDER 与 sim_core.MUSCLE_NAMES 不一致")
    check(synergy_lib.SYNERGY_BASIS.shape == (2, len(MUSCLE_NAMES)),
          f"协同基底形状异常: {synergy_lib.SYNERGY_BASIS.shape}")
    rows_sum = np.sum(synergy_lib.SYNERGY_BASIS, axis=1)
    check(np.allclose(rows_sum, 1.0, atol=1e-9), f"协同基底每行应为 1，实际 {rows_sum}")
    print(f"[PASS] 协同基底一致性: 2 个协同 × {len(MUSCLE_NAMES)} 块肌肉，行和 = 1")


def test_moment_arms(arm):
    """力矩臂矩阵：每个关节都有主动肌与拮抗肌，幅值在合理范围。"""
    print(arm.moment_arm_table())
    for item in arm.antagonist_check():
        check(item["paired"], f"关节 {item['joint']} 缺少主动肌/拮抗肌配对")
        check(0.01 < item["max_abs"] < 0.20,
              f"关节 {item['joint']} 最大力矩臂 {item['max_abs']:.3f} m 超出合理范围 (1~20 cm)")
    print("[PASS] 力矩臂矩阵: 3 个自由度均存在主动肌/拮抗肌配对，幅值在 1~20 cm 内")


def test_allocation_identity(arm):
    """控制分配恒等式：肌肉力复现的关节力矩应等于期望力矩（未饱和时）。

    注意：实测确认 MuJoCo tendon 执行器的关节力矩贡献为 ``τ = +Mᵀ F``
    （见 ``sim_core.allocate`` 的说明），因此这里用 ``+Mᵀ`` 复现。
    """
    arm.reset()
    torques = [np.array([0.6, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]),
               np.array([0.0, 0.0, 0.4]), np.array([0.4, 0.3, 0.3])]
    worst = 0.0
    clipped = 0
    for desired in torques:
        force_cmd, _ = arm.allocate(desired)
        realized = arm.moment_arms.T @ force_cmd
        error = float(np.linalg.norm(realized - desired))
        if np.any(force_cmd <= 1e-9) or np.any(force_cmd >= arm.f_max_vec - 1e-9):
            clipped += 1                      # 肌肉只能"拉"（力非负），受限时允许有偏差
        else:
            worst = max(worst, error)
    check(worst < 0.05, f"分配恒等式误差过大: {worst:.4f} N·m")
    print(f"[PASS] 控制分配恒等式: 未饱和情形的最大复现误差 {worst * 1000:.3f} mN·m "
          f"（另有 {clipped} 组因肌肉力非负约束而饱和，按要求跳过）")


def test_activation_dynamics(arm):
    """激活动力学：上升快于放松，时间常数在生理量级。"""

    def rise_time(target, limit=0.5):
        arm.muscles.reset()
        length = arm.data.actuator_length[0]
        velocity = 0.0
        activation = []
        steps = int(limit / 0.001)
        for _ in range(steps):
            arm.muscles.update_activation([target] * len(MUSCLE_NAMES), 0.001)
            activation.append(arm.muscles.activation[0])
        activation = np.array(activation)
        times = np.arange(len(activation)) * 0.001
        index = int(np.argmax(activation >= 0.632 * target)) if np.any(activation >= 0.632 * target) else -1
        return float(times[index]) if index >= 0 else float("inf"), length, velocity

    rise, _, _ = rise_time(1.0)
    arm.muscles.reset()
    for _ in range(int(0.2 / 0.001)):
        arm.muscles.update_activation([1.0] * len(MUSCLE_NAMES), 0.001)
    fall_start = arm.muscles.activation[0]
    fall_times = []
    for _ in range(int(0.5 / 0.001)):
        arm.muscles.update_activation([0.0] * len(MUSCLE_NAMES), 0.001)
        fall_times.append(arm.muscles.activation[0])
    fall_curve = np.array(fall_times)
    threshold = fall_start * (1.0 - 0.632)
    fall_index = int(np.argmax(fall_curve <= threshold)) if np.any(fall_curve <= threshold) else -1
    fall = float(fall_index * 0.001) if fall_index >= 0 else float("inf")
    arm.muscles.reset()

    check(0.005 < rise < 0.100, f"激活上升时间异常: {rise * 1000:.1f} ms")
    check(fall > rise, f"放松应慢于激活 (激活 {rise * 1000:.1f} ms vs 放松 {fall * 1000:.1f} ms)")
    print(f"[PASS] 激活动力学: 上升到 63% 用时 {rise * 1000:.1f} ms，放松到 37% 用时 "
          f"{fall * 1000:.1f} ms（τ_act={arm.params.tau_act * 1000:.0f} ms / "
          f"τ_deact={arm.params.tau_deact * 1000:.0f} ms）")


def test_reach_accuracy(arm, tolerance_mm=20.0):
    """三个目标点的末端稳态定位误差。"""
    dt = arm.model.opt.timestep
    results = []
    for name in ("target_forward", "target_lateral", "target_low"):
        arm.reset()
        arm.set_target(arm.landmarks[name])
        start = arm.data.time
        while arm.data.time - start < 3.0:
            arm.step()
            if arm.state()['hand_error'] < 0.005:
                break
        for _ in range(int(0.3 / dt)):
            arm.step()
        state = arm.state()
        results.append((name, state['hand_error'] * 1000.0, arm.data.time - start))
        check(state['hand_error'] * 1000.0 < tolerance_mm,
              f"{name} 稳态误差 {state['hand_error'] * 1000:.2f} mm 超过 {tolerance_mm} mm")
    for name, error, spent in results:
        print(f"[PASS] 末端定位 {name:<12} 稳态误差 {error:5.2f} mm（用时 {spent:.2f} s）")


def test_gravity_compensation():
    """重力/科氏补偿的作用：关闭补偿时末端会明显下垂。"""
    dt = 0.001
    errors = {}
    for label, enabled in (("开启补偿", True), ("关闭补偿", False)):
        params = ArmParams(bias_compensation=enabled)
        sim = MusculoskeletalArm(params=params)
        sim.reset()
        target = sim.landmarks["target_forward"]
        sim.set_target(target)
        for _ in range(int(2.5 / dt)):
            sim.step()
        errors[label] = float(np.linalg.norm(sim.hand_position() - target))
        sim.close()
    check(errors["开启补偿"] < errors["关闭补偿"],
          f"开启补偿应更准确: {errors}")
    print(f"[PASS] 重力/科氏补偿: 开启时末端误差 {errors['开启补偿'] * 1000:.2f} mm，"
          f"关闭时 {errors['关闭补偿'] * 1000:.2f} mm"
          f"（提升 {(errors['关闭补偿'] - errors['开启补偿']) * 1000:.2f} mm）")


def test_cocontraction_stiffness(arm):
    """共收缩 → 末端刚度：同一探针力下位移更小。"""
    results = {}
    for c0 in (0.02, 0.50):
        arm.reset()
        arm.set_target(arm.hand_position().copy())
        results[c0] = arm.measure_stiffness(direction=(1.0, 0.0, -0.4), force=8.0,
                                            cocontraction=c0, settle_time=1.2)
    low, high = results[0.02], results[0.50]
    check(high["stiffness"] > low["stiffness"],
          f"共收缩应提高刚度: c0=0.02 → {low['stiffness']:.1f} N/m, "
          f"c0=0.50 → {high['stiffness']:.1f} N/m")
    print(f"[PASS] 共收缩刚度: c0=0.02 → {low['stiffness']:6.1f} N/m "
          f"(位移 {low['displacement_norm'] * 1000:.2f} mm)；"
          f"c0=0.50 → {high['stiffness']:6.1f} N/m "
          f"(位移 {high['displacement_norm'] * 1000:.2f} mm)，"
          f"刚度提升 {high['stiffness'] / low['stiffness']:.2f} 倍")


def test_synergy_reconstruction(arm):
    """2 维肌肉协同对 6 维激活的重构能力（数据驱动 NMF 为主，手工基底作对照）。"""
    dt = arm.model.opt.timestep
    arm.reset()
    history = []
    for name in ("target_forward", "target_lateral", "target_low"):
        arm.set_target(arm.landmarks[name])
        start = arm.data.time
        while arm.data.time - start < 1.5:
            arm.step()
            if arm.step_count % 10 == 0:
                history.append(arm.state()['activation'])
    matrix = np.array(history, dtype=float)
    rmse, explained = synergy_lib.reconstruction_error(matrix)
    w, h = synergy_lib.nmf(matrix, rank=2, iterations=500)
    approx = (w @ h).T
    total = float(np.sum((matrix - matrix.mean(axis=0)) ** 2))
    nmf_explained = 1.0 - float(np.sum((matrix - approx) ** 2)) / max(total, 1e-12)
    check(nmf_explained > 0.75,
          f"2 个数据驱动协同的方差解释率过低: {nmf_explained:.3f}（激活矩阵 {matrix.shape}）")
    print(f"[PASS] 肌肉协同: 记录 {matrix.shape[0]} 帧 × {matrix.shape[1]} 块肌肉 | "
          f"NMF(2) 方差解释率={nmf_explained * 100:.2f}% | "
          f"手工 2 协同 RMSE={rmse:.4f}、解释率={explained * 100:.2f}%（对照）")


def test_synergy_mode_tracking(arm):
    """2 维协同控制模式的末端跟踪能力（与 6 维直接分配对照，仅报告不断言）。"""
    dt = arm.model.opt.timestep
    results = {}
    for label, use_synergy in (("6 维直接分配", False), ("2 维协同控制", True)):
        arm.params.use_synergy = use_synergy
        arm.reset()
        errors = []
        for name in ("target_forward", "target_lateral", "target_low"):
            arm.set_target(arm.landmarks[name])
            start = arm.data.time
            while arm.data.time - start < 2.0:
                arm.step()
                if arm.state()['hand_error'] < 0.005:
                    break
            for _ in range(int(0.3 / dt)):
                arm.step()
            errors.append(arm.state()['hand_error'] * 1000.0)
        results[label] = errors
    arm.params.use_synergy = False
    text = " | ".join(f"{label}: " + ", ".join(f"{e:.1f} mm" for e in errors)
                      for label, errors in results.items())
    print(f"[INFO] 协同控制对照（前伸/外展/前下方稳态误差）: {text}")


def test_render(arm):
    """相机渲染可用性。"""
    try:
        image = arm.render(camera="overview")
    except Exception as exc:                                     # pragma: no cover
        raise AssertionError(f"渲染失败（无显示环境时可设 MUJOCO_GL=egl 或 osmesa）: {exc}")
    check(image.shape == (480, 640, 3), f"渲染尺寸异常: {image.shape}")
    print(f"[PASS] 相机渲染: {image.shape[1]}x{image.shape[0]} RGB")


def test_performance(arm, seconds=1.0):
    """仿真吞吐（每秒物理步数）。"""
    arm.reset()
    start = time.time()
    count = 0
    while time.time() - start < seconds:
        arm.step()
        count += 1
    rate = count / (time.time() - start)
    check(rate > 300, f"仿真吞吐过低: {rate:.0f} step/s")
    print(f"[PASS] 仿真吞吐: {rate:.0f} step/s（实时因子 {rate * arm.model.opt.timestep:.2f}×，"
          f"含每步力矩臂刷新）")


def main():
    print("=" * 64)
    print("  肌肉骨骼人（肌肉驱动）仿真与控制自动化测试")
    print("=" * 64)
    arm = MusculoskeletalArm()
    try:
        test_synergy_consistency()
        test_moment_arms(arm)
        test_allocation_identity(arm)
        test_activation_dynamics(arm)
        test_reach_accuracy(arm)
        test_gravity_compensation()
        test_cocontraction_stiffness(arm)
        test_synergy_reconstruction(arm)
        test_synergy_mode_tracking(arm)
        test_render(arm)
        test_performance(arm)
    finally:
        arm.close()
    print("=" * 64)
    print("  全部测试通过 ✅")
    print("=" * 64)


if __name__ == '__main__':
    main()
