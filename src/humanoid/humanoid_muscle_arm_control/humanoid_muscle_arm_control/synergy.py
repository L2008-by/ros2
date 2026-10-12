# -*- coding: utf-8 -*-
"""
synergy.py —— 肌肉协同（muscle synergy）降维
===========================================

肌肉骨骼系统的标志性结论：中枢神经系统并不逐个控制几十块肌肉，而是通过少量
**协同（synergy）** 基底线性组合来产生动作。本模块提供三件事：

1. **手工设计的 2 组协同**（``SYNERGY_BASIS``）：把 6 块肌肉的激活压到 2 维控制量，
   用于"键盘只控 2 个协同"的低维控制模式；
2. **投影**：把 6 维激活投影回 2 维协同系数（最小二乘 + 非负截断）；
3. **NMF 提取**：对仿真记录的激活矩阵做非负矩阵分解，得到数据驱动的协同基底，
   并给出重构误差 / 方差解释率——用于文档中的"6 维 vs 2 维"对比分析。

肌肉顺序与 :mod:`sim_core` 的 ``MUSCLE_NAMES`` 严格一致：
``(shoulder_flexor, shoulder_extensor, shoulder_abductor,
   shoulder_adductor, elbow_flexor, elbow_extensor)``
"""

from __future__ import annotations

import numpy as np

#: 肌肉顺序（必须与 sim_core.MUSCLE_NAMES 一致；test_standalone 里有断言校验）
MUSCLE_ORDER = ("shoulder_flexor", "shoulder_extensor", "shoulder_abductor",
                "shoulder_adductor", "elbow_flexor", "elbow_extensor")

SYNERGY_NAMES = ("前伸协同", "屈曲内收协同")

#: 手工协同基底（2 × 6），每行和为 1：系数 1.0 表示该协同的最大募集
SYNERGY_BASIS = np.array([
    # 前伸：肩屈 + 伸肘 + 少量外展/内收平衡
    [0.45, 0.05, 0.05, 0.05, 0.05, 0.35],
    # 屈曲内收：屈肘 + 肩内收 + 肩屈
    [0.15, 0.05, 0.05, 0.25, 0.45, 0.05],
], dtype=float)


def synthesis(coefficients):
    """2 维协同系数 → 6 维肌肉激励（线性组合，截断到 [0, 1]）。"""
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    if coefficients.size != SYNERGY_BASIS.shape[0]:
        raise ValueError(f"协同系数应为 {SYNERGY_BASIS.shape[0]} 维，收到 {coefficients.size} 维")
    return np.clip(coefficients @ SYNERGY_BASIS, 0.0, 1.0)


def project(activation, nonnegative=True):
    """6 维肌肉激活 → 2 维协同系数（最小二乘解，可选非负截断）。"""
    activation = np.asarray(activation, dtype=float).reshape(-1)
    if activation.size != SYNERGY_BASIS.shape[1]:
        raise ValueError(f"激活向量应为 {SYNERGY_BASIS.shape[1]} 维，收到 {activation.size} 维")
    basis_t = SYNERGY_BASIS.T                                  # (6, 2)
    gram = basis_t.T @ basis_t                                 # (2, 2)
    coefficients = np.linalg.solve(gram + 1e-9 * np.eye(gram.shape[0]), basis_t.T @ activation)
    if nonnegative:
        coefficients = np.clip(coefficients, 0.0, 1.0)
    return coefficients


def reconstruction_error(activation_matrix):
    """用 2 维协同重构整段激活记录，返回 (均方根误差, 解释方差比)。

    解释方差比的分母是**去均值后的方差**（标准 VAF 定义），因此对手工基底可能出现
    负值（说明该基底还不如"用均值代表一切"）——文档中同时给出数据驱动 NMF 的结果。
    """
    data = np.asarray(activation_matrix, dtype=float)          # (时间, 6)
    if data.ndim != 2 or data.shape[1] != SYNERGY_BASIS.shape[1]:
        raise ValueError("激活矩阵形状应为 (样本数, 6)")
    coefficients = np.stack([project(row, nonnegative=False) for row in data])   # (T, 2)
    reconstructed = coefficients @ SYNERGY_BASIS
    residual = data - reconstructed
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    total = float(np.sum((data - data.mean(axis=0)) ** 2))
    explained = 1.0 - float(np.sum(residual ** 2)) / max(total, 1e-12)
    return rmse, explained


def nmf(activation_matrix, rank=2, iterations=600, seed=0):
    """乘法更新法非负矩阵分解，返回 ``(W, H)``：``激活 ≈ W @ H``。

    W: (6, rank) 肌肉协同基底；H: (rank, 样本数) 协同时间曲线。
    仅用 numpy 实现，避免额外依赖。
    """
    data = np.asarray(activation_matrix, dtype=float).T         # (6, T)
    rng = np.random.default_rng(seed)
    w = rng.random((data.shape[0], rank)) + 0.1
    h = rng.random((rank, data.shape[1])) + 0.1
    eps = 1e-9
    for _ in range(iterations):
        h = h * (w.T @ data) / (w.T @ w @ h + eps)
        w = w * (data @ h.T) / (w @ h @ h.T + eps)
    return w, h
