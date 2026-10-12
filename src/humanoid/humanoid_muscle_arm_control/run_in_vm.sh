#!/bin/bash
# ============================================================================
# 肌肉骨骼人（肌肉驱动）模块 —— 虚拟机一键验证 + 素材采集脚本
# ----------------------------------------------------------------------------
# 用法（在模块根目录，也就是有 main.py 的那一层）:
#     bash run_in_vm.sh                 # 结果输出到 ~/musculoskeletal_out
#     bash run_in_vm.sh /path/to/out    # 自定义输出目录
#     bash run_in_vm.sh <仓库>/docs/img/humanoid/humanoid_muscle_arm_control
#                                       # 直接产出到仓库配图目录（推荐）
#
# 脚本只做三件事：自检 → 自动化测试 → 采集素材（关键帧/动图/激活曲线/metrics.json），
# 不会安装任何软件、不改系统配置。
# ============================================================================
set -u

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HOME/musculoskeletal_out}"
ASSETS="$OUT/assets"
mkdir -p "$OUT" "$ASSETS"
cd "$PKG_DIR" || exit 1

echo "============================================================"
echo " 肌肉骨骼人（肌肉驱动）模块 —— 虚拟机验证"
echo " 模块目录: $PKG_DIR"
echo " 输出目录: $OUT"
echo "============================================================"

# ---------------------------------------------------------------- 环境信息
{
  echo "== 环境 =="
  uname -a
  python3 -c "import sys; print('python', sys.version.split()[0])"
  python3 -c "import mujoco, numpy; print('mujoco', mujoco.__version__, '| numpy', numpy.__version__)" 2>&1
  python3 -c "import yaml; print('pyyaml OK')" 2>&1 || echo "pyyaml 缺失（standalone 将使用内置默认参数）"
  python3 -c "import imageio; print('imageio OK')" 2>&1 || echo "imageio 缺失（跳过动图）"
  python3 -c "import matplotlib; print('matplotlib OK')" 2>&1 || echo "matplotlib 缺失（跳过激活曲线）"
  echo "DISPLAY=${DISPLAY:-<empty>}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<empty>}  MUJOCO_GL=${MUJOCO_GL:-<empty>}"
} 2>&1 | tee "$OUT/env.log"

# 无显示环境时给 MuJoCo 渲染一个回退后端（仅当变量为空时设置）
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${MUJOCO_GL:-}" ]; then
    export MUJOCO_GL=osmesa
    echo "[INFO] 无显示环境，已设置 MUJOCO_GL=osmesa" | tee -a "$OUT/env.log"
fi

# ---------------------------------------------------------------- 1) 自检
echo
echo "==> 1/3 模型与控制自检 (main.py --check)"
python3 main.py --check 2>&1 | tee "$OUT/check.log"
CHECK_RC=${PIPESTATUS[0]}

# ---------------------------------------------------------------- 2) 测试
echo
echo "==> 2/3 自动化测试 (main.py --test)"
python3 main.py --test 2>&1 | tee "$OUT/test.log"
TEST_RC=${PIPESTATUS[0]}

# ---------------------------------------------------------------- 3) 素材
echo
echo "==> 3/3 采集素材 (main.py --capture)"
python3 main.py --capture "$ASSETS" 2>&1 | tee "$OUT/capture.log"
CAPTURE_RC=${PIPESTATUS[0]}

# ---------------------------------------------------------------- 汇总
echo
echo "============================================================"
echo " 结果汇总"
echo "------------------------------------------------------------"
echo " 自检    exit=$CHECK_RC   → $OUT/check.log"
echo " 测试    exit=$TEST_RC    → $OUT/test.log"
echo " 采集    exit=$CAPTURE_RC → $OUT/capture.log"
echo "------------------------------------------------------------"
ls -lh "$ASSETS" 2>/dev/null
echo
if [ -f "$ASSETS/metrics.json" ]; then
    echo "===== metrics.json ====="
    cat "$ASSETS/metrics.json"
    echo "========================"
fi
echo
echo "下一步（二选一）："
echo "  A. 把上面 metrics.json + 三个 .log 的内容发给我（我据此回填文档实测数字）"
echo "  B. 直接把 $ASSETS 里的 6 个文件放进仓库的"
echo "     docs/img/humanoid/humanoid_muscle_arm_control/"
echo "     （scene_home/reach/side/up.png、musculoskeletal_demo.gif、muscle_activation.png）"
echo "============================================================"
