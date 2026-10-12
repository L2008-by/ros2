#!/bin/bash
# 肌肉骨骼人（肌肉驱动）MuJoCo 仿真与键盘运动控制一键运行脚本（课程规范主入口）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source ROS2 环境（若存在）
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Source 工作空间环境（若已编译）
if [ -f ~/ros2_ws/install/setup.bash ]; then
    source ~/ros2_ws/install/setup.bash
fi

# 无显示环境下 MuJoCo 渲染回退（有显示环境时无需设置）
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
fi

# 切换到包根目录并确保当前源码树优先加载
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=================================================="
echo "  肌肉骨骼人（肌肉驱动）仿真 - 模块一键运行脚本"
echo "=================================================="

# 运行主入口
python3 main.py "$@"
