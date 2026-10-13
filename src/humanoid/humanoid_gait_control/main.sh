#!/usr/bin/env bash
# 双足步态模块 standalone 运行脚本（无需 ROS）
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ "$1" = "--test" ]; then
    exec "$PY" main.py --test
elif [ "$1" = "--check" ]; then
    exec "$PY" main.py --check
elif [ "$1" = "--capture" ]; then
    exec "$PY" main.py --capture "${2:-./docs_img}" ${3:+"$3"}
else
    exec "$PY" main.py --demo "$@"
fi
