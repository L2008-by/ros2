# -*- coding: utf-8 -*-
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="humanoid_gait_control",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/humanoid_gait_control"]),
        ("share/" + "humanoid_gait_control", ["package.xml"]),
        ("share/humanoid_gait_control", ["main.py", "main.sh"]),
        ("share/humanoid_gait_control/config", ["config/sim_params.yaml"]),
        ("share/humanoid_gait_control/models", ["models/humanoid_gait.xml"]),
        ("share/humanoid_gait_control/launch", ["launch/main.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@todo.todo",
    description="双足步态生成与行走控制（LIPM 步态 + 腿部 IK + 关节 PD，MuJoCo 仿真）",
    long_description=long_description,
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = humanoid_gait_control.mujoco_sim_node:main",
            "gait_teleop_node = humanoid_gait_control.gait_teleop_node:main",
        ],
    },
)
