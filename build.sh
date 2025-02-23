#!/bin/bash

# Ensure auto_generated_scene.yaml exists
mkdir -p ./src/percept_core/assets/benchmark_scenes
touch ./src/percept_core/assets/benchmark_scenes/auto_generated_scene.yaml

source /opt/ros/humble/setup.bash
conda activate percept_env
colcon clean workspace -y
colcon build \
    --symlink-install \
    --base-paths \
        src/percept_core \
        src/percept_interfaces \
        src/mp_eval
source install/setup.bash