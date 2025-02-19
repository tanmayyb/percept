#!/bin/bash
source /opt/ros/humble/setup.bash
conda activate percept_env
colcon clean workspace -y
colcon build \
    --symlink-install \
    --base-paths \
        src/percept_core \
        src/percept_interfaces
source install/setup.bash