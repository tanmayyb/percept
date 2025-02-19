#!/bin/bash
set -e

# set flags
export LDFLAGS='-Wl,--no-as-needed'

# removing old build and install folders
echo "removing old build directories"
rm -rf build install log

# build workspace
echo "building workspace"
colcon build --symlink-install

# source setup
echo "attempting to source"
source install/setup.bash
echo "done"