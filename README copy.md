1. clone the repository

    git clone git@github.com:loewt/ga_circular_fields_experiments.git

2. initialize all submodules

    git submodule update --init --recursive

3. build the workspace

    colcon build --symlink-install

4. launch experiment for oriented pointmass

    ros2 launch experiments oriented_pointmass_launch.py

4. launch experiment for manipulator

    ros2 launch experiments manipulator_launch.py


# Docker

Note: this directory is mounted as a volume in the container. the changes made to this directory in the container will be reflected in the host machine.

allow docker access to X11

    xhost +local:docker


build the container 

    docker-compose up -d

connect to the container

    docker exec -it gavec_planner bash

start the container if it is not running

    docker start gavec_planner

inside the container:
    
    . build.sh
    ros2 launch experiments oriented_pointmass_launch.py
    ros2 launch experiments manipulator_launch.py