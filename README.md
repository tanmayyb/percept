# Perception Pipeline

Goal: Fast (~30Hz) perception pipline for workspace approximation research for heuristic-based multi-agent planning framework (PMAF) by Laha et al 2023. Support both sim/real-world perception for multiple cameras.

![Velocity Heuristic](imgs/250218_velocity_heuristic.png)
![Rviz PointCloud](imgs/250121_rviz_pointclouds.png)
![Rviz Primitives](imgs/250121_rviz_primitives.png)


## Requirements


| Hardware Stack                                     |
| ---------------------------------------------------- |
| 1. CUDA-capable PC (Our Setup: i7 8000U + RTX3070) |
| 2. Intel Realsense 435i (for Real-World)           |
| 3. Franka Emika Panda Robot Arms (for Real-World)  |


| Software Stack                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------- |
| 1. Ubuntu 22.04 Jammy Jellyfish ([iso](https://releases.ubuntu.com/focal/https:/))                                      |
| 2. ROS 2 Humble ([debian](http://wiki.ros.org/noetic/Installation/Debianhttps:/))                                   |
| 3. Conda + CUDA ([instructions](https://x.com/jeremyphoward/status/1697435241152127369))                            |
| 4. librealsenseSDK ([debian](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)) |
| 5. Sim (Under Development)      |

## Setup

- Install aforementioned requirements and clone this git repo
    ```
    git clone https://github.com/tanmayyb/percept.git
    cd percept
    git submodule update --init --recursive
    ```
- Create a virtual environment and install all dependencies and libraries
    ```
    ./setup.sh
    ```

- Build the workspace
    ```
    . build.sh
    ```

- Run the perception pipeline
    ```
    ros2 launch percept rs_static.py
    ```
- Or source the environment if not already sourced
    ```
    . env.sh
    ```

## Credits

### Collaborators

- Riddhiman Laha
- Tinayu Ren

### Projects

- CuPoch
- Open3D
- Numba
- Cupy
- CoppeliaSim
- PyRep (by stepjam)

## Performance


Performance w/o RBS:

![Profiling Perception Pipeline](imgs/250121_pipeline_perf-min.jpg)
