FROM osrf/ros:rolling-desktop
# FROM osrf/ros:humble-desktop # not supported yet

RUN mkdir /ros2_ws
WORKDIR /ros2_ws

RUN echo "source /opt/ros/rolling/setup.bash" >> ~/.bashrc
# RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc # not supported yet