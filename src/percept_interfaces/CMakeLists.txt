cmake_minimum_required(VERSION 3.18)
project(percept_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # services
    "srv/AgentStateToCircForce.srv"
    "srv/AgentPoseToMinObstacleDist.srv"
    "srv/SetGoal.srv"
  # messages
    "msg/Pointcloud1M.msg"
    "msg/Pointcloud100.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_package()