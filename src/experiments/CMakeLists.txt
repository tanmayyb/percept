cmake_minimum_required(VERSION 3.11)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(experiments)

find_package(ament_cmake REQUIRED)

find_package(gafro REQUIRED)
find_package(sackmesser_ros2 REQUIRED)

add_subdirectory(src)

ament_package()