cmake_minimum_required(VERSION 3.11)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(sackmesser_ros2)

find_package(ament_cmake REQUIRED)

find_package(sackmesser REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(sackmesser_ros2 SHARED)
add_library(sackmesser_ros2::sackmesser_ros2 ALIAS sackmesser_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(sackmesser_ros2 PUBLIC -Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(src/sackmesser_ros2)

ament_target_dependencies(sackmesser_ros2
    PUBLIC
        rclcpp
        ament_index_cpp
        std_msgs
)

target_include_directories(sackmesser_ros2 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:include/>
)

ament_export_targets(sackmesser_ros2-config-targets HAS_LIBRARY_TARGET)

ament_export_dependencies(
    rclcpp
    ament_index_cpp
    std_msgs
    sackmesser
)

target_link_libraries(sackmesser_ros2
    PUBLIC
        sackmesser::sackmesser
        sackmesser::sackmesser_runtime
)

install(
    TARGETS sackmesser_ros2
    EXPORT sackmesser_ros2-config-targets
    LIBRARY DESTINATION lib
)

ament_package()