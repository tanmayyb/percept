find_package(percept_interfaces REQUIRED)
find_package(sackmesser_ros2 REQUIRED)
find_package(gafro_ros2 REQUIRED)
find_package(ga_circular_fields_planner REQUIRED)
find_package(gafro_robot_descriptions REQUIRED)

add_library(message_interface SHARED
    ForceServiceClient.cpp
    ObstacleDistanceServiceClient.cpp
    SetGoalServiceServer.cpp
)

target_compile_options(message_interface
    PUBLIC
        "-march=native"
        "-msse2"
        "-O3"
)

target_include_directories(message_interface
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/..
)

target_link_libraries(message_interface
  PUBLIC
    sackmesser_ros2::sackmesser_ros2
    gafro::gafro
    ga_circular_fields_planner::ga_circular_fields_planner
)

ament_target_dependencies(message_interface PUBLIC percept_interfaces gafro_ros2 sackmesser_ros2)

install(TARGETS message_interface DESTINATION lib/${PROJECT_NAME})