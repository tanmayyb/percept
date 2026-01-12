find_package(yaml-cpp REQUIRED)
find_package(ga_circular_fields_planner REQUIRED)
find_package(sackmesser_ros2 REQUIRED)
find_package(gafro_ros2 REQUIRED)
find_package(gafro_robot_descriptions REQUIRED)

add_executable(oriented_pointmass oriented_pointmass.cpp)

target_compile_options(oriented_pointmass PUBLIC -Wall -Wextra -Wpedantic -Werror)

ament_target_dependencies(oriented_pointmass
  PUBLIC
    sackmesser_ros2
    gafro_ros2
)

ament_export_dependencies(
  sackmesser_ros2
  gafro_ros2
)

target_link_libraries(oriented_pointmass
  PUBLIC
    ga_circular_fields_planner::ga_circular_fields_planner
    ${YAML_CPP_LIBRARIES}
    message_interface
)

install(
  TARGETS 
    oriented_pointmass
  DESTINATION
    lib/${PROJECT_NAME}
)

install(
  FILES
    oriented_pointmass.rviz
    oriented_pointmass_launch.py
    oriented_pointmass_launch_headless.py
    oriented_pointmass_config.yaml
    obstacles_1.yaml
    obstacles_2.yaml
    start_goal.yaml
  DESTINATION 
    share/${PROJECT_NAME}/oriented_pointmass/
)