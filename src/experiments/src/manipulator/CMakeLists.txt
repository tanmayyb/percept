find_package(yaml-cpp REQUIRED)
find_package(ga_circular_fields_planner REQUIRED)
find_package(sackmesser_ros2 REQUIRED)
find_package(gafro_ros2 REQUIRED)
find_package(gafro_robot_descriptions REQUIRED)

add_executable(manipulator manipulator.cpp)

target_compile_options(manipulator PUBLIC -Wall -Wextra -Wpedantic -Werror)

ament_target_dependencies(manipulator
  PUBLIC
    sackmesser_ros2
    gafro_ros2
)

ament_export_dependencies(
  sackmesser_ros2
  gafro_ros2
)

target_link_libraries(manipulator
  PUBLIC
    ga_circular_fields_planner::ga_circular_fields_planner
    ${YAML_CPP_LIBRARIES}
    message_interface
)

install(
  TARGETS 
    manipulator
  DESTINATION
    lib/${PROJECT_NAME}
)

install(
  FILES
    launch/manipulator.rviz
    launch/manipulator_launch.py
    launch/rwe.py
    launch/sim_manip.py
    
    manipulator_config.yaml
    start_configuration.yaml
  DESTINATION 
    share/${PROJECT_NAME}/manipulator/
)