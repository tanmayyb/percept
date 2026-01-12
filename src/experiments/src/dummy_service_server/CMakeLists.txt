add_executable(dummy_service_server dummy_service_server.cpp)

target_compile_options(dummy_service_server PUBLIC -Wall -Wextra -Wpedantic -Werror)

ament_target_dependencies(dummy_service_server
  PUBLIC
    sackmesser_ros2
)

ament_export_dependencies(
  sackmesser_ros2
)

target_link_libraries(dummy_service_server
  PUBLIC
    message_interface
)

install(
  TARGETS 
    dummy_service_server
  DESTINATION
    lib/${PROJECT_NAME}
)

install(
  FILES
    dummy_service_server_config.yaml
  DESTINATION 
    share/${PROJECT_NAME}/dummy_service_server/
)