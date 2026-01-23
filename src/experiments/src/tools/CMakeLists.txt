find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)
find_package(OpenMP REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(vfe_service_tester test_vfe.cpp)
add_executable(parallel_service_tester parallel_service_tester.cpp)
add_executable(planner_service_tester test_planner.cpp)
add_executable(task_sequencer task_sequencer.cpp)
add_executable(shm_monitor shm_monitor.cpp)



target_link_libraries(vfe_service_tester
	PUBLIC  
		rclcpp::rclcpp
	percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
)

target_link_libraries(planner_service_tester
	PUBLIC  
		rclcpp::rclcpp
		percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
)

target_link_libraries(parallel_service_tester
  PUBLIC  
    rclcpp::rclcpp
    percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
  PRIVATE
    OpenMP::OpenMP_CXX
)

target_link_libraries(task_sequencer
  PUBLIC  
    rclcpp::rclcpp
    percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
)

target_link_libraries(shm_monitor
  PUBLIC  
    rclcpp::rclcpp
    percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)


install(
	TARGETS
    vfe_service_tester
    planner_service_tester	
    parallel_service_tester
    task_sequencer
    shm_monitor
  DESTINATION 
		lib/${PROJECT_NAME}
)

install(PROGRAMS
  joint_state_node.py
  task_planner.py
  simp_conf.py
  panda_control.py
  DESTINATION lib/${PROJECT_NAME}
)

# Launch Files
install(
	FILES
		launch/vfe_service_test.py
		launch/planner_service_test.py
		launch/parallel_service_test.py
	DESTINATION
		share/${PROJECT_NAME}/tools/
)
