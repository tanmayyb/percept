find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)
find_package(OpenMP REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(vfe_service_tester test_vfe.cpp)

target_link_libraries(vfe_service_tester
	PUBLIC  
		rclcpp::rclcpp
	percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
)

install(
	TARGETS
  	vfe_service_tester
  DESTINATION 
		lib/${PROJECT_NAME}
)

add_executable(planner_service_tester test_planner.cpp)

target_link_libraries(planner_service_tester
	PUBLIC  
		rclcpp::rclcpp
		percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
)

install(
	TARGETS
  	planner_service_tester
  DESTINATION 
		lib/${PROJECT_NAME}
)


add_executable(parallel_service_tester parallel_service_tester.cpp)

target_link_libraries(parallel_service_tester
  PUBLIC  
    rclcpp::rclcpp
    percept_interfaces::percept_interfaces__rosidl_typesupport_cpp
  PRIVATE
    OpenMP::OpenMP_CXX
)

install(
	TARGETS
  	parallel_service_tester
  DESTINATION 
		lib/${PROJECT_NAME}
)


# Joint State Node Configuration Copy GUI
install(PROGRAMS
  joint_state_node.py
  manipulator_service_test.py
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
