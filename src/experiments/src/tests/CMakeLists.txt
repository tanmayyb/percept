find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(percept_interfaces REQUIRED)

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

install(
	FILES
		vfe_service_test.py
		planner_service_test.py
	DESTINATION
		share/${PROJECT_NAME}/tests/
)