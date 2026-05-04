@PACKAGE_INIT@

get_filename_component(sackmesser_ros2_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET sackmesser_ros2::sackmesser_ros2)
    include("${sackmesser_ros2_CMAKE_DIR}/sackmesser_ros2-config-targets.cmake")
    include("${sackmesser_ros2_CMAKE_DIR}/sackmesser_ros2-packages.cmake")
endif()

get_target_property(sackmesser_ros2_INCLUDE_DIRS sackmesser_ros2::sackmesser_ros2 INTERFACE_INCLUDE_DIRECTORIES)

check_required_components(sackmesser_ros2)