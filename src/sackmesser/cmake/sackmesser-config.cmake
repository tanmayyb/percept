@PACKAGE_INIT@

get_filename_component(sackmesser_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET sackmesser::sackmesser)
    include("${sackmesser_CMAKE_DIR}/sackmesser-config-targets.cmake")
    include("${sackmesser_CMAKE_DIR}/sackmesser-packages.cmake")
endif()

get_target_property(sackmesser_INCLUDE_DIRS sackmesser::sackmesser INTERFACE_INCLUDE_DIRECTORIES)

check_required_components(sackmesser)