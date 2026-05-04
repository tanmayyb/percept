@PACKAGE_INIT@


# Include the exported CMake file
get_filename_component(ga_circular_fields_planner_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# This macro enables usage of find_dependency().
# https://cmake.org/cmake/help/v3.11/module/CMakeFindDependencyMacro.html
include(CMakeFindDependencyMacro)
find_package(Eigen3 REQUIRED)

# Declare the used packages in order to communicate the requirements upstream.
if(NOT TARGET ga_circular_fields_planner::ga_circular_fields_planner)
    include("${ga_circular_fields_planner_CMAKE_DIR}/ga_circular_fields_planner-config-targets.cmake")
    include("${ga_circular_fields_planner_CMAKE_DIR}/ga_circular_fields_planner-packages.cmake")
else()
    set(BUILD_TARGET ga_circular_fields_planner::ga_circular_fields_planner)

    get_target_property(TARGET_INCLUDE_DIRS ${BUILD_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    set(TARGET_INCLUDE_DIRS "${TARGET_INCLUDE_DIRS}" CACHE PATH "${BUILD_TARGET} include directories")
    list(APPEND ga_circular_fields_planner_INCLUDE_DIRS ${TARGET_INCLUDE_DIRS})
endif()

check_required_components(ga_circular_fields_planner)