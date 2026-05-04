cmake_minimum_required(VERSION 3.5)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(sackmesser)

cmake_policy(SET CMP0079 NEW)

set(${PROJECT_NAME}_VERSION_MAJOR 1)
set(${PROJECT_NAME}_VERSION_MINOR 1)
set(${PROJECT_NAME}_VERSION_PATCH 0)

set(${PROJECT_NAME}_VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(${PROJECT_NAME}_PREFIX_PACKAGE share/${PROJECT_NAME}/cmake)
set(${PROJECT_NAME}_PREFIX_INCLUDE include/${PROJECT_NAME})

add_subdirectory(src)
add_subdirectory(doc)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/sackmesser-config.cmake
    "${PROJECT_BINARY_DIR}/sackmesser-config.cmake"
    INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    VERSION ${${PROJECT_NAME}_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/${PROJECT_NAME}-packages.cmake "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-packages.cmake")

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_runtime EXPORT ${PROJECT_NAME}-config-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION ${${PROJECT_NAME}_PREFIX_INCLUDE}/${PROJECT_NAME}
)

install(
  FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake" 
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-packages.cmake"
  DESTINATION 
    ${${PROJECT_NAME}_PREFIX_PACKAGE}
  COMPONENT 
    Devel
)

install(EXPORT ${PROJECT_NAME}-config-targets
    FILE ${PROJECT_NAME}-config-targets.cmake
    NAMESPACE sackmesser::
    DESTINATION ${${PROJECT_NAME}_PREFIX_PACKAGE}
)

export(EXPORT ${PROJECT_NAME}-config-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-targets.cmake"
    NAMESPACE sackmesser::
)

export(PACKAGE sackmesser)