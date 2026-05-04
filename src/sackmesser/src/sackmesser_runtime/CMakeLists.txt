find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME}_runtime SHARED)
add_library(${PROJECT_NAME}::${PROJECT_NAME}_runtime ALIAS ${PROJECT_NAME}_runtime)

target_sources(${PROJECT_NAME}_runtime
	PRIVATE
	    ConfigurationsYAML.cpp
	    Interface.cpp
)

target_include_directories(${PROJECT_NAME}_runtime
	PUBLIC
		$<INSTALL_INTERFACE:${${PROJECT_NAME}_PREFIX_INCLUDE}>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
		$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/..
		${YAML_CPP_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_runtime
	PUBLIC
		${PROJECT_NAME}
	PRIVATE
		${YAML_CPP_LIBRARIES}
) 

install(
	FILES 
		ConfigurationsYAML.hpp
	    Interface.hpp
	DESTINATION 
		${${PROJECT_NAME}_PREFIX_INCLUDE}/${PROJECT_NAME}_runtime
	COMPONENT 
		Devel
)

add_subdirectory(impl)