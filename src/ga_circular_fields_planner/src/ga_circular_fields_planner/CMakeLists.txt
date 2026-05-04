find_package(Eigen3 REQUIRED)
find_package(gafro REQUIRED)
find_package(gafro_robot_descriptions REQUIRED)
find_package(sackmesser REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_BUILD_TYPE Debug)


option(DISABLE_PLANNER_TIME_LOGGING         "Disable time logging for the planner"        OFF)
option(DISABLE_PLANNER_COST_LOGGING         "Disable cost logging for the planner"        OFF)
option(DISABLE_PLANNER_BEST_AGENT_LOGGING   "Disable best agent logging for the planner"  OFF)


add_library(${PROJECT_NAME} SHARED
    Agent.cpp
    PointmassAgent.cpp
    ManipulatorAgent.cpp
    CircularFieldPlanner.cpp
    AttractorForce.cpp
    RepellerForce.cpp
    Costs.cpp
)


target_compile_options(${PROJECT_NAME}
  PUBLIC
    # "-Og"
  "-O3"
    ${OpenMP_CXX_FLAGS}
)

# LOGGING FLAGS
if(DISABLE_PLANNER_TIME_LOGGING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LOG_TIMING_DISABLE)
endif()

if(DISABLE_PLANNER_COST_LOGGING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LOG_COSTS_DISABLE)
endif()

if(DISABLE_PLANNER_BEST_AGENT_LOGGING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LOG_BEST_AGENT_DISABLE)
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${${PROJECT_NAME}_VERSION}
  SOVERSION 1
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION 1
)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<INSTALL_INTERFACE:${${PROJECT_NAME}_PREFIX_INCLUDE}>
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/..
    PRIVATE
        $<$<BOOL:${ITT_ENABLED}>:${ITT_INCLUDE_PATH}>
)

target_include_directories(${PROJECT_NAME} 
	SYSTEM PRIVATE
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    gafro::gafro
    gafro_robot_descriptions::gafro_robot_descriptions
    sackmesser::sackmesser
  PRIVATE
    OpenMP::OpenMP_CXX
)

install(
	FILES 
    Preproc.hpp
    Agent.hpp
    PointmassAgent.hpp
    ManipulatorAgent.hpp
    CircularFieldPlanner.hpp
    ObstacleForceInterface.hpp
    ObstacleDistanceInterface.hpp
    SetGoalInterface.hpp
    AttractorForce.hpp
    RepellerForce.hpp
    Costs.hpp
	DESTINATION 
		${${PROJECT_NAME}_PREFIX_INCLUDE}/${PROJECT_NAME} 
	COMPONENT 
		Devel
)