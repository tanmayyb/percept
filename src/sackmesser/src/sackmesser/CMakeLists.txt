add_library(sackmesser SHARED)
add_library(sackmesser::sackmesser ALIAS sackmesser)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${${PROJECT_NAME}_VERSION}
  SOVERSION 1
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION 1
)

target_compile_options(
	${PROJECT_NAME}
		PUBLIC
    "-Wall"
    "-Wextra"
    "-Werror=vla"
    "-Wno-unused-function"
    "-Wno-missing-braces"
    "-Wno-unknown-pragmas"
    "-Wno-parentheses"
    "-pedantic"
    "-Wconversion"
    "-Werror=pedantic"
    "-O2"
)

target_sources(sackmesser
	PRIVATE
	    Callbacks.cpp
	    Configurations.cpp
	    Logger.cpp
	    Interface.cpp
	    Logger.cpp
	    Timer.cpp
	    UtilityFunctions.cpp
)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<INSTALL_INTERFACE:${${PROJECT_NAME}_PREFIX_INCLUDE}>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>/..
)

configure_file(${CMAKE_CURRENT_LIST_DIR}/sackmesser_package_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/sackmesser/sackmesser_package_config.hpp")

install(
	FILES 
		"${CMAKE_CURRENT_BINARY_DIR}/sackmesser/sackmesser_package_config.hpp" 
	    Interface.hpp

			Logger.hpp
			
			Configuration.hpp
			Configurations.hpp
			Configurations.hxx

	    Callbacks.hpp
	    Callbacks.hxx

	    Factory.hpp
	    Factory.hxx
	    FactoryClass.hpp
	    FactoryClass.hxx

	    Counter.hpp
	    Counter.hxx
	    
	    Timer.hpp
	DESTINATION 
		${${PROJECT_NAME}_PREFIX_INCLUDE}/${PROJECT_NAME} 
	COMPONENT 
		Devel
)