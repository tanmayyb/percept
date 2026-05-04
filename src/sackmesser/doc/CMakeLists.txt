find_package(Doxygen)

if(${DOXYGEN_FOUND})
    set(DOXYGEN_INPUT_DIRS ${CMAKE_CURRENT_LIST_DIR}/..)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)

    set(DOXYGEN_PROJECT_NAME "${PROJECT_NAME}")
    set(DOXYGEN_PROJECT_VERSION "${${PROJECT_NAME}_VERSION}")
    set(DOXYGEN_PROJECT_BRIEF "This library is a collection of useful tools for programming.")
    set(DOXYGEN_INPUT_DIRS "${CMAKE_CURRENT_LIST_DIR}/MAINPAGE.md")

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(sackmesser_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif(${DOXYGEN_FOUND})