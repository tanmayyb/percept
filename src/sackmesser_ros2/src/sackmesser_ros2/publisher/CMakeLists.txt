target_sources(sackmesser_ros2
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/Bool.cpp
        ${CMAKE_CURRENT_LIST_DIR}/String.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Float64.cpp
)

install(
    FILES
        Bool.hpp
        String.hpp
    DESTINATION
        include/sackmesser_ros2/publisher
)