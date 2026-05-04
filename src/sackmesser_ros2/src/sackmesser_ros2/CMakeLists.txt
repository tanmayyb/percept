target_sources(sackmesser_ros2
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/Interface.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Configurations.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Publisher.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Subscriber.cpp
)

add_subdirectory(publisher)

install(
    FILES
        Interface.hpp
        Subscriber.hpp
        Subscriber.hxx
        Publisher.hpp
        Publisher.hxx
        CallbackClient.hpp
        CallbackClient.hxx
        CallbackServer.hpp
        CallbackServer.hxx
    DESTINATION
        include/sackmesser_ros2
)
