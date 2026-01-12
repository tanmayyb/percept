#include <sackmesser_ros2/Interface.hpp>

int main(int argc, char **argv)
{
    auto interface = sackmesser_ros::Interface::create(argc, argv, "dummy_service_server", "experiments");

    interface->loop([]() {});

    return 1;
}