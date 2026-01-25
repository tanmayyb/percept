#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ga_circular_fields_planner/CircularFieldPlanner.hpp>
#include <ga_circular_fields_planner/PointmassAgent.hpp>
#include <sackmesser_ros2/Interface.hpp>

using namespace ga_circular_fields_planner;

Eigen::Vector3d readVector3d(const YAML::Node &node)
{
    auto vec = node.as<std::vector<double>>();
    return Eigen::Vector3d(vec[0], vec[1], vec[2]);
}

Eigen::Quaterniond readQuaternion(const YAML::Node &node)
{
    auto quat = node.as<std::vector<double>>();
    return Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
}

int main(int argc, char **argv)
{
    auto interface = sackmesser_ros::Interface::create(argc, argv, "oriented_pointmass", "experiments");

    std::string package_path = ament_index_cpp::get_package_share_directory("experiments");
    YAML::Node start_goal = YAML::LoadFile(package_path + "/oriented_pointmass/start_goal.yaml");

    Eigen::Vector3d start_pos = readVector3d(start_goal["start_pos"]);
    Eigen::Vector3d goal_pos = readVector3d(start_goal["goal_pos"]);
    Eigen::Quaterniond start_orientation = readQuaternion(start_goal["start_orientation"]);
    Eigen::Quaterniond goal_orientation = readQuaternion(start_goal["goal_orientation"]);

    CircularFieldPlanner cf_planner(interface, "cf_planner");

    std::shared_ptr<Agent::State> state =
      std::make_shared<PointmassAgent::State>(
        gafro::Motor<double>(start_pos, start_orientation), gafro::Twist<double>({ 0.0, 0.0, 0.0, 0.1, 0.0, 0.0 }));

    gafro::Motor<double> target(goal_pos, goal_orientation);
    std::vector<gafro::Motor<double>> trajectory;

    cf_planner.startPlanning(state, target);

    int k = 0;

    std::thread simulation_thread = std::thread([&]() {
        auto next_wakeup = std::chrono::steady_clock::now();
        const std::chrono::milliseconds interval(10);

        while (interface->ok())
        {
            next_wakeup += interval;
            state = cf_planner.getStateUpdate(state);
            std::this_thread::sleep_until(next_wakeup);
        }
    });

    interface->loop([&]() {
				trajectory.push_back(state->getPose());  
				if (++k % 10 == 0)
				{
        	interface->getCallbacks()->invoke("trajectory", trajectory);
				}

        auto updated_target = cf_planner.getCurrentTarget();

        interface->getCallbacks()->invoke("target", updated_target);
        interface->getCallbacks()->invoke("pose", state->getPose());
    });

    return 0;
}
