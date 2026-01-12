#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ga_circular_fields_planner/CircularFieldPlanner.hpp>
#include <ga_circular_fields_planner/ManipulatorAgent.hpp>
#include <gafro_robot_descriptions/FrankaEmikaRobot.hpp>
#include <sackmesser_ros2/Interface.hpp>

using namespace ga_circular_fields_planner;


Eigen::Vector<double, 7> readVector7d(const YAML::Node &node)
{
    auto vec = node.as<std::vector<double>>();
    return Eigen::Vector<double, 7>(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6]);
}


int main(int argc, char **argv)
{
    auto interface = sackmesser_ros::Interface::create(argc, argv, "manipulator", "experiments");

    gafro::FrankaEmikaRobot<double> panda;

    std::string package_path = ament_index_cpp::get_package_share_directory("experiments");
    YAML::Node start_goal = YAML::LoadFile(package_path + "/manipulator/start_configuration.yaml");

    // Joint	Min	Max
    // 0	-2.8973	2.8973
    // 1	-1.7628	1.7628
    // 2	-2.8973	2.8973
    // 3	-3.0718	-0.0698
    // 4	-2.8973	2.8973
    // 5	-0.0175	3.7525
    // 6	-2.8973	2.8973

    // auto q0 = panda.getRandomConfiguration();
    // auto qt = panda.getRandomConfiguration();

    Eigen::Vector<double, 7> q0 = readVector7d(start_goal["start_configuration"]);
    Eigen::Vector<double, 7> qt = readVector7d(start_goal["goal_configuration"]);

    gafro::Motor<double> initial_pose = panda.getEEMotor(q0);
    gafro::Motor<double> target = panda.getEEMotor(qt);

    CircularFieldPlanner cf_planner(interface, "cf_planner");

    std::shared_ptr<Agent::State> state = std::make_shared<ManipulatorAgent::State>(q0, Eigen::Vector<double, 7>::Zero(), initial_pose, gafro::Twist<double>::Zero());

    std::vector<gafro::Motor<double>> trajectory;

    cf_planner.startPlanning(state, target);

    int k = 0;

    std::thread simulation_thread = std::thread([&]() {
        while (interface->ok())
        {
            state = cf_planner.getStateUpdate(state, target);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    interface->loop([&]() {
        if (++k % 10 == 0)
        {
            trajectory.push_back(state->getPose());
        }

        interface->getCallbacks()->invoke("trajectory", trajectory);
        interface->getCallbacks()->invoke("target", target);
        interface->getCallbacks()->invoke("pose", state->getPose());
        interface->getCallbacks()->invoke("robot", Eigen::MatrixXd(std::dynamic_pointer_cast<ManipulatorAgent::State>(state)->getJointPosition()), gafro::Motor<double>());
    });

    return 0;
}