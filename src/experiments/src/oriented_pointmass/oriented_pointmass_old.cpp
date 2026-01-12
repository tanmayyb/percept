#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ga_circular_fields_planner/CircularFieldPlanner.hpp>
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

std::vector<Obstacle> readObstacles(const YAML::Node &node)
{
    std::vector<Obstacle> obstacles;
    for (const auto &obstacle : node)
    {
        std::string name = obstacle["name"].as<std::string>();
        Eigen::Vector3d position = readVector3d(obstacle["position"]);
        Eigen::Vector3d velocity = readVector3d(obstacle["velocity"]);
        double radius = obstacle["radius"].as<double>();
        bool is_dynamic = obstacle["is_dynamic"].as<bool>();
        double angular_speed = obstacle["angular_speed"] ? obstacle["angular_speed"].as<double>() : 0.0;

        obstacles.emplace_back(name, position, velocity, radius, is_dynamic, angular_speed);
    }
    return obstacles;
}

int main(int argc, char **argv)
{
    auto interface = sackmesser_ros::Interface::create(argc, argv, "oriented_pointmass", "experiments");

    std::string package_path = ament_index_cpp::get_package_share_directory("experiments");
    YAML::Node start_goal = YAML::LoadFile(package_path + "/oriented_pointmass/start_goal.yaml");
    YAML::Node obstacles_yaml = YAML::LoadFile(package_path + "/oriented_pointmass/obstacles_1.yaml");

    Eigen::Vector3d start_pos = readVector3d(start_goal["start_pos"]);
    Eigen::Vector3d goal_pos = readVector3d(start_goal["goal_pos"]);
    Eigen::Quaterniond start_orientation = readQuaternion(start_goal["start_orientation"]);
    Eigen::Quaterniond goal_orientation = readQuaternion(start_goal["goal_orientation"]);

    std::vector<Obstacle> obstacles = readObstacles(obstacles_yaml["obstacles"]);

    Environment environment(obstacles);

    CircularFieldPlanner cf_planner(interface, "cf_planner");

    auto state =
      std::make_shared<Agent::State>(gafro::Motor<double>(start_pos, start_orientation), gafro::Twist<double>({ 0.0, 0.0, 0.0, 0.01, 0.0, 0.0 }));

    gafro::Motor<double> target(goal_pos, goal_orientation);
    std::vector<gafro::Motor<double>> trajectory;

    int k = 0;

    interface->loop([&]() {
        environment.step();

        cf_planner.plan(state, target, environment);

        gafro::Wrench<double> force = cf_planner.getBestAgent()->computeForce(state, target, environment);

        state = cf_planner.getBestAgent()->updateState(state, force, 0.01);

        if (++k % 10 == 0)
        {
            trajectory.push_back(state->getPose());
        }

        std::vector<gafro::Sphere<double>> obstacles_msg;

        for (const auto &o : environment.getObstacles())
        {
            obstacles_msg.push_back(
              gafro::Sphere<double>(gafro::Point<double>(o.getPosition().x(), o.getPosition().y(), o.getPosition().z()), o.getRadius()));
        }

        interface->getCallbacks()->invoke("trajectory", trajectory);
        interface->getCallbacks()->invoke("obstacles", obstacles_msg);
        interface->getCallbacks()->invoke("target", target);
        interface->getCallbacks()->invoke("pose", state->getPose());
    });

    return 1;
}
