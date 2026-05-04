#include <ga_circular_fields_planner/ObstacleDistanceCost.hpp>
#include <ga_circular_fields_planner/ObstacleDistanceInterface.hpp>
#include <sackmesser/Callbacks.hpp>
#include <sackmesser/Configurations.hpp>

namespace ga_circular_fields_planner
{

    ObstacleDistanceCost::ObstacleDistanceCost(const sackmesser::Interface::Ptr &interface, const std::string &name) : interface_(interface)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/obstacle_distance_cost/");
    }

    double ObstacleDistanceCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        double cost = 0.0;

        ObstacleDistanceRequest request;
        request.agent_pose = agent->getPath().back();

        ObstacleDistanceResponse response;

        if (interface_->getCallbacks()->execute("get_min_obstacle_distance", response, request))
        {
            // cost = config_.weight * response.distance;
            cost = response.distance;
        }       

        return cost;
    }

    bool ObstacleDistanceCost::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "weight", &weight, true);
    }

    double ObstacleDistanceCost::getWeight() const
    {
        return config_.weight;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::ObstacleDistanceCost, "obstacle_distance_cost")