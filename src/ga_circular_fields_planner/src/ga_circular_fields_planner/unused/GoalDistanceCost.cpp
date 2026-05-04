#include <ga_circular_fields_planner/GoalDistanceCost.hpp>
#include <sackmesser/Configurations.hpp>

namespace ga_circular_fields_planner
{

    GoalDistanceCost::GoalDistanceCost(const sackmesser::Interface::Ptr &interface, const std::string &name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/goal_distance_cost/");
    }

    double GoalDistanceCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        double cost = gafro::Motor<double>(agent->getPath().back().reverse() * target).log().vector().norm();

        return cost;
    }

    bool GoalDistanceCost::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "weight", &weight, true);
    }

    double GoalDistanceCost::getWeight() const
    {
        return config_.weight;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::GoalDistanceCost, "goal_distance_cost")