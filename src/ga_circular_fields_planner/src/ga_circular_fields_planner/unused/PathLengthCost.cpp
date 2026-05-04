#include <ga_circular_fields_planner/PathLengthCost.hpp>
#include <sackmesser/Configurations.hpp>

namespace ga_circular_fields_planner
{

    PathLengthCost::PathLengthCost(const sackmesser::Interface::Ptr &interface, const std::string &name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/path_length_cost/");
    }

    double PathLengthCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        double cost = 0.0;

        int k = agent->getPath().size();
        if ( k > 1) {
            cost += gafro::Motor<double>(agent->getPath().back().reverse() * agent->getPath()[k - 2]).log().vector().norm();
        }

        // return config_.weight * cost;

        return cost;
    }

    bool PathLengthCost::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "weight", &weight, true);
    }

    double PathLengthCost::getWeight() const
    {
        return config_.weight;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::PathLengthCost, "path_length_cost")