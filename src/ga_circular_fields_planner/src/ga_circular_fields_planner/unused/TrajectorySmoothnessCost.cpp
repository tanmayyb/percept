#include <ga_circular_fields_planner/TrajectorySmoothnessCost.hpp>
#include <sackmesser/Configurations.hpp>

namespace ga_circular_fields_planner
{

    TrajectorySmoothnessCost::TrajectorySmoothnessCost(const sackmesser::Interface::Ptr &interface, const std::string &name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/trajectory_smoothness_cost/");
    }

    double TrajectorySmoothnessCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        double cost = 0.0;

        int k = agent->getPath().size();
        if ( k > 2){
            auto path = agent->getPath();
            
            // f_smooth = (1/2)sum(  | q_(k-1) - 2q_(k-2) + q_(k-3) |^2 ), i = k-2
            cost = (path[k-1].getTranslator().toTranslationVector() - 2*path[k-2].getTranslator().toTranslationVector() + path[k-3].getTranslator().toTranslationVector()).squaredNorm()/2.0;
        }

        return cost;
    }

    bool TrajectorySmoothnessCost::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "weight", &weight, true);
    }

    double TrajectorySmoothnessCost::getWeight() const
    {
        return config_.weight;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::TrajectorySmoothnessCost, "trajectory_smoothness_cost")