// #include <ga_circular_fields_planner/ObstacleDistanceCost.hpp>

#include <ga_circular_fields_planner/ObstacleDistanceInterface.hpp>
#include <sackmesser/Callbacks.hpp>
#include <sackmesser/Configurations.hpp>
#include <ga_circular_fields_planner/Costs.hpp>

namespace ga_circular_fields_planner
{

    // Obstacle Distance Cost
    ObstacleDistanceCost::ObstacleDistanceCost(const sackmesser::Interface::Ptr &interface, const std::string &name) : interface_(interface)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/obstacle_distance_cost/");
    }

    double ObstacleDistanceCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        double cost = 0.0;

        ObstacleDistanceRequest request;
        request.agent_pose = agent->getPath().back();

        request.radius = config_.radius;

        ObstacleDistanceResponse response;

        if (interface_->getCallbacks()->execute("get_min_obstacle_distance", response, request))
        {
            cost = 1 - (std::min(config_.radius, response.distance) / config_.radius);

            // cost = response.distance;
        }       

        return cost;
    }

    bool ObstacleDistanceCost::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return (server->loadParameter(ns + "weight", &weight, true) && \
                server->loadParameter(ns + "radius", &radius, true));
    }

    double ObstacleDistanceCost::getWeight() const
    {
        return config_.weight;
    }


    // Goal Distance Cost
    GoalDistanceCost::GoalDistanceCost(const sackmesser::Interface::Ptr &interface, const std::string &name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/goal_distance_cost/");
    }

    double GoalDistanceCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        double cost = std::min( config_.radius, 
                                gafro::Motor<double>(
                                  agent->getPath().back().reverse() * target
                                ).log().vector().norm()
                              ) / config_.radius; // min(max_radius, dist(p_agent, p_goal)) / max_radius < = 1 always

        return cost;
    }

    bool GoalDistanceCost::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        // return server->loadParameter(ns + "weight", &weight, true);

        return (server->loadParameter(ns + "weight", &weight, true) && \
                server->loadParameter(ns + "radius", &radius, true));
    }

    double GoalDistanceCost::getWeight() const
    {
        return config_.weight;
    }


    // Path Length Cost
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


    // Trajectory Smoothness Cost
    TrajectorySmoothnessCost::TrajectorySmoothnessCost(const sackmesser::Interface::Ptr &interface, const std::string &name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/trajectory_smoothness_cost/");
    }

    double TrajectorySmoothnessCost::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
    {
        auto path = agent->getPath();

        int k = path.size();

        if ( k >= 4){
          return (
              path[k-1].getTranslator().toTranslationVector() \
            - 3 * path[k-2].getTranslator().toTranslationVector() \
            + 3 * path[k-3].getTranslator().toTranslationVector() \
            - path[k-4].getTranslator().toTranslationVector()
          ).squaredNorm(); // third-order backward finite difference of the position vector
        }

        return 0.0;
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


REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::ObstacleDistanceCost,      "obstacle_distance_cost");
REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::GoalDistanceCost,          "goal_distance_cost");
REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::PathLengthCost,            "path_length_cost");
REGISTER_CLASS(ga_circular_fields_planner::Cost, ga_circular_fields_planner::TrajectorySmoothnessCost,  "trajectory_smoothness_cost");