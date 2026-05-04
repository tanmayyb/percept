#include <ga_circular_fields_planner/ObstacleForceInterface.hpp>
#include <ga_circular_fields_planner/APFHeuristicForce.hpp>
#include <sackmesser/Callbacks.hpp>

namespace ga_circular_fields_planner
{

    APFHeuristicForce::APFHeuristicForce(const sackmesser::Interface::Ptr &interface, const std::string &name) : interface_(interface)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/apf_heuristic_force/");
    }

    APFHeuristicForce::~APFHeuristicForce() = default;

    gafro::Wrench<double> APFHeuristicForce::computeForce(const Agent::State &state,  //
                                                               const gafro::Motor<double> &target) const
    {
        ObstacleForceRequest request;
        request.agent_pose = state.getPose();
        request.agent_velocity = state.getVelocity();
        request.target_pose = target;
        request.detect_shell_radius = config_.detect_shell_radius;
        request.k_force = config_.k_force;
        request.max_allowable_force = config_.max_allowable_force;

        ObstacleForceResponse response;

        if (interface_->getCallbacks()->execute("get_apf_heuristic_force", response, request))
        {
            return config_.k_gain * response.wrench;
        }

        return gafro::Wrench<double>::Zero();
    }

    bool APFHeuristicForce::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "k_gain", &k_gain, true) &&  //
               server->loadParameter(ns + "detect_shell_radius", &detect_shell_radius, true) &&
               server->loadParameter(ns + "k_force", &k_force, true) &&
               server->loadParameter(ns + "max_allowable_force", &max_allowable_force, true);
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::APFHeuristicForce, "apf_heuristic_force")