#pragma once

#include <ga_circular_fields_planner/Force.hpp>
#include <ga_circular_fields_planner/ObstacleForceInterface.hpp>
#include <sackmesser/Callbacks.hpp>


namespace ga_circular_fields_planner
{
    template <const char* UID>
    class RepellerForce : public Force
    {
    public:
        RepellerForce(const sackmesser::Interface::Ptr &interface, const std::string &name) 
            : interface_(interface), uid_name_(UID)
        {
            config_ = interface->getConfigurations()->load<Configuration>(name + "/" + uid_name_ + "/");
        }

        virtual ~RepellerForce() = default;

        gafro::Wrench<double> computeForce(const Agent::State &state, 
                                           const gafro::Motor<double> &target) const override
        {
            return computeForce(state, 0.0, target);
        }

        gafro::Wrench<double> computeForce(const Agent::State &state, 
                                           const double &agent_radius,
                                           const gafro::Motor<double> &target) const override
        {
            ObstacleForceRequest request;
            request.agent_pose = state.getPose();
            request.agent_velocity = state.getVelocity();
            request.target_pose = target;
            request.agent_radius = agent_radius;
            request.detect_shell_radius = config_.detect_shell_radius;
            request.k_force = config_.k_force;
            request.max_allowable_force = config_.max_allowable_force;

            ObstacleForceResponse response;

            if (interface_->getCallbacks()->execute("get_" + uid_name_, response, request))
            {
                return config_.k_gain * response.wrench;
            }

            return gafro::Wrench<double>::Zero();
        }

        double getMaxForce() const override
        {
          return config_.k_gain * config_.max_allowable_force;
        }

    private:
        sackmesser::Interface::Ptr interface_;
        std::string uid_name_;

        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server) override
            {
                return server->loadParameter(ns + "k_gain", &k_gain, true) &&
                       server->loadParameter(ns + "detect_shell_radius", &detect_shell_radius, true) &&
                       server->loadParameter(ns + "k_force", &k_force, true) &&
                       server->loadParameter(ns + "max_allowable_force", &max_allowable_force, true);
            }

            double k_gain;
            double detect_shell_radius;
            double k_force;
            double max_allowable_force;
        } config_;
    };
}