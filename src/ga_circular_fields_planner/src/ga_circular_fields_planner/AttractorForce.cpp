#include <ga_circular_fields_planner/AttractorForce.hpp>
#include <sackmesser/Configurations.hpp>

namespace ga_circular_fields_planner
{

    AttractorForce::AttractorForce(const sackmesser::Interface::Ptr &interface, const std::string &name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/attractor_force/");

        stiffness_ = gafro::Inertia<double>(config_.k_stiffness_linear,             //
                                            config_.k_stiffness_angular, 0.0, 0.0,  //
                                            config_.k_stiffness_angular, 0.0,       //
                                            config_.k_stiffness_angular);

        damping_ = gafro::Inertia<double>(config_.k_damping_linear,             //
                                          config_.k_damping_angular, 0.0, 0.0,  //
                                          config_.k_damping_angular, 0.0,       //
                                          config_.k_damping_angular);
    }

    AttractorForce::~AttractorForce() = default;

    gafro::Wrench<double> AttractorForce::computeForce(const Agent::State &state,  //
                                                       const gafro::Motor<double> &target) const
    {
        gafro::Twist<double> twist = gafro::Motor<double>(state.getPose().reverse() * target).log();

        return config_.k_gain * (stiffness_(twist) - damping_(state.getVelocity()));
    }

    gafro::Wrench<double> AttractorForce::computeForce(const Agent::State &state,
                                                      const double &agent_radius,
                                                      const gafro::Motor<double> &target) const
    {
        // Ignore agent_radius and call the standard implementation
        return this->computeForce(state, target);
    }

    bool AttractorForce::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "k_gain", &k_gain, true) &&                          //
               server->loadParameter(ns + "k_damping_linear", &k_damping_linear, true) &&      //
               server->loadParameter(ns + "k_damping_angular", &k_damping_angular, true) &&    //
               server->loadParameter(ns + "k_stiffness_linear", &k_stiffness_linear, true) &&  //
               server->loadParameter(ns + "k_stiffness_angular", &k_stiffness_angular, true);
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::AttractorForce, "attractor_force")