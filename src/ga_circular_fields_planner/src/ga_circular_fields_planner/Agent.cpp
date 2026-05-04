#include <ga_circular_fields_planner/Agent.hpp>
#include <ga_circular_fields_planner/Force.hpp>
#include <sackmesser/Callbacks.hpp>
#include <sackmesser/Configurations.hpp>
#include <sackmesser/Timer.hpp>

namespace ga_circular_fields_planner
{
    Agent::Agent(const sackmesser::Interface::Ptr &interface, const std::string &name) : interface_(interface), name_(name)
    {
        config_ = interface->getConfigurations()->load<Configuration>(name + "/");

        double ii = config_.mass * config_.radius * config_.radius;
        inertia_ = gafro::Inertia<double>(config_.mass, ii, 0.0, 0.0, ii, 0.0, ii);

        for (const std::string &force : config_.forces)
        {
            forces_.push_back(Force::getFactory()->createShared(force, interface, name));
        }
    }

    gafro::Wrench<double> Agent::computeForce(const std::shared_ptr<State> &state, const gafro::Motor<double> &target)
    {
        gafro::Wrench<double> force = gafro::Wrench<double>::Zero();

        for (const auto &field_force : forces_)
        {
          force += field_force->computeForce((*state.get()), config_.radius, target);
        }

        return force;
    }

    std::shared_ptr<Agent::State> Agent::planStep(const std::shared_ptr<State> &state, const gafro::Motor<double> &target, const double delta_t)
    {
        // send request to fields computer and get force
        gafro::Wrench<double> force = computeForce(state, target);
        // update state
        auto updated_state = updateState(state, force, delta_t);
        // add to trajectory
        trajectory_.push_back(updated_state->getPose());
        // update state
        return updated_state;
    }

    void Agent::updateTrajectory(const gafro::Motor<double> &pose)
    {
        trajectory_.push_back(pose);
    }

    const std::vector<gafro::Motor<double>> &Agent::getPath() const
    {
        return trajectory_;
    }

    double Agent::getDistFromGoal(const gafro::Motor<double> &pose, const gafro::Motor<double> &target) const
    {
        return (pose.getTranslator().toTranslationVector() - target.getTranslator().toTranslationVector()).norm();
    }

    bool Agent::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
    {
        return server->loadParameter(ns + "mass", &mass, true) &&                            //
               server->loadParameter(ns + "radius", &radius, true) &&                        //
               server->loadParameter(ns + "forces", &forces);
    }

    Agent::State::State(const gafro::Motor<double> &pose, const gafro::Twist<double> &velocity) : pose_(pose), velocity_(velocity) {}

    void Agent::State::setPose(const gafro::Motor<double> &pose)
    {
        pose_ = pose;
    }

    void Agent::State::setVelocity(const gafro::Twist<double> &velocity)
    {
        velocity_ = velocity;
    }

    std::shared_ptr<Agent::State> Agent::State::copy() const
    {
        return std::make_shared<State>(pose_, velocity_);
    }

    const gafro::Motor<double> &Agent::State::getPose() const
    {
        return pose_;
    }

    const gafro::Twist<double> &Agent::State::getVelocity() const
    {
        return velocity_;
    }

    const Agent::Configuration &Agent::getConfiguration() const
    {
        return config_;
    }

    const std::string &Agent::getName() const
    {
        return name_;
    }

    const double Agent::getMaxAcceleration()
    { 
      double force_mag;
      
      for (const auto &field_force : forces_)
      {
        force_mag += field_force->getMaxForce();
      }
      
      return force_mag / config_.mass;
    }

}  // namespace ga_circular_fields_planner