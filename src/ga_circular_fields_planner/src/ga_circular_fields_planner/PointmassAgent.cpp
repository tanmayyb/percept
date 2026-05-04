#include <ga_circular_fields_planner/PointmassAgent.hpp>

namespace ga_circular_fields_planner
{
    PointmassAgent::PointmassAgent(const sackmesser::Interface::Ptr &interface, const std::string &name) : Agent(interface, name) {}

    std::shared_ptr<PointmassAgent::State> PointmassAgent::updateState(const std::shared_ptr<State> &state, const gafro::Wrench<double> &force,
                                                                       const double delta_t)
    {
        gafro::Twist<double> acceleration = inertia_(force);

        gafro::Twist<double> velocity = state->getVelocity() + 0.5 * delta_t * acceleration;

        gafro::Motor<double> pose = state->getPose() * gafro::Motor<double>::exp(delta_t * velocity);

        return std::make_shared<State>(pose, velocity);
    }

    void PointmassAgent::updateStateInPlace(
      std::shared_ptr<State> &state, const gafro::Wrench<double> &force, const double delta_t)
    {
        gafro::Twist<double> acceleration = inertia_(force);

        gafro::Twist<double> velocity = state->getVelocity() + 0.5 * delta_t * acceleration;
        
        gafro::Motor<double> pose = state->getPose() * gafro::Motor<double>::exp(delta_t * velocity);

        state->setPose(pose);

        state->setVelocity(velocity);  
    }

}  // namespace ga_circular_fields_planner