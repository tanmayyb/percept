#include <ga_circular_fields_planner/CircularFieldForce.hpp>

namespace ga_circular_fields_planner
{

    CircularFieldForce::CircularFieldForce() = default;

    CircularFieldForce::~CircularFieldForce() = default;

    gafro::Wrench<double> CircularFieldForce::computeForce(const Agent *agent,                  //
                                                           const Agent::State &state,           //
                                                           const gafro::Motor<double> &target,  //
                                                           const Environment &environment) const
    {
        if (agent->getDistFromGoal(state.getPose(), target) < agent->getConfiguration().approach_distance ||
            (state.getVelocity().getLinear().vector().norm() < 0.5 * agent->getConfiguration().max_velocity &&
             (state.getPose().getTranslator().toTranslationVector() - agent->getPath().front().getTranslator().toTranslationVector()).norm() < 0.2))
        {
            return gafro::Wrench<double>::Zero();
        }

        gafro::Wrench<double> force = gafro::Wrench<double>::Zero();

        for (const auto &obstacle : environment.getObstacles())
        {
            gafro::Vector<double> robot_obstacle_vec(obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector());
            gafro::Vector<double> relative_velocity(state.getVelocity().getLinear().vector() - obstacle.getVelocity());

            double obstacle_distance = robot_obstacle_vec.norm() - (agent->getConfiguration().radius + obstacle.getRadius());
            obstacle_distance = std::max(obstacle_distance, 1e-5);

            if (obstacle_distance < agent->getConfiguration().detect_shell_radius)
            {
                if (relative_velocity.norm() != 0)
                {
                    relative_velocity.normalize();

                    gafro::Vector<double> current = computeCurrent(state, target, obstacle);

                    gafro::Wrench<double>::Linear circular_field_force =
                      gafro::Scalar<double>(agent->getConfiguration().k_circular_force * std::exp(-obstacle_distance)) *
                      (((current ^ (relative_velocity)) * gafro::E123<double>(1.0)) ^ (relative_velocity)) * gafro::E0123<double>(-1.0);

                    force = force + circular_field_force;
                }
            }
        }

        return force;
    }

}  // namespace ga_circular_fields_planner