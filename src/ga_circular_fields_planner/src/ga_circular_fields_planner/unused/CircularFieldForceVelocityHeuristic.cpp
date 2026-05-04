#include <ga_circular_fields_planner/CircularFieldForceVelocityHeuristic.hpp>

namespace ga_circular_fields_planner
{

    Eigen::Vector3d CircularFieldForceVelocityHeuristic::currentVector(const Agent::State &state,           //
                                                                       const gafro::Motor<double> &target,  //
                                                                       const Obstacle &obstacle,            //
                                                                       const Eigen::Vector3d & /*field_rotation_vec*/) const
    {
        Eigen::Vector3d normalized_vel = state.getVelocity().getLinear().vector().normalized();
        Eigen::Vector3d normalized_obs_to_agent{ (obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector()).normalized() };
        Eigen::Vector3d current{ normalized_vel - (normalized_obs_to_agent * normalized_vel.dot(normalized_obs_to_agent)) };
        if (current.norm() < 1e-10)
        {
            current << 0.0, 0.0, 1.0;
            // current = makeRandomVector();
        }
        current.normalize();
        return current;
    }

    Eigen::Vector3d CircularFieldForceVelocityHeuristic::calculateRotationVector(const Agent::State & /*state*/,           //
                                                                                 const gafro::Motor<double> & /*target*/,  //
                                                                                 const Obstacle & /*obstacle*/) const
    {
        return { 0.0, 0.0, 1.0 };
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::CircularFieldForceVelocityHeuristic,
               "circular_field_force_velocity_heuristic")