#include <ga_circular_fields_planner/CircularFieldForceGoalHeuristic.hpp>

namespace ga_circular_fields_planner
{

    Eigen::Vector3d CircularFieldForceGoalHeuristic::currentVector(const Agent::State &state,           //
                                                                   const gafro::Motor<double> &target,  //
                                                                   const Obstacle &obstacle,            //
                                                                   const Eigen::Vector3d & /*field_rotation_vec*/) const
    {
        Eigen::Vector3d goal_vec{ target.getTranslator().toTranslationVector() - state.getPose().getTranslator().toTranslationVector() };
        Eigen::Vector3d cfagent_to_obs{ obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector() };
        cfagent_to_obs.normalize();
        Eigen::Vector3d current{ goal_vec - cfagent_to_obs * (cfagent_to_obs.dot(goal_vec)) };

        if (current.norm() < 1e-10)
        {
            current << 0.0, 0.0, 1.0;
        }
        current.normalize();

        return current;
    }

    Eigen::Vector3d CircularFieldForceGoalHeuristic::calculateRotationVector(const Agent::State & /*state*/,           //
                                                                             const gafro::Motor<double> & /*target*/,  //
                                                                             const Obstacle & /*obstacle*/) const
    {
        return { 0.0, 0.0, 1.0 };
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::CircularFieldForceGoalHeuristic, "circular_field_force_goal_heuristic")