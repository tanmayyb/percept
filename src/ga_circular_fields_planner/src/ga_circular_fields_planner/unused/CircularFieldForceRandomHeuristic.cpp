#include <ga_circular_fields_planner/CircularFieldForceRandomHeuristic.hpp>

namespace ga_circular_fields_planner
{

    Eigen::Vector3d CircularFieldForceRandomHeuristic::currentVector(const Agent::State &state,           //
                                                                     const gafro::Motor<double> &target,  //
                                                                     const Obstacle &obstacle,            //
                                                                     const Eigen::Vector3d &field_rotation_vec) const
    {
        Eigen::Vector3d cfagent_to_obs{ obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector() };
        cfagent_to_obs.normalize();
        Eigen::Vector3d current = cfagent_to_obs.cross(field_rotation_vec);
        current.normalize();
        return current;
    }

    Eigen::Vector3d CircularFieldForceRandomHeuristic::calculateRotationVector(const Agent::State &state,           //
                                                                               const gafro::Motor<double> &target,  //
                                                                               const Obstacle & /*obstacle*/) const
    {
        Eigen::Vector3d goal_vec{ target.getTranslator().toTranslationVector() - state.getPose().getTranslator().toTranslationVector() };
        goal_vec.normalize();
        Eigen::Vector3d rot_vec = goal_vec.cross(Eigen::Vector3d::Random());
        // Eigen::Vector3d rot_vec = goal_vec.cross(random_vecs_.at(obstacle_id));
        return rot_vec;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::CircularFieldForceRandomHeuristic,
               "circular_field_force_random_heuristic")