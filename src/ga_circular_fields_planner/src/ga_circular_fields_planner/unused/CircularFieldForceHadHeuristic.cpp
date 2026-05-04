#include <ga_circular_fields_planner/CircularFieldForceHadHeuristic.hpp>

namespace ga_circular_fields_planner
{

    Eigen::Vector3d CircularFieldForceHadHeuristic::currentVector(const Agent::State &state,                //
                                                                  const gafro::Motor<double> & /*target*/,  //
                                                                  const Obstacle &obstacle,                 //
                                                                  const Eigen::Vector3d &field_rotation_vec) const
    {
        Eigen::Vector3d cfagent_to_obs{ obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector() };
        cfagent_to_obs.normalize();
        Eigen::Vector3d current = cfagent_to_obs.cross(field_rotation_vec);
        current.normalize();
        return current;
    }

    Eigen::Vector3d CircularFieldForceHadHeuristic::calculateRotationVector(const Agent::State &state,           //
                                                                            const gafro::Motor<double> &target,  //
                                                                            const Obstacle &obstacle) const
    {
        Eigen::Vector3d obs_pos = obstacle.getPosition();
        Eigen::Vector3d goal_vec{ target.getTranslator().toTranslationVector() - state.getPose().getTranslator().toTranslationVector() };
        Eigen::Vector3d rob_obs_vec{ obs_pos - state.getPose().getTranslator().toTranslationVector() };
        Eigen::Vector3d d =
          state.getPose().getTranslator().toTranslationVector() + goal_vec * (rob_obs_vec.dot(goal_vec) / pow(goal_vec.norm(), 2)) - obs_pos;
        Eigen::Vector3d rot_vec = d.cross(goal_vec) / (d.cross(goal_vec)).norm();
        return rot_vec;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::CircularFieldForceHadHeuristic, "circular_field_force_had_heuristic")