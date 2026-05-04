#include <ga_circular_fields_planner/CircularFieldForceGoalObstacleHeuristic.hpp>

namespace ga_circular_fields_planner
{

    Eigen::Vector3d CircularFieldForceGoalObstacleHeuristic::currentVector(const Agent::State &state,                //
                                                                           const gafro::Motor<double> & /*target*/,  //
                                                                           const Obstacle &obstacle,                 //
                                                                           const Eigen::Vector3d &field_rotation_vec) const
    {
        Eigen::Vector3d cfagent_to_obs{ obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector() };
        cfagent_to_obs.normalize();
        Eigen::Vector3d current{ cfagent_to_obs.cross(field_rotation_vec) };
        current.normalize();
        return current;
    }

    Eigen::Vector3d CircularFieldForceGoalObstacleHeuristic::calculateRotationVector(const Agent::State &state,           //
                                                                                     const gafro::Motor<double> &target,  //
                                                                                     const Obstacle &obstacle) const
    {
        // Vector from active obstacle to the obstacle which is closest to the
        // active obstacle
        Eigen::Vector3d obstacle_vec = obstacle.getClosestNeighbour()->getPosition() - obstacle.getPosition();
        Eigen::Vector3d cfagent_to_obs{ obstacle.getPosition() - state.getPose().getTranslator().toTranslationVector() };
        cfagent_to_obs.normalize();
        // Current vector is perpendicular to obstacle surface normal and shows in
        // opposite direction of obstacle_vec
        Eigen::Vector3d obst_current{ (cfagent_to_obs * obstacle_vec.dot(cfagent_to_obs)) - obstacle_vec };
        Eigen::Vector3d goal_vec{ state.getPose().getTranslator().toTranslationVector() - target.getTranslator().toTranslationVector() };
        Eigen::Vector3d goal_current{ goal_vec - cfagent_to_obs * (cfagent_to_obs.dot(goal_vec)) };
        Eigen::Vector3d current{ goal_current.normalized() + obst_current.normalized() };

        if (current.norm() < 1e-10)
        {
            current << 0.0, 0.0, 1.0;
            // current = makeRandomVector();
        }
        current.normalize();
        Eigen::Vector3d rot_vec{ current.cross(cfagent_to_obs) };
        rot_vec.normalize();
        return rot_vec;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::CircularFieldForceGoalObstacleHeuristic,
               "circular_field_force_goal_obstacle_heuristic")