#include <ga_circular_fields_planner/RepellerForce.hpp>

namespace ga_circular_fields_planner
{

    RepellerForce::RepellerForce() = default;

    RepellerForce::~RepellerForce() = default;

    gafro::Wrench<double> RepellerForce::computeForce(const Agent *agent,                  //
                                                      const Agent::State &state,           //
                                                      const gafro::Motor<double> &target,  //
                                                      const Environment &environment) const
    {
        gafro::Wrench<double> force = gafro::Wrench<double>::Zero();

        Eigen::Vector3d total_repel_force{ 0.0, 0.0, 0.0 };
        Eigen::Vector3d robot_obstacle_vec{ environment.getObstacles().back().getPosition() - state.getPose().getTranslator().toTranslationVector() };
        Eigen::Vector3d rel_vel{ state.getVelocity().getLinear().vector() - environment.getObstacles().back().getVelocity() };
        Eigen::Vector3d dist_vec = -robot_obstacle_vec;
        double dist_obs{ dist_vec.norm() - (agent->getConfiguration().radius + environment.getObstacles().back().getRadius()) };
        dist_obs = std::max(dist_obs, 1e-5);
        Eigen::Vector3d repel_force{ 0.0, 0.0, 0.0 };
        if (dist_obs < agent->getConfiguration().detect_shell_radius)
        {
            Eigen::Vector3d obs_to_robot = state.getPose().getTranslator().toTranslationVector() - environment.getObstacles().back().getPosition();
            obs_to_robot.normalize();
            repel_force = agent->getConfiguration().k_repel_force * obs_to_robot *
                          (1.0 / dist_obs - 1.0 / agent->getConfiguration().detect_shell_radius) / (dist_obs * dist_obs);
        }
        total_repel_force += repel_force;

        force = force + gafro::Wrench<double>::Linear(total_repel_force);

        return force;
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::RepellerForce, "repeller_force")