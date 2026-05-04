#pragma once

#include <ga_circular_fields_planner/Agent.hpp>
#include <gafro/gafro.hpp>

namespace ga_circular_fields_planner
{

    struct ObstacleForceRequest
    {
        gafro::Motor<double> agent_pose;
        gafro::Twist<double> agent_velocity;
        gafro::Motor<double> target_pose;
        double agent_radius;
        double detect_shell_radius;
        double k_force;
        double max_allowable_force;
    };

    struct ObstacleForceResponse
    {
        gafro::Wrench<double> wrench;
    };

}  // namespace ga_circular_fields_planner