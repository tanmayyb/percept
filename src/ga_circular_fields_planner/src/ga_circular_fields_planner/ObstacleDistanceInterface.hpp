#pragma once

#include <ga_circular_fields_planner/Agent.hpp>
#include <gafro/gafro.hpp>

namespace ga_circular_fields_planner
{

    struct ObstacleDistanceRequest
    {
        gafro::Motor<double> agent_pose;

        float radius;
    };

    struct ObstacleDistanceResponse
    {
        double distance;
    };

}  // namespace ga_circular_fields_planner