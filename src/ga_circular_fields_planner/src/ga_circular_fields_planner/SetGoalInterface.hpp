#pragma once

#include <ga_circular_fields_planner/CircularFieldPlanner.hpp>
#include <Eigen/Dense>


namespace ga_circular_fields_planner
{

    struct SetGoalRequest
    {
        Eigen::Vector<double, 7> joint_positions;
    };

    struct SetGoalResponse
    {
        bool success;
    };

}  // namespace ga_circular_fields_planner