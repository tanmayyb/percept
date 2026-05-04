#pragma once

#include <ga_circular_fields_planner/CircularFieldForce.hpp>

namespace ga_circular_fields_planner
{

    class CircularFieldForceRandomHeuristic : public CircularFieldForce
    {
      public:
        Eigen::Vector3d currentVector(const Agent::State &state,           //
                                      const gafro::Motor<double> &target,  //
                                      const Obstacle &obstacle,            //
                                      const Eigen::Vector3d &field_rotation_vec) const override;

        Eigen::Vector3d calculateRotationVector(const Agent::State &state,           //
                                                const gafro::Motor<double> &target,  //
                                                const Obstacle &obstacle) const override;
    };

}  // namespace ga_circular_fields_planner
