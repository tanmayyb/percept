#pragma once

#include <ga_circular_fields_planner/CircularFieldForce.hpp>

namespace ga_circular_fields_planner
{

    class CircularFieldForceObstacleHeuristic : public CircularFieldForce
    {
      public:
        gafro::Vector<double> computeCurrent(const Agent::State &state,           //
                                             const gafro::Motor<double> &target,  //
                                             const Obstacle &obstacle) const override;
    };

}  // namespace ga_circular_fields_planner
