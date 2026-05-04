#include <ga_circular_fields_planner/CircularFieldForceObstacleHeuristic.hpp>

namespace ga_circular_fields_planner
{

    gafro::Vector<double> CircularFieldForceObstacleHeuristic::computeCurrent(const Agent::State &state,                //
                                                                              const gafro::Motor<double> & /*target*/,  //
                                                                              const Obstacle &obstacle) const
    {
        gafro::Vector<double> o1(obstacle.getPosition());
        gafro::Vector<double> o2(obstacle.getClosestNeighbour()->getPosition());
        gafro::Vector<double> p(state.getPose().getTranslator().toTranslationVector());

        gafro::Vector<double> o1p = (o1 - p).evaluate().normalized();
        gafro::Vector<double> o1o2 = (o2 - o1).evaluate().normalized();

        gafro::Rotor<double>::Generator b = ((o1p ^ o1o2) * o1p.inverse()) ^ o1p;

        return (b | o1p);
    }

}  // namespace ga_circular_fields_planner

REGISTER_CLASS(ga_circular_fields_planner::Force, ga_circular_fields_planner::CircularFieldForceObstacleHeuristic,
               "circular_field_force_obstacle_heuristic")