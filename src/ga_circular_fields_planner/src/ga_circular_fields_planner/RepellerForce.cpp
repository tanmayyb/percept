#include <ga_circular_fields_planner/RepellerForce.hpp>

namespace ga_circular_fields_planner
{
    // Define unique symbols for template parameters
    extern constexpr char apf_uid[] = "apf_heuristic_force";
    extern constexpr char vel_uid[] = "velocity_heuristic_force";
    extern constexpr char goal_uid[] = "goal_heuristic_force";
    extern constexpr char goalobs_uid[] = "goalobstacle_heuristic_force";
    extern constexpr char obs_uid[] = "obstacle_heuristic_force";
    extern constexpr char rand_uid[] = "random_heuristic_force";
}

// Register derived classes using the template
REGISTER_CLASS(ga_circular_fields_planner::Force, 
               ga_circular_fields_planner::RepellerForce<ga_circular_fields_planner::apf_uid>, 
               "apf_heuristic_force");

REGISTER_CLASS(ga_circular_fields_planner::Force, 
               ga_circular_fields_planner::RepellerForce<ga_circular_fields_planner::vel_uid>, 
               "velocity_heuristic_force");

REGISTER_CLASS(ga_circular_fields_planner::Force, 
               ga_circular_fields_planner::RepellerForce<ga_circular_fields_planner::goal_uid>, 
               "goal_heuristic_force");

REGISTER_CLASS(ga_circular_fields_planner::Force, 
               ga_circular_fields_planner::RepellerForce<ga_circular_fields_planner::goalobs_uid>, 
               "goalobstacle_heuristic_force");

REGISTER_CLASS(ga_circular_fields_planner::Force, 
               ga_circular_fields_planner::RepellerForce<ga_circular_fields_planner::obs_uid>, 
               "obstacle_heuristic_force");

REGISTER_CLASS(ga_circular_fields_planner::Force, 
               ga_circular_fields_planner::RepellerForce<ga_circular_fields_planner::rand_uid>, 
               "random_heuristic_force");