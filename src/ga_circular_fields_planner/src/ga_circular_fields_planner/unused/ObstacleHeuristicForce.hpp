#pragma once

#include <ga_circular_fields_planner/Force.hpp>

namespace ga_circular_fields_planner
{

    class ObstacleHeuristicForce : public Force
    {
      public:
        ObstacleHeuristicForce(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~ObstacleHeuristicForce();

        gafro::Wrench<double> computeForce(const Agent::State &state,  //
                                           const gafro::Motor<double> &target) const;

      protected:
      private:
        sackmesser::Interface::Ptr interface_;

        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double k_gain;

            double detect_shell_radius;
            double k_force;
            double max_allowable_force;
            
        } config_;
    };

}  // namespace ga_circular_fields_planner