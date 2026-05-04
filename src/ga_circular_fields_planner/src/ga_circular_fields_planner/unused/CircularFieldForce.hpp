#pragma once

#include <ga_circular_fields_planner/Force.hpp>

namespace ga_circular_fields_planner
{

    class CircularFieldForce : public Force
    {
      public:
        CircularFieldForce(const sackmesser::Interface::Ptr &);

        virtual ~CircularFieldForce();

        gafro::Wrench<double> computeForce(const Agent *agent,                  //
                                           const Agent::State &state,           //
                                           const gafro::Motor<double> &target,  //
                                           const Environment &environment) const;

      protected:
      private:
        virtual gafro::Vector<double> computeCurrent(const Agent::State &state,           //
                                                     const gafro::Motor<double> &target,  //
                                                     const Obstacle &obstacle) const = 0;
    };

}  // namespace ga_circular_fields_planner