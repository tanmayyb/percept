#pragma once

#include <ga_circular_fields_planner/Force.hpp>

namespace ga_circular_fields_planner
{

    class RepellerForce : public Force
    {
      public:
        RepellerForce();

        virtual ~RepellerForce();

        gafro::Wrench<double> computeForce(const Agent *agent,                  //
                                           const Agent::State &state,           //
                                           const gafro::Motor<double> &target,  //
                                           const Environment &environment) const;

      protected:
      private:
    };

}  // namespace ga_circular_fields_planner