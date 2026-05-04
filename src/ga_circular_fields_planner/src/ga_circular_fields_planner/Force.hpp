#pragma once

#include <ga_circular_fields_planner/Agent.hpp>
#include <gafro/gafro.hpp>
#include <sackmesser/FactoryClass.hpp>

namespace ga_circular_fields_planner
{

    class Force : public sackmesser::FactoryClass<Force, const sackmesser::Interface::Ptr &, const std::string &>
    {
      public:
        Force() = default;

        virtual ~Force() = default;

        virtual gafro::Wrench<double> computeForce(const Agent::State &state,  //
                                                   const gafro::Motor<double> &target) const = 0;

        virtual gafro::Wrench<double> computeForce(const Agent::State &state, 
                                                  const double &agent_radius,
                                                  const gafro::Motor<double> &target) const 
        {
            return computeForce(state, target);
        }
        
        virtual double getMaxForce() const = 0;

      protected:
      private:
    };

}  // namespace ga_circular_fields_planner