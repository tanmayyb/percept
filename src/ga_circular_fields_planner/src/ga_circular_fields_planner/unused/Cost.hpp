#pragma once

#include <ga_circular_fields_planner/Agent.hpp>
#include <gafro/gafro.hpp>
#include <sackmesser/FactoryClass.hpp>

namespace ga_circular_fields_planner
{

    class Cost : public sackmesser::FactoryClass<Cost, const sackmesser::Interface::Ptr &, const std::string &>
    {
      public:
        Cost() = default;

        virtual ~Cost() = default;

        virtual double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const = 0;

        virtual std::string getName() const = 0;

        virtual double getWeight() const = 0;

      protected:
      private:
    };

}  // namespace ga_circular_fields_planner