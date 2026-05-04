#pragma once

#include <ga_circular_fields_planner/Cost.hpp>
#include <gafro/gafro.hpp>
#include <sackmesser/FactoryClass.hpp>

namespace ga_circular_fields_planner
{

    class PathLengthCost : public Cost
    {
      public:
        PathLengthCost(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~PathLengthCost() = default;

        double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const;

        std::string getName() const override { return "path_length_cost"; }

        double getWeight() const;

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double weight;
        } config_;
    };

}  // namespace ga_circular_fields_planner