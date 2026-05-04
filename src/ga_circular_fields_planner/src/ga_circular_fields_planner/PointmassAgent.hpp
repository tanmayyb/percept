/*
   Class for circular field agents
*/
#pragma once

#include <ga_circular_fields_planner/Agent.hpp>

namespace ga_circular_fields_planner
{

    class PointmassAgent : public Agent
    {
      public:
        PointmassAgent(const sackmesser::Interface::Ptr &interface, const std::string &name);

        virtual ~PointmassAgent() = default;

        std::shared_ptr<State> updateState(const std::shared_ptr<State> &state, const gafro::Wrench<double> &force, const double delta_t);
        
        void updateStateInPlace(std::shared_ptr<State> &state, const gafro::Wrench<double> &force, const double delta_t) override;
    };
}  // namespace ga_circular_fields_planner
