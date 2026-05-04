#pragma once

#include <ga_circular_fields_planner/Force.hpp>

namespace ga_circular_fields_planner
{

    class AttractorForce : public Force
    {
      public:
        AttractorForce(const sackmesser::Interface::Ptr &, const std::string &);

        virtual ~AttractorForce();

        gafro::Wrench<double> computeForce(const Agent::State &state,  //
                                           const gafro::Motor<double> &target) const;

        gafro::Wrench<double> computeForce(const Agent::State &state,
                                   const double &agent_radius,
                                   const gafro::Motor<double> &target) const override;

        double getMaxForce() const override
        {
          // return config_.k_gain * config_.max_allowable_force;
          
          return 0.0;
        }

      protected:
      private:
        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            double k_gain;

            double k_damping_linear;
            double k_damping_angular;

            double k_stiffness_linear;
            double k_stiffness_angular;
        } config_;

        gafro::Inertia<double> stiffness_;

        gafro::Inertia<double> damping_;
    };

}  // namespace ga_circular_fields_planner