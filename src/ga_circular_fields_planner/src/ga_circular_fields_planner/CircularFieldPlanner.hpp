/*
   Class for circular field agent manager
*/
#pragma once

#include <ga_circular_fields_planner/Preproc.hpp>
#include <ga_circular_fields_planner/Agent.hpp>
#include <ga_circular_fields_planner/Costs.hpp>
#include <sackmesser/Interface.hpp>
#include <thread>
#include <vector>
#include <Eigen/Dense>

namespace ga_circular_fields_planner
{

    class CircularFieldPlanner
    {
      public:
        CircularFieldPlanner(const sackmesser::Interface::Ptr &interface, const std::string &name);

        ~CircularFieldPlanner();

        void savePlannerInfo();

        void startPlanning(const std::shared_ptr<Agent::State> &state, const gafro::Motor<double> &target);

        void stopPlanning();

        std::shared_ptr<Agent::State> getStateUpdate(const std::shared_ptr<Agent::State> &state);

        void setCurrentState(const std::shared_ptr<Agent::State> &state);

        void setCurrentTarget(const gafro::Motor<double> &target);

        std::shared_ptr<Agent::State> getCurrentState();

        gafro::Motor<double> getCurrentTarget();

      private:

        void run();

      private:
        sackmesser::Interface::Ptr interface_;

        struct Configuration : public sackmesser::Configuration
        {
            bool load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server);

            int n_agents;

            std::string agent_type;

            double delta_t;

            int max_prediction_steps;

            int planning_frequency;

            double agent_switch_factor;

            std::vector<std::string> costs;

        } config_;


        std::thread thread_;

        std::atomic<bool> running_{false};

        std::mutex planning_mutex_;

        std::mutex state_mutex_;

        std::mutex target_mutex_;

        double best_agent_cost_;

        std::shared_ptr<Agent> best_agent_;

        std::vector<std::shared_ptr<Agent>> agents_;

        std::shared_ptr<Agent::State> current_state_;

        gafro::Motor<double> current_target_;

        std::vector<std::shared_ptr<Cost>> costs_;
    };

}  // namespace ga_circular_fields_planner
