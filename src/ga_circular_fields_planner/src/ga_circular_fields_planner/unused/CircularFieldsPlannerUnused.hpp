/*
   Class for circular field agent manager
*/
#pragma once

#include <pthread.h>

#include <Eigen/Dense>
#include <ga_circular_fields_planner/Agent.hpp>
#include <ga_circular_fields_planner/Cost.hpp>
#include <ga_circular_fields_planner/Obstacle.hpp>
#include <sackmesser/Interface.hpp>
#include <thread>
#include <vector>
#include "ThreadPool.h"

namespace ga_circular_fields_planner
{

    class CircularFieldPlanner
    {
      public:
        CircularFieldPlanner(const sackmesser::Interface::Ptr &interface, const std::string &name);

        ~CircularFieldPlanner();

        void startPlanning(const std::shared_ptr<Agent::State> &state, const gafro::Motor<double> &target);

        std::shared_ptr<Agent::State> getStateUpdate(const std::shared_ptr<Agent::State> &state, const gafro::Motor<double> &target);

      private:
        // void evaluateAgents(const gafro::Motor<double> &target);

        // double computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const;

        void setCurrentState(const std::shared_ptr<Agent::State> &state);

        std::shared_ptr<Agent::State> getCurrentState();

      private:
        sackmesser::Interface::Ptr interface_;

        ThreadPool pool_;

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

        std::shared_ptr<Agent> best_agent_;

        double best_agent_cost_;

        std::vector<std::shared_ptr<Agent>> agents_;

        std::vector<std::thread> planning_threads_;

        std::mutex planning_mutex_;

        std::mutex state_mutex_;

        std::shared_ptr<Agent::State> current_state_;

        std::vector<std::shared_ptr<Cost>> costs_;
    };

}  // namespace ga_circular_fields_planner