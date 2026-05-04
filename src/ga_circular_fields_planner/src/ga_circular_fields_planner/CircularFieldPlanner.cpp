#include <cassert>
#include <cmath>
#include <ga_circular_fields_planner/CircularFieldPlanner.hpp>
#include <ga_circular_fields_planner/ManipulatorAgent.hpp>
#include <ga_circular_fields_planner/PointmassAgent.hpp>
#include <iostream>
#include <limits>
#include <mutex>
#include <sackmesser/Callbacks.hpp>
#include <sackmesser/Configurations.hpp>
#include <chrono>

namespace ga_circular_fields_planner
{

	CircularFieldPlanner::CircularFieldPlanner(const sackmesser::Interface::Ptr &interface, const std::string &name) : 
		interface_(interface)
	{
		config_ = interface->getConfigurations()->load<Configuration>(name + "/");

		for (int k = 0; k < config_.n_agents; ++k)
		{
			if (config_.agent_type == "manipulator")
			{
				agents_.push_back(std::make_shared<ManipulatorAgent>(interface, name + "/agent_" + std::to_string(k + 1)));
			}
			else if (config_.agent_type == "oriented_pointmass")
			{
				agents_.push_back(std::make_shared<PointmassAgent>(interface, name + "/agent_" + std::to_string(k + 1)));
			}
			else
			{
				interface->log()->fatal() << "CircularFieldPlanner: unknown agent type " << config_.agent_type << std::endl;
			}
		}

		for (const std::string &cost : config_.costs)
		{
			costs_.push_back(Cost::getFactory()->createShared(cost, interface, name));
		}

		best_agent_cost_ = std::numeric_limits<double>::max();

    savePlannerInfo();	
  }

	CircularFieldPlanner::~CircularFieldPlanner()
	{
    stopPlanning();
	}

  void CircularFieldPlanner::savePlannerInfo()
  {
    FILE *fptr = fopen("/tmp/planner_info.txt", "w");
    
    if(fptr) fprintf(fptr, "%zu %zu\n", agents_.size(), costs_.size());
    
    if(fptr) fflush(fptr);

    if(fptr) fclose(fptr);

    fptr = nullptr;
  }

  void CircularFieldPlanner::run()
  {
    const auto loop_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / config_.planning_frequency)
    );

    size_t num_agents = config_.n_agents;

    size_t num_costs = costs_.size();

    int max_steps = config_.max_prediction_steps;

    // PRE-ALLOCATION
    std::vector<std::shared_ptr<Agent::State>> agent_states(num_agents);

    std::vector<double> final_costs(num_agents, 0.0);

    std::vector<double> cost_matrix(num_agents * num_costs, 0.0);

    std::vector<double> a_max(num_agents, 0.0);

    // LOGGING SETUP
    FILE *timing_log = nullptr;
  
    FILE *cost_log = nullptr;

    FILE *best_agent_log = nullptr;


    LOG_OPEN_TIMING_PERF(timing_log, "/tmp/timing_log.txt");

    LOG_OPEN_AGENT_COSTS(cost_log, "/tmp/cost_log.txt");

    LOG_OPEN_BEST_AGENT(best_agent_log, "/tmp/best_agent_log.txt");

    CAPTURE_TIME(t_last_iter);

    CAPTURE_TIME(t_planning_start)

    double d_global_point = 0.0, d_iter = 0.0, d_data = 0.0, d_copy = 0.0, d_sim = 0.0, d_agg = 0.0, d_sel = 0.0;

    // PLANNING CYCLES
    while(running_.load())
    {
      
      CAPTURE_TIME(t_start);

      CALCULATE_DURATION(d_iter, t_start, t_last_iter);

      auto current_state = getCurrentState();

      auto current_target = getCurrentTarget(); 

      CAPTURE_TIME(t_data);
      
      for (size_t i = 0; i < num_agents; ++i)
      {        
        agent_states[i] = current_state->copy();

        agents_[i]->clearTrajectory();

        agents_[i]->addToTrajectory(agent_states[i]->getPose());

        agents_[i]->clearPlanningTime();

        std::fill(
          cost_matrix.begin() + (i * num_costs), cost_matrix.begin() + ((i + 1) * num_costs), 0.0
        );

        a_max[i] = agents_[i]->getMaxAcceleration();

      } // setup data structs for the planning cycle

      double s_min = gafro::Motor<double>(
        current_state->getPose().reverse() * current_target
      ).log().vector().norm(); // euclidean distance to target, aka shortest path/s_min

      CAPTURE_TIME(t_copy);

      CALCULATE_DURATION(d_global_point, t_planning_start, t_copy);
      
      // PLANNING TRAJECTORY ROLLOUT
      #pragma omp parallel for schedule(dynamic)
      for(size_t i = 0; i < num_agents; ++i) 
      {
        auto& agent = agents_[i];

        auto& state = agent_states[i];

        double sum_force = 0.0, sum_state = 0.0, sum_traj = 0.0, sum_cost = 0.0, sum_total = 0.0;

        for (int step = 0; step < max_steps; ++step) 
        {

          CAPTURE_TIME(t0);
          
          gafro::Wrench<double> force = agent->computeForce(state, current_target);

          CAPTURE_TIME(t1);
          
          agent->updateStateInPlace(state, force, config_.delta_t);

          CAPTURE_TIME(t2);

          agent->updateTrajectory(state->getPose());

          CAPTURE_TIME(t3);

          for (size_t j = 0; j < num_costs; ++j)
          {
            cost_matrix[i * num_costs + j] += costs_[j]->computeCost(agent, current_target);
          }
          
          // TIMING CAPTURE
          CAPTURE_TIME(t4);

          ACCUMULATE_DURATION(sum_force,  t0, t1);

          ACCUMULATE_DURATION(sum_state,  t1, t2);

          ACCUMULATE_DURATION(sum_traj,   t2, t3);

          ACCUMULATE_DURATION(sum_cost,   t3, t4);

          ACCUMULATE_DURATION(sum_total,  t0, t4);

        }

        LOG_INFO_TIMING_PERF(
          timing_log, "%.6f %zu %.2f %.2f %.2f %.2f %.2f\n", 
          d_global_point/1000.0, i, sum_force, sum_state, sum_traj, sum_cost, sum_total
        );
      }

      CAPTURE_TIME(t_sim);

      CALCULATE_DURATION(d_global_point, t_planning_start, t_sim);

      // #pragma omp parallel for schedule (static)
      for (size_t i = 0; i < num_agents; ++i)
      {
        LOG_INFO_AGENT_COSTS(
          cost_log, " %.6f %zu ", 
          d_global_point/1000.0, i
        );
        
        double total_c = 0.0;

        for (size_t j = 0; j < num_costs; ++j)
        {
          const std::string& name = costs_[j]->getName();

          double raw_cost = cost_matrix[i * num_costs + j];

          double weighted_cost = 0.0;

          if(name == "path_length_cost")
          {
            if (s_min * raw_cost < 1.0e-4)  raw_cost = 0.0;

            else if (raw_cost >= s_min) raw_cost = 1 - (s_min / raw_cost);

            else raw_cost = 1 - (raw_cost / s_min);
          }
          else if (name == "trajectory_smoothness_cost")
          {
            raw_cost /= (max_steps * std::pow(config_.delta_t, 6)) ;

            double j_max = std::pow( a_max[i] / config_.delta_t, 2);

            raw_cost = std::min(1.0, raw_cost / j_max);
          }
          else
          {
            raw_cost /= max_steps;
          }

          weighted_cost = costs_[j]->getWeight() * raw_cost;

          total_c += weighted_cost;

          LOG_INFO_AGENT_COSTS(cost_log, "%.6f %.6f ", raw_cost, weighted_cost);
        }      

        final_costs[i] = total_c;

        LOG_INFO_AGENT_COSTS(cost_log, "%.6f\n", total_c);

      } // accumulate total cost

      CAPTURE_TIME(t_agg);

      double local_best_cost = std::numeric_limits<double>::max();

      auto local_best_agent = agents_[0];

      int best_agent_idx = 0;

      for (size_t i = 0; i < num_agents; ++i)
      {
        if (final_costs[i] < config_.agent_switch_factor * local_best_cost)
        {
          local_best_agent = agents_[i];

          local_best_cost = final_costs[i];

          best_agent_idx = i;
        }
      }

      {
        std::unique_lock<std::mutex> lock(planning_mutex_);

        best_agent_ = local_best_agent;

        best_agent_cost_ = local_best_cost;
      }

      CAPTURE_TIME(t_sel);

      // CALCULATE_DURATION(d_data,  t_start,  t_data);

      // CALCULATE_DURATION(d_copy,  t_data,   t_copy);

      // CALCULATE_DURATION(d_sim,   t_copy,   t_sim);

      // CALCULATE_DURATION(d_agg,   t_sim,    t_agg);

      // CALCULATE_DURATION(d_sel,   t_agg,    t_sel);

      // if (d_data > 10.0)            LOG_INFO_TIMING_PERF(timing_log, "Data Acquisition Stall: %.2f ms\n", d_data);
      
      // if (d_copy > 10.0)            LOG_INFO_TIMING_PERF(timing_log, "State Copy Stall: %.2f ms\n",       d_copy);
      
      // if (d_sim > 1.5 * max_steps)  LOG_INFO_TIMING_PERF(timing_log, "Simulation Loop Stall: %.2f ms\n",  d_sim);
      
      // if (d_agg > 10.0)             LOG_INFO_TIMING_PERF(timing_log, "Aggregation Stall: %.2f ms\n",      d_agg);
      
      // if (d_sel > 10.0)             LOG_INFO_TIMING_PERF(timing_log, "Selection Stall: %.2f ms\n",        d_sel);
      
      // if (d_iter > 10.0)            LOG_INFO_TIMING_PERF(timing_log, "Iteration Stall: %.2f ms\n",        d_iter);

      LOG_INFO_BEST_AGENT(best_agent_log, "%.6f %zu %.6f\n", d_global_point/1000.0, best_agent_idx, best_agent_cost_);

      LOG_FLUSH_TIMING_PERF(timing_log);

      LOG_FLUSH_AGENT_COSTS(cost_log);

      LOG_FLUSH_BEST_AGENT(best_agent_log);

      CAPTURE_TIME(t_last_iter);

      if(!interface_->ok()) running_ = false;
    }

    LOG_CLOSE_TIMING_PERF(timing_log);

    LOG_CLOSE_AGENT_COSTS(cost_log);

    LOG_CLOSE_BEST_AGENT(best_agent_log);
  }

  void CircularFieldPlanner::startPlanning(const std::shared_ptr<Agent::State> &state, const gafro::Motor<double> &target)
  {
    setCurrentState(state);

    setCurrentTarget(target);
    
    running_ = true;

    thread_ = std::thread(&CircularFieldPlanner::run, this);
  }

  void CircularFieldPlanner::stopPlanning()
  {
    running_ = false;

    if (thread_.joinable())
    {
      thread_.join();
    }
  }
  

  std::shared_ptr<Agent::State> CircularFieldPlanner::getStateUpdate(const std::shared_ptr<Agent::State> &state)
	{
    auto target = getCurrentTarget();

    std::unique_lock<std::mutex> lock(planning_mutex_);

		if (!best_agent_)
		{
			return state;
		}

		gafro::Wrench<double> force = best_agent_->computeForce(state, target);

		setCurrentState(best_agent_->updateState(state, force, config_.delta_t));

		return current_state_;
	}


	void CircularFieldPlanner::setCurrentState(const std::shared_ptr<Agent::State> &state)
	{
		std::unique_lock<std::mutex> lock(state_mutex_);

		current_state_ = state->copy();
	}

	std::shared_ptr<Agent::State> CircularFieldPlanner::getCurrentState()
	{
		std::unique_lock<std::mutex> lock(state_mutex_);

		return current_state_->copy();
	}


	void CircularFieldPlanner::setCurrentTarget(const gafro::Motor<double> &target)
	{
		std::unique_lock<std::mutex> lock(target_mutex_);

		current_target_ = target;
	}


	gafro::Motor<double> CircularFieldPlanner::getCurrentTarget()
	{
		std::unique_lock<std::mutex> lock(target_mutex_);

		return current_target_;
	}


	bool CircularFieldPlanner::Configuration::load(const std::string &ns, const std::shared_ptr<sackmesser::Configurations> &server)
	{
		return server->loadParameter(ns + "n_agents", &n_agents, false) &&                         //
			   server->loadParameter(ns + "agent_type", &agent_type) &&                            //
			   server->loadParameter(ns + "delta_t", &delta_t, true) &&                            //
			   server->loadParameter(ns + "max_prediction_steps", &max_prediction_steps, true) &&  //
			   server->loadParameter(ns + "planning_frequency", &planning_frequency, true) &&      //
			   server->loadParameter(ns + "agent_switch_factor", &agent_switch_factor, true) &&    //
			   server->loadParameter(ns + "costs", &costs);                                        //
	}

}  // namespace ga_circular_fields_planner
