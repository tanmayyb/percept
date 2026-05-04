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
#include "ThreadPool.h"

#ifdef ITT_ENABLED
#include "ittnotify.h"
#endif

namespace ga_circular_fields_planner
{

	CircularFieldPlanner::CircularFieldPlanner(const sackmesser::Interface::Ptr &interface, const std::string &name) : 
		interface_(interface),
		pool_(std::thread::hardware_concurrency())  // Initialize thread pool as member
	{
		config_ = interface->getConfigurations()->load<Configuration>(name + "/");

		for (int k = 0; k < config_.n_agents; ++k)
		{
			if (config_.agent_type == "manipulator")
			{
				agents_.push_back(std::make_shared<ManipulatorAgent>(interface, name + "/agent_" + std::to_string(k + 1)));
			}
			else if (config_.agent_type == "pointmass")
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
	}

	CircularFieldPlanner::~CircularFieldPlanner()
	{
		for (std::thread &thread : planning_threads_)
		{
			thread.join();
		}
	}

	void CircularFieldPlanner::startPlanning(const std::shared_ptr<Agent::State> &state, const gafro::Motor<double> &target)
	{
		setCurrentState(state);

#ifdef ITT_ENABLED
		__itt_domain* domain = __itt_domain_create("PlanningDomain");
#else
		void* domain = nullptr;
#endif

		// Create a single planning thread that coordinates all agents
		planning_threads_.push_back(std::thread([this, target, domain]() {
			// Planning cycle counter for determinism
			unsigned int planning_cycle_count = 0; // NOTE: may encounter overflow

			// Setup planning loop timing
			using clock = std::chrono::steady_clock;
			const double loop_freq = config_.planning_frequency;
			const auto loop_period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / loop_freq));

			auto next_time = clock::now(); // Move next_time inside the lambda
			
			while (interface_->ok())
			{
				planning_cycle_count++;

#ifdef ITT_ENABLED 
				// trace planning cycle start
				std::string label_str = ("PlanningCycleExecution" + std::to_string(planning_cycle_count));
				__itt_task_begin(domain, __itt_null, __itt_null, __itt_string_handle_create(label_str.c_str()));
#endif

				// Initialize agent states for this planning cycle
				std::vector<std::shared_ptr<Agent::State>> agent_states;
				std::shared_ptr<Agent::State> current = getCurrentState();
				
				// Initialize trajectories for all agents
				for (size_t i = 0; i < agents_.size(); i++) {
					agent_states.push_back(current->copy());
					agents_[i]->clearTrajectory();
					agents_[i]->addToTrajectory(agent_states[i]->getPose());
					agents_[i]->clearPlanningTime();
				}


				// initialize costs
				std::vector<std::vector<double>> costs(agents_.size(), std::vector<double>(costs_.size(), 0.0));

				// Plan in lockstep, ensuring all agents complete each prediction step together
				for (unsigned step = 0; step < static_cast<unsigned>(config_.max_prediction_steps); ++step) {

#ifdef ITT_ENABLED 
					// trace planning step
					std::string label_str = ("PlanningStep" + std::to_string(step));
					__itt_task_begin(domain, __itt_null, __itt_null, __itt_string_handle_create(label_str.c_str()));
#endif
					std::vector<std::future<void>> futures;
					for (size_t i = 0; i < agents_.size(); i++) {
						futures.emplace_back(

							pool_.enqueue([this, i, &agent_states, &target, &costs]() {

								auto start_time = clock::now();
#ifdef ITT_ENABLED 			
								// trace agent request time
								std::string label_str = ("Agent" + std::to_string(i) + "Request");
								__itt_task_begin(domain, __itt_null, __itt_null, __itt_string_handle_create(label_str.c_str()));
#endif
								// Plan step for each agent
								agent_states[i] = agents_[i]->planStep(agent_states[i], target, config_.delta_t)->copy();

#ifdef ITT_ENABLED 
								// trace agent request time
								__itt_task_end(domain);
#endif
								
#ifdef ITT_ENABLED 
								// trace agent cost calculation time
								label_str = ("Agent" + std::to_string(i) + "Cost");
								__itt_task_begin(domain, __itt_null, __itt_null, __itt_string_handle_create(label_str.c_str()));
#endif

								for (size_t j = 0; j < costs_.size(); j++){
									// accumulate cost for each agent for each step
									// i is agent index, j is cost index
									costs[i][j] += costs_[j]->computeCost(agents_[i], target);
								}

#ifdef ITT_ENABLED 
								// trace agent cost time
								__itt_task_end(domain);
#endif

								double duration = std::chrono::duration<double, std::milli>(clock::now() - start_time).count();
								agents_[i]->accumulatePlanningTime(duration);
							})
						);
					}
#ifdef ITT_ENABLED 
					// trace sync time
					__itt_task_begin(domain, __itt_null, __itt_null, __itt_string_handle_create("PlanningStepSync"));
#endif
					// Wait for all agent threads to complete
					for (auto &future : futures) {
						future.get();
					}
#ifdef ITT_ENABLED
					// trace sync time
					__itt_task_end(domain);
					// trace planning step end
					__itt_task_end(domain);
#endif
				}
				
				// Evaluate agents
				{
					best_agent_cost_ = std::numeric_limits<double>::max();
					std::unique_lock<std::mutex> lock(planning_mutex_);
					for (size_t i = 0; i < agents_.size(); i++) {

						// publish topics
						interface_->getCallbacks()->invoke(agents_[i]->getName() + "/planning_time", agents_[i]->getPlanningTime());
				
						double this_agent_cost = 0.0;
						
						for (size_t j = 0; j < costs_.size(); j++) 
						{
							double cost_value = costs[i][j];
							std::string cost_name = costs_[j]->getName();

							// normalization
							if (cost_name == "trajectory_smoothness_cost" || cost_name == "goal_distance_cost")
							{
								cost_value = costs_[j]->getWeight() * std::log(1 + cost_value);
							}
							
#ifdef PUB_DIFF_COSTS
							interface_->getCallbacks()->invoke(agents_[i]->getName() + "/" + cost_name, cost_value);
#endif
							this_agent_cost += cost_value;
						}

						if (this_agent_cost < config_.agent_switch_factor * best_agent_cost_){
							best_agent_ = agents_[i];
							best_agent_cost_ = this_agent_cost;
						}

						// publish costs for each agent
						interface_->getCallbacks()->invoke(agents_[i]->getName() + "/cost", this_agent_cost);

					}
					// Publish best agent name
					interface_->getCallbacks()->invoke("best_agent_name", best_agent_->getName());
				}
				
				next_time += loop_period;
				const auto now = clock::now();

				// Check to sleep or not
				if (next_time > now){
					std::this_thread::sleep_until(next_time);
				}
				else{
					// behind schedule, catch up to avoid accumulating delay
					next_time = now; 
				}
				
#ifdef ITT_ENABLED
				// trace planning cycle end
				__itt_task_end(domain);
#endif
			}
		}));
	}

	std::shared_ptr<Agent::State> CircularFieldPlanner::getStateUpdate(const std::shared_ptr<Agent::State> &state, const gafro::Motor<double> &target)
	{
		std::unique_lock<std::mutex> lock(planning_mutex_);

		if (!best_agent_)
		{
			return state;
		}

		gafro::Wrench<double> force = best_agent_->computeForce(state, target);

		// setCurrentState(best_agent_->updateState(state, force, config_.delta_t));
		setCurrentState(best_agent_->updateState(state, force, 0.010));

		return current_state_;
	}

	// double CircularFieldPlanner::computeCost(const std::shared_ptr<Agent> &agent, const gafro::Motor<double> &target) const
	// {
	//     double total_cost = 0.0;

	//     for (const auto &cost : costs_)
	//     {
	//         total_cost += cost->computeCost(agent, target);
	//     }

	//     return total_cost;
	// }

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