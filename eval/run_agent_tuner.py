import argparse
import yaml
import os
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import time
import subprocess

import numpy as np
from skopt import Optimizer
from skopt.space import Real
import time

from copy import deepcopy
from process_results import MetricsReader


from run_plan import setup_logging, read_plan, run_plan
from dataclasses import dataclass

from process_results import MetricsReader
from typing import Dict, List, Tuple

@dataclass
class plan_args:
  plan_dir: str
  enable_metrics: bool = False
  wait: float = 10.0
  duration: float = 0.0


class AgentTuner:
  def __init__(self, args):
    self.args = args    
    self.plan_dir = Path(os.path.abspath(args.plan_dir))
    self.logger = logging.getLogger('agent_tuner')
    self.logger.setLevel(logging.INFO)
    self.logger.info(f"Agent tuner initialized with plan directory: {self.plan_dir}")

    self.configs = {}
    self.agents_states = {}
    self.current_agent_to_optimize = None
    self.workload_id = 1
    self.num_iterations = None

  def setup(self):
    self._get_configs()
    self.agents_states = self.configs['agents']
    self.plan_config = self.configs['plan']
    self.num_iterations = self.plan_config['num_iterations']
    self.num_optimization_steps = self.plan_config['num_optimization_steps']
    self.num_pose_randomizations = self.plan_config['num_pose_randomizations']
    self._generate_new_plan_func = self._get_plan_module()

  def _get_configs(self):
    # load all agents from config file
    config_filepath = str(self.plan_dir / 'agents.cfg')
    if not os.path.exists(config_filepath):
      self.logger.error(f"Could not find agents.cfg file in {self.plan_dir}")
      raise FileNotFoundError(f"Could not find agents.cfg file in {self.plan_dir}")
    with open(config_filepath, 'r') as f:
      self.configs = yaml.safe_load(f)

  def _get_plan_module(self):
    """
    Check if the plan module has a generate_new_plan function
    """
    import importlib.util
    import sys
    module_name = 'plan'
    module_path = str(self.plan_dir / 'plan.py')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
      raise ValueError(f"Could not find plan.py file in {self.plan_dir}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    func_name = 'generate_new_plan'
    if not hasattr(module, func_name):
      raise ValueError(f"Could not find {func_name} function in {module_path}")
    
    return getattr(module, func_name)

  def _evaluate_this_agent(self, agent_name:str, new_k_force:float):
    """
    Evaluate the cost of the new agent configuration
    """
    agent_config = deepcopy(self.agents_states[agent_name])
    agent_config['k_force'] = new_k_force
    self.agents_states[agent_name] = agent_config
    return self._get_cost(agent_config)
  
  def _evaluate_all_agents(self):
    """
    Evaluate the cost of all agents
    """
    agents_configs = deepcopy(self.agents_states)
    return self._get_cost(agents_configs)

  def _get_cost(self, agents_config:dict)->Dict[str, Tuple[float]]:
    """
    Get the cost of the agent configuration
    """
    # current agent config
    self._create_plan(agents_config)

    # run plan
    exit_code, ret = run_plan(self.args, self.plan_dir)
    if exit_code != 0:
      self.logger.error(f"Plan failed to run with exit code {exit_code}")
      raise Exception(f"Plan failed to run with exit code {exit_code}")

    # read results
    # results_dir = self.plan_dir / 'results'
    cost = 0.0

    return cost

  def _create_plan(self, agents_config:dict):
    """
    Create a new plan with the given agent configuration
    """
    num_random_poses = self.num_pose_randomizations
    plan_config = {
      'agents': agents_config,
      'num_random_poses': num_random_poses
    }
    workload_id = self._generate_new_plan_func(self.workload_id, plan_config)
    self.workload_id = workload_id # update workload id


  def _optimize_worst_agent(self):
    """
    Optimize the worst agent
    """
    agent_to_optimize = self.current_agent_to_optimize

    # Number of evaluations
    n_calls = self.num_optimization_steps
    agent_config = self.agents_states[agent_to_optimize]
    min_param_val = agent_config['min_param_val']
    max_param_val = agent_config['max_param_val']
    
    search_space = [Real(min_param_val, max_param_val, prior="log-uniform", name="param")]
    self.optimizer = Optimizer(dimensions=search_space, base_estimator="gp", acq_func="EI", random_state=42)
    evaluations = []

    for i in range(n_calls):
      # Suggest a new parameter to evaluate
      next_x = self.optimizer.ask()[0]
      
      # run the workloads with new parameter
      cost = self._evaluate_this_agent(agent_to_optimize, next_x)    

      # Store the result
      evaluations.append((next_x, cost))
      
      # Tell the optimizer the result
      self.optimizer.tell([next_x], [cost])

      time.sleep(1)

    # Print the best found parameter
    best_index = np.argmin([e[1] for e in evaluations])
    best_param, best_cost = evaluations[best_index]

  def _test_for_worst_agent(self):
    """
    Test for the worst agent
    """
    self._evaluate_all_agents()

  def run(self):
    """
    Run the agent tuner
    """
    for i in range(self.num_iterations):
      self._test_for_worst_agent()
      # self._optimize_worst_agent()


def main():
    parser = argparse.ArgumentParser(description='Run a plan of workloads')
    parser.add_argument('plan_dir', type=str, help='Directory containing the run.plan file')
    parser.add_argument('--enable_metrics', action='store_true', help='Enable metrics collection')
    parser.add_argument('--wait', type=float, default=10.0, help='Wait time between workloads in seconds')
    parser.add_argument('--duration', type=float, default=0.0, help='Duration for each workload in seconds')
    args = parser.parse_args()

    plan_dir = os.path.abspath(args.plan_dir)
    
    log_file = setup_logging(plan_dir)
    start_time = time.time()  # Start timing
    logging.info(f"Starting agent tuner from directory: {plan_dir}")
    logging.info(f"Log file: {log_file}")
    
    try:
      tuner = AgentTuner(args)
      tuner.setup()
      tuner.run()
      total_time = time.time() - start_time  # Calculate total time
      hours = int(total_time // 3600)
      minutes = int((total_time % 3600) // 60)
      seconds = total_time % 60
      logging.info(f"Total execution time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
      return 0
    except Exception as e:
      logging.error(f"Error tuning agents: {str(e)}", exc_info=True)
      return 1


if __name__ == "__main__":
  exit(main())