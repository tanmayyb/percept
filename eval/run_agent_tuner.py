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
import pandas as pd
from skopt import Optimizer
from skopt.space import Real

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from run_plan import setup_logging, read_plan, run_plan
from process_results import MetricsReader

class AgentTuner:
	# load args from user
	# use agent.cfg to get params for which agent to optimize
	# use plan.py to generate new plans for randomization etc.
	# use process_results.py to get costs to optimize
	# use skopt to optimize the agent
	# save new params and logs
	def __init__(self, args):
		"""
		Initialize the agent tuner
		"""
		self.args = args
		self._plan_dir = Path(os.path.abspath(args.plan_dir))
		self._logger = logging.getLogger('agent_tuner')
		self._logger.setLevel(logging.INFO)
		self._logger.info(f"Agent tuner initialized with plan directory: {self._plan_dir}")

	def setup(self):
		"""
		Initialize the agent tuner
		"""
		self._get_configs()
		self._tuning_plan = self._get_plan_module()()
		self._results_dir = Path('eval/results') / self._plan_config['name']
		self._setup_optimizer()

	def _get_configs(self):
		"""
		Load the agents.cfg file and extract the configs
		"""
		config_filepath = str(self._plan_dir / 'agents.cfg')
		if not os.path.exists(config_filepath):
			self._logger.error(f"Could not find agents.cfg file in '{self._plan_dir}'")
			raise FileNotFoundError(f"Could not find agents.cfg file in '{self._plan_dir}'")
		try:
			with open(config_filepath, 'r') as f:
				configs = yaml.safe_load(f)
		except Exception as e:
			self._logger.error(f"Error loading agents.cfg file in '{self._plan_dir}': {str(e)}")
			raise e
		# extract configs
		self._agents_states = configs['agents']
		self._plan_config = configs['plan']
		self._evaluation_config = configs['evaluation']
		self._tuning_config = configs['tuning']
	
	
	def _get_plan_module(self):
		"""
		Check if the plan module has a generate_new_plan function
		"""
		import importlib.util
		import sys
		module_name = 'plan'
		class_name = 'TuningPlan'
		module_path = str(self._plan_dir / 'plan.py')
		spec = importlib.util.spec_from_file_location(module_name, module_path)
		if spec is None:
			raise ValueError(f"Could not find plan.py file in '{self._plan_dir}'")
		module = importlib.util.module_from_spec(spec)
		sys.modules[module_name] = module
		spec.loader.exec_module(module)
		if not hasattr(module, class_name):
			raise ValueError(f"Could not find '{class_name}' class in '{module_path}'")
		return getattr(module, class_name)

	def _setup_optimizer(self):
		"""
		Setup the optimizer
		"""
		agent = self._agents_states[0]
		agent_params = agent['params']
		self._agent_heuristic = agent['heuristic']
		self._base_agent_config = self._tuning_plan.get_base_agent_config()

		self._base_agent_config.force_list += [self._agent_heuristic]
		self._base_agent_config.force_configs[self._agent_heuristic] = {}
		self._base_agent_config.force_configs[self._agent_heuristic]['k_gain'] = agent_params['k_gain']['init']
		self._base_agent_config.force_configs[self._agent_heuristic]['k_force'] = agent_params['k_force']['init']
		self._base_agent_config.force_configs[self._agent_heuristic]['detect_shell_radius'] = agent_params['detect_shell_radius']['init']
		self._base_agent_config.force_configs[self._agent_heuristic]['max_allowable_force'] = agent_params['max_allowable_force']['init']

	
		self._search_space = [Real(agent_params['k_force']['min_val'], agent_params['k_force']['max_val'], prior="log-uniform", name="k_force")]
		self._optimizer = Optimizer(dimensions=self._search_space, base_estimator="gp", acq_func="EI", random_state=42)
		self._evaluations = []


	def run(self):
		"""
		Run the agent tuner
		"""
		num_optimization_steps = self._tuning_config['num_optimization_steps']
		k_force = self._optimizer.ask()
		for i in range(num_optimization_steps):
			cost = self._test_agent(k_force[0])
			k_force = self._optimize_agent(k_force, cost)
			self._logger.info(f"Sleeping for {self.args.wait} seconds before next optimization step")
			time.sleep(self.args.wait)

	def _test_agent(self, k_force: float):
		"""
		Test the agent
		"""
		agent_config = deepcopy(self._base_agent_config)
		agent_config.force_configs[self._agent_heuristic]['k_force'] = k_force

		self._tuning_plan.set_agent_config(agent_config)
		labels = self._tuning_plan.generate_workloads(runtime_limit=self._evaluation_config['runtime_limit'])
		
		self._logger.info(f"Testing agent with k_force: {k_force}")
		exit_code, ret = run_plan(self.args, self._plan_dir)
		if exit_code != 0:
			self._logger.error(f"Failed to run plan: {ret}")
			raise Exception(f"Failed to run plan: {ret}")
		
		filepaths = sorted([f for f in self._results_dir.glob('*.result') if any(label in f.name for label in labels)])
            # self.pointclouds = pd.DataFrame([obs["position"] for obs in pointcloud_data], columns=["x", "y", "z"])
		# assert len(filepaths) == len(labels)

		# self._logger.info(f"Processing {len(filepaths)} results")

		results = {}
		for i, filepath in enumerate(filepaths):
			reader = MetricsReader(
				filepath,
				success_time_limit=self._evaluation_config['success_time_limit'],
				acceptable_min_distances_to_target=self._evaluation_config['acceptable_min_distances_to_target']
			)
			costs = reader.agent_costs.groupby("agent_id")["cost"].mean()
			label = reader.info['label']
			results[label] = costs
		costs_df = pd.DataFrame(results).T
		costs_list = list(costs_df.mean().to_dict().values())
		return costs_list[0]

	def _optimize_agent(self, k_force: float, cost: float):
		"""
		Optimize the agent
		"""
		self._evaluations.append((k_force, cost))
		self._optimizer.tell(k_force, cost)
		return self._optimizer.ask()

def main():
		"""
		Main function to run the agent tuner
		"""
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
			# tuner.save_results()
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