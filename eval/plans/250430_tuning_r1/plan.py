'''
Workload:
-metadata
-percept_config
--scene_config
--rviz_config
--fields_config
-planner_config
--poses
--agents
'''

import argparse
import yaml
from pathlib import Path
import os
from copy import deepcopy
from yaml.representer import Representer
from collections import OrderedDict
import numpy as np
import datetime
from typing import Dict
import random
from mp_eval.classes.workload import *
import logging



base_args = {
	# workload config
	"plan_name": "250430_tuning_r1",
	"delimiter": "-",

	# misc
	"show_rviz": False,
	"publish_trajectory": False,
	
	# planner config
	"path_length_cost_weight": 1.0,
	"goal_distance_cost_weight": 1.0,
	"obstacle_distance_cost_weight": 1.0,

	# fields config
	"agent_radius": 0.020,
	"mass_radius": 0.020,
	"potential_detect_shell_rad": 1.0,
	
	# agent config
	"max_velocity": 0.05,
	"approach_distance": 0.01,

	# pose config
	"poses": {
		"start_pos": [0.0, -2.0, 0.0],
		"goal_pos": [0.0, 2.0, 0.0],
		"start_orientation": [1.0, 0.0, 0.0, 0.0],
		"goal_orientation": [1.0, 0.0, 0.0, 0.0]
	},

}



def get_base_workload():
	global base_args
	return WorkloadConfig(
		metadata=Metadata(
			created_at=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
			plan_name=base_args["plan_name"],
			label="",
			tags=["None"],
			info="None",
			run_planner_profiler=False
		),
		percept_config=PerceptConfig(
			namespace="oriented_pointmass",
			mode="static",
			scene_config=SceneConfig(
				scene_type="generated",
				scene_params=yaml.safe_load(
"""
scene_generation_type: narrow_passage
scene_generation_params:
 wall1:
  spacing: 0.01
  plane: xz
  wall_width: 2.0
  wall_height: 2.0
  wall_center_i: 0.0
  wall_center_j: 0.0
  wall_offset_k: 0.0
  include_hole: true
  hole_center_x: 0.5
  hole_center_y: 0.0
  hole_width: 0.50
  hole_height: 0.50
""")
			),
			fields_config=FieldsConfig(
				agent_radius=base_args["agent_radius"],
				mass_radius=base_args["mass_radius"],
				potential_detect_shell_rad=base_args["potential_detect_shell_rad"],
				show_processing_delay=False,
				show_requests=False,
				use_cpu=False
			),
			rviz_config=RvizConfig(
				show_rviz=base_args["show_rviz"],
				rviz_config_path="rviz2/planning.rviz"
			)
		),
		planner_config=PlannerConfig(
			experiment_type="oriented_pointmass",
			loop_frequency=1000,
			service_timeout=10000,
			delta_t=0.01,
			max_prediction_steps=100,
			planning_frequency=30,
			agent_switch_factor=1.0,
			path_length_cost_weight=base_args["path_length_cost_weight"],
			goal_distance_cost_weight=base_args["goal_distance_cost_weight"],
			obstacle_distance_cost_weight=base_args["obstacle_distance_cost_weight"],
			publish_trajectory=base_args["publish_trajectory"],
			poses=base_args["poses"],
			agents=[]
		),
	)

def get_base_agent_config():
	global base_args
	return AgentConfig(
		mass=base_args["mass_radius"],
		radius=base_args["agent_radius"],
		max_velocity=base_args["max_velocity"],
		approach_distance=base_args["approach_distance"],
		force_list=['attractor_force'],
		force_configs={
			'attractor_force': {
				'k_gain': 1.0,
				'k_stiffness_linear': 1.0,
				'k_stiffness_angular': 0.01,
				'k_damping_linear': 2.5,
				'k_damping_angular': 0.025
			}
		}
	)



# Add custom representer to maintain dictionary order
def ordered_dict_representer(dumper, data):
	return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, ordered_dict_representer)


class TuningPlan:
	def __init__(self):
		self._workload_counter = 1
		self._base_workload = get_base_workload()
		self._base_agent_config = get_base_agent_config()
		self._workload_queue = []
		self._logger = logging.getLogger('tuning_plan')
		self._logger.setLevel(logging.INFO)
		self._logger.info(f"Tuning plan initialized")

	def set_workload_counter(self, workload_counter:int):
		self._workload_counter = workload_counter

	def get_base_workload(self):
		return deepcopy(self._base_workload)
	
	def get_base_agent_config(self):
		return deepcopy(self._base_agent_config)

	def set_agent_config(self, agent_config:dict):
		workload = deepcopy(self._base_workload)
		workload.planner_config.agents = [agent_config]
		self._workload_queue.append(workload)

	def generate_workloads(self, runtime_limit:float=60.0):
		global base_args
		plan = OrderedDict({'plan_name': base_args["plan_name"], 'workloads': [], 'runtime': []})
		labels = []


		plan_directory = Path(os.path.dirname(__file__))
		plan_index_file = plan_directory / "run.plan"
		workloads_directory = plan_directory / "workloads"
		os.makedirs(workloads_directory, exist_ok=True)	

		delimiter = base_args["delimiter"]
		for workload in self._workload_queue:
			# set label and filename
			heuristic_name = workload.planner_config.agents[0].force_list[1]
			label = f"{self._workload_counter}{delimiter}{heuristic_name}"
			workload.metadata.label = label
			filename = label + ".yaml"
			labels.append(label)

			# write workload to file and add to plan
			workload_path = workloads_directory / filename
			with open(workload_path, "w") as f:
				f.write(workload.to_yaml())
				self._workload_counter += 1
				plan['workloads'].append(str(workload_path))
				plan['runtime'].append(runtime_limit)

		# write plan to file
		with open(plan_index_file, "w") as f:
			yaml.dump(plan, f)
		
		self._workload_queue = []
		return labels