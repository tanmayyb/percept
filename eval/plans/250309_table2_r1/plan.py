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

# globals
PLAN_NAME = "table2_r1"
EXPERIMENT_TYPE = "oriented_pointmass"
RVIZ_CONFIG_PATH = "rviz2/planning.rviz"
SHOW_RVIZ = False
SHOW_PROCESSING_DELAY = False
SHOW_REQUESTS = False
DELIMETER = "-"

PLANNER_LOOP_FREQUENCY = 1000
DELTA_T = 0.01
PLANNING_FREQUENCY=50

AGENT_SWITCH_FACTOR = 1.0
PATH_LENGTH_COST_WEIGHT = 1.0
GOAL_DISTANCE_COST_WEIGHT = 1.0
OBSTACLE_DISTANCE_COST_WEIGHT = 1.0

AGENT_MASS = 1.0
AGENT_RADIUS = 0.020
MASS_RADIUS = 0.020
MAX_VELOCITY = 0.1
APPROACH_DISTANCE = 0.25

FORCE_LIST = ["attractor_force", "velocity_heuristic_force"]
FORCE_CONFIGS = {
	"attractor_force": {
		"k_gain": 1.0,
		"k_stiffness_linear": 1.0,
		"k_stiffness_angular": 0.01,
		"k_damping_linear": 2.5,
		"k_damping_angular": 0.025
	},
	"velocity_heuristic_force": {
		"k_gain": 1.0,
		# "k_force": 5.0e-2,

		"k_force": 2.5e-3,

		"detect_shell_radius": 20.0,
		"max_allowable_force": 40.0
	}
}

AGENTS_CONFIGS = [
	AgentConfig(
		mass=AGENT_MASS,
		radius=AGENT_RADIUS,
		max_velocity=MAX_VELOCITY,
		approach_distance=APPROACH_DISTANCE,
		force_list=FORCE_LIST,
		force_configs=FORCE_CONFIGS
	)
]

PARAMS_KEYS = ['label', 'tags', 'service_timeout', 'max_prediction_steps', 'scene_params', 'use_cpu', 'poses', 'agents_config']


# Add custom representer to maintain dictionary order
def ordered_dict_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, ordered_dict_representer)


def _generate_agents_config(k_force):
	force_configs = deepcopy(FORCE_CONFIGS)
	agents_config = deepcopy(AGENTS_CONFIGS)
	force_configs['velocity_heuristic_force']['k_force'] = k_force
	agents_config[0].force_configs = force_configs
	return agents_config



def _generate_scene_params():
	scenes = {}

	narrow_passage = {
		"scene_generation_type": "narrow_passage",
		"scene_generation_params": {
			"wall1": {
				"spacing": 0.040,
				"plane": "xz",
				"wall_width": 4.0,
				"wall_height": 4.0,
				"wall_center_i": 0.0,
				"wall_center_j": 0.0,
				"wall_offset_k": 0.75,
				"include_hole": True,
				"hole_center_x": -0.25,
				"hole_center_y": 0.0,
				"hole_width": 0.5,
				"hole_height": 4.0
			},
			"wall2": {
				"spacing": 0.040,
				"plane": "xz",
				"wall_width": 4.0,
				"wall_height": 4.0,
				"wall_center_i": 0.0,
				"wall_center_j": 0.0,
				"wall_offset_k": 1.50,
				"include_hole": True,
				"hole_center_x": 0.25,
				"hole_center_y": 0.0,
				"hole_width": 0.5,
				"hole_height": 4.0
			}
		}
	}

	cluttered1k = {
		"scene_generation_type": "cluttered",
		"scene_generation_params": {
			"num_obstacles": 1000,
			"seed": 42,
			"bbox": [-2.25, 0.0, -2.25, 2.25, 2.25, 2.25],
			"include_deadzones": True,
			"deadzone_bbox1": [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
			"deadzone_bbox2": [-0.5, 1.75, -0.5, 0.5, 2.75, 0.5]
		}
	}

	cluttered5k = {
		"scene_generation_type": "cluttered",
		"scene_generation_params": {
			"num_obstacles": 5000,
			"seed": 42,
			"bbox": [-2.25, 0.0, -2.25, 2.25, 2.25, 2.25],
			"include_deadzones": True,
			"deadzone_bbox1": [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
			"deadzone_bbox2": [-0.5, 1.75, -0.5, 0.5, 2.75, 0.5]
		}
	}

	cluttered10k = {
		"scene_generation_type": "cluttered",
		"scene_generation_params": {
			"num_obstacles": 10000,
			"seed": 42,
			"bbox": [-2.25, 0.0, -2.25, 2.25, 2.25, 2.25],
			"include_deadzones": True,
			"deadzone_bbox1": [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
			"deadzone_bbox2": [-0.5, 1.75, -0.5, 0.5, 2.75, 0.5]
		}
	}

	scenes[0] = narrow_passage
	scenes[1] = cluttered1k
	scenes[2] = cluttered5k
	scenes[3] = cluttered10k

	return scenes

def _generate_poses():
	random.seed(42)
	poses = []
	num_poses = 10

	start_orientation = [1.0, 0.0, 0.0, 0.0]
	goal_orientation = [1.0, 0.0, 0.0, 0.0]
	rangex = 0.5
	rangez = 0.5


	for i in range(num_poses):
		start_pos = [random.uniform(-rangex, rangex), 0.0, random.uniform(-rangez, rangez)]
		goal_pos = [random.uniform(-rangex, rangex), 2.25, random.uniform(-rangez, rangez)]

		pose = {
			"start_pos": start_pos,
			"goal_pos": goal_pos,
			"start_orientation": start_orientation,
			"goal_orientation": goal_orientation
		}

		poses.append(pose)

	return poses


def _generate_procedurally():
	all_params = OrderedDict()
	workload_id = 1
	scenes = _generate_scene_params()
	cpu_options = {0: False, 1: True}
	planning_windows = {0: 50, 1: 100, 2: 150}
	poses = _generate_poses()
	
	k_force = [2.5e-3, 5.0e-2, 5.0e-2, 5.0e-2]

	for i, scene_params in scenes.items():
		for j, use_cpu in cpu_options.items():
			for k, planning_window in planning_windows.items():
				for l, pose in enumerate(poses):
					label = f"scene_{i}{DELIMETER}use_cpu_{j}{DELIMETER}pred_steps_{k}{DELIMETER}pose_{l}"
					params = {}
					params['label'] = label
					params['tags'] = [
						PLAN_NAME,
						f"scene_{i}",
						f"use_cpu_{j}",
						f"planning_window_{k}",
						f"pose_{l}"
					]

					timeout = 10000
					runtime = 120.0
					
					params['service_timeout'] = timeout
					params['max_prediction_steps'] = planning_window
					params['scene_params'] = scene_params
					params['use_cpu'] = use_cpu
					params['poses'] = pose
					params['agents_config'] = _generate_agents_config(k_force[i]) # depends on scene
					filename = f"{workload_id}{DELIMETER}{label}.yaml"
					all_params[filename] = (params, runtime)
					workload_id += 1
	return all_params

def _generate_workload_yaml(params: Dict):
  
	missing_keys = [key for key in PARAMS_KEYS if key not in params]
	if missing_keys:
		raise ValueError(f"Missing required parameters: {', '.join(missing_keys)}")
  
	metadata = Metadata(
			created_at=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
			plan_name=PLAN_NAME,
			label=params['label'],
			tags=params['tags'],
			info="None"
	)

	
	percept_config = PerceptConfig(
		namespace=EXPERIMENT_TYPE,
		mode="static",
		scene_config=SceneConfig(
			scene_type="generated",
			scene_params=params['scene_params']
		),
		fields_config=FieldsConfig(
			agent_radius=AGENT_RADIUS,
			mass_radius=MASS_RADIUS,
			publish_force_vector=False,
			show_processing_delay=SHOW_PROCESSING_DELAY,
			show_requests=SHOW_REQUESTS,
			use_cpu=params['use_cpu']
		),
		rviz_config=RvizConfig(
			show_rviz=SHOW_RVIZ,
			rviz_config_path=RVIZ_CONFIG_PATH
		)
	)

	planner_config = PlannerConfig(
    experiment_type=EXPERIMENT_TYPE,	
    loop_frequency=PLANNER_LOOP_FREQUENCY,
    service_timeout=params['service_timeout'],
		delta_t=DELTA_T,
    max_prediction_steps=params['max_prediction_steps'],
    planning_frequency=PLANNING_FREQUENCY,
    agent_switch_factor=AGENT_SWITCH_FACTOR,
    path_length_cost_weight=PATH_LENGTH_COST_WEIGHT,
    goal_distance_cost_weight=GOAL_DISTANCE_COST_WEIGHT,
    obstacle_distance_cost_weight=OBSTACLE_DISTANCE_COST_WEIGHT,
    poses=params['poses'],
    agents=params['agents_config']
	)
	
	workload_config = WorkloadConfig(
		metadata=metadata,
		percept_config=percept_config,
		planner_config=planner_config
	)
	return workload_config.to_yaml()


def main():
	plan_directory = Path(os.path.dirname(__file__))
	plan_index_file = plan_directory / "run.plan"
	os.makedirs(plan_directory / "workloads", exist_ok=True)
	workload_params = _generate_procedurally()
	plan = OrderedDict({'plan_name': PLAN_NAME, 'workloads': [], 'runtime': []})
	
	for filename, (params, runtime) in workload_params.items():
		workload = _generate_workload_yaml(params)
		workload_path = plan_directory / "workloads" / filename
		with open(workload_path, "w") as f:
			f.write(workload)

		plan['workloads'].append(f"eval/plans/{plan_directory.name}/workloads/{filename}")
		plan['runtime'].append(runtime)
	with open(plan_index_file, "w") as f:
		yaml.dump(plan, f)

if __name__ == "__main__":
	main()
