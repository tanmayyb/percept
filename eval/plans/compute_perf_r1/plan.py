#! /usr/bin/env python3

import yaml
import datetime
from mp_eval.classes.workload import *
from pathlib import Path
import os
from copy import deepcopy
from collections import OrderedDict
import random

def ordered_dict_representer(dumper, data):
  return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, ordered_dict_representer)

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

""" --- Manual Fine-Tuned Parameters --- """

base_args = {
  # workload config
  "plan_name": "compute_perf_r1",

  # debug 
  "show_rviz": False,
  "publish_trajectory": False,
  
  # fields config
  "agent_radius": 0.020,
  "mass_radius": 0.020,
  "potential_detect_shell_rad": 0.25,
  
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

experiment_args = {
  'delim1': '-', # delim between cases
  'delim2': '_', # delim within case

  'cpu_options': {0: False, 1: True},
  'planning_windows': {0: 50, 1: 100, 2: 200},

  'num_poses': 10,
  'runtime': 120.0,


}

scene_ids = {
  0: 'htw',
  1: 'passage',
  2: 'trap',
  3: 'cluttered'
}


scene_config = {
  'htw': {
    'scene_type': 'generated', 
    'scene_params': yaml.safe_load(
"""
publish_once: true
scene_generation_type: walls
scene_generation_params:
  dense_wall1:
    thickness: 4
    spacing: 0.040
    plane: xz
    wall_width: 2.0
    wall_height: 2.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: -1.0
    include_hole: true
    hole_center_x: -0.5
    hole_center_y: 0.0
    hole_width: 0.50
    hole_height: 0.50
  dense_wall2:
    thickness: 4
    spacing: 0.040
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
  dense_wall3:
    thickness: 4
    spacing: 0.040
    plane: xz
    wall_width: 2.0
    wall_height: 2.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: 1.0
    include_hole: true
    hole_center_x: -0.5
    hole_center_y: 0.0
    hole_width: 0.50
    hole_height: 0.50
""")},
  'passage': {
    'scene_type': 'generated', 
    'scene_params': yaml.safe_load(
"""
publish_once: true
scene_generation_type: passage
scene_generation_params:
  dense_wall1:
    thickness: 4
    spacing: 0.040
    plane: xz
    wall_width: 4.0
    wall_height: 2.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: 1.0
    include_hole: true
    hole_center_x: -1.0
    hole_center_y: 0.0
    hole_width: 0.50
    hole_height: 4.0
  dense_wall2:
    thickness: 4
    spacing: 0.040
    plane: xz
    wall_width: 4.0
    wall_height: 2.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: 0.0
    include_hole: true
    hole_center_x: 1.0
    hole_center_y: 0.0
    hole_width: 0.50
    hole_height: 4.0
  dense_wall3:
    thickness: 4
    spacing: 0.040
    plane: xz
    wall_width: 4.0
    wall_height: 2.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: -1.0
    include_hole: true
    hole_center_x: -1.0
    hole_center_y: 0.0
    hole_width: 0.50
    hole_height: 4.0
""")},
  'trap': {
    'scene_type': 'generated', 
    'scene_params': yaml.safe_load(
"""
publish_once: true
scene_generation_type: narrow_passage
scene_generation_params:
  dense_wall1:
    thickness: 4
    spacing: 0.040
    plane: xz
    wall_width: 1.0
    wall_height: 1.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: 0.5
    include_hole: false
  dense_wall2:
    thickness: 4
    spacing: 0.040
    plane: yz
    wall_width: 1.0
    wall_height: 1.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: 0.5
    include_hole: false
  dense_wall3:
    thickness: 4
    spacing: 0.040
    plane: yz
    wall_width: 1.0
    wall_height: 1.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: -0.5
    include_hole: false
  dense_wall4:
    thickness: 4
    spacing: 0.040
    plane: xy
    wall_width: 1.0
    wall_height: 1.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: -0.5
    include_hole: false
  dense_wall5:
    thickness: 4
    spacing: 0.040
    plane: xy
    wall_width: 1.0
    wall_height: 1.0
    wall_center_i: 0.0
    wall_center_j: 0.0
    wall_offset_k: 0.5
    include_hole: false
""")},
  'cluttered': {
    'scene_type': 'static', 
    'scene_params': {
      'static_scene_path': 'assets/benchmark_scenes/indoor_cluttered.yaml'
    }
  }
}

known_agent_configs = {
  'htw': [
    AgentConfig(
    mass=1.0,
    radius=0.020,
    max_velocity=0.010,
    approach_distance=0.25,
    force_list=["attractor_force", "velocity_heuristic_force"],
    force_configs={
      "attractor_force": {
        "k_gain": 0.50,
        "k_stiffness_linear": 1.0,
        "k_stiffness_angular": 0.00,
        "k_damping_linear": 2.5,
        "k_damping_angular": 0.025
      },
      "velocity_heuristic_force": {
        "k_gain": 1.0,
        "k_force": 5.0e-3,
        "detect_shell_radius": 0.50,
        "max_allowable_force": 100.0
      }
    }
  )],
  'passage': [
    AgentConfig(
    mass=1.0,
    radius=0.020,
    max_velocity=0.010,
    approach_distance=0.25,
    force_list=["attractor_force", "apf_heuristic_force"],
    force_configs={
      "attractor_force": {
        "k_gain": 2.0,
        "k_stiffness_linear": 1.0,
        "k_stiffness_angular": 0.00,
        "k_damping_linear": 2.5,
        "k_damping_angular": 0.025
      },
      "apf_heuristic_force": {
        "k_gain": 2.5,
        "k_force": 2.5e-4,
        "detect_shell_radius": 3.0,
        "max_allowable_force": 100.0
      }
    }
  ),
    AgentConfig(
    mass=1.0,
    radius=0.020,
    max_velocity=0.010,
    approach_distance=0.25,
    force_list=["attractor_force", "velocity_heuristic_force"],
    force_configs={
      "attractor_force": {
        "k_gain": 1.5,
        "k_stiffness_linear": 1.0,
        "k_stiffness_angular": 0.00,
        "k_damping_linear": 2.5,
        "k_damping_angular": 0.025
      },
      "velocity_heuristic_force": {
        "k_gain": 2.5,
        "k_force": 1.0e-2,
        "detect_shell_radius": 0.5,
        "max_allowable_force": 100.0
      }
    }
  )],
  'trap': [
    AgentConfig(
      mass=0.50,
      radius=0.020,
      max_velocity=0.010,
      approach_distance=0.25,
      force_list=["attractor_force", "apf_heuristic_force"],
      force_configs={
        "attractor_force": {
          "k_gain": 1.0,
          "k_stiffness_linear": 1.0,
          "k_stiffness_angular": 0.01,
          "k_damping_linear": 2.5,
          "k_damping_angular": 0.025
        },
        "apf_heuristic_force": {
          "k_gain": 1.0,
          "k_force": 5.0e-3,
          "detect_shell_radius": 1.0,
          "max_allowable_force": 100.0
        }
      }
    ),
    AgentConfig(
      mass=5.0,
      radius=0.020,
      max_velocity=0.010,
      approach_distance=0.25,
      force_list=["attractor_force", "velocity_heuristic_force"],
      force_configs={
        "attractor_force": {
          "k_gain": 0.50,
          "k_stiffness_linear": 1.0,
          "k_stiffness_angular": 0.00,
          "k_damping_linear": 2.5,
          "k_damping_angular": 0.025
        },
        "velocity_heuristic_force": {
          "k_gain": 10.0,
          "k_force": 1.0e-2,
          "detect_shell_radius": 0.20,
          "max_allowable_force": 1000.0
        }
      }
    )],
  'cluttered': [AgentConfig(
      mass=1.0,
      radius=0.020,
      max_velocity=0.010,
      approach_distance=0.010,
      force_list=["attractor_force", "velocity_heuristic_force"],
      force_configs={
        "attractor_force": {
          "k_gain": 1.0,
          "k_stiffness_linear": 1.0,
          "k_stiffness_angular": 0.00,
          "k_damping_linear": 2.5,
          "k_damping_angular": 0.00
        },
        "velocity_heuristic_force": {
          "k_gain": 2.0,
          "k_force": 2.5e-2,
          "detect_shell_radius": 0.25,
          "max_allowable_force": 100.0
        }
      }
    )],
}


known_planner_configs = {
"htw": {
  "path_length_cost_weight": 1.0,
  "goal_distance_cost_weight": 1.0,
  "obstacle_distance_cost_weight": 1.0,
  "trajectory_smoothness_cost_weight": 1.0,
},
"passage": {
  "path_length_cost_weight": 0.80,
  "goal_distance_cost_weight": 1.0e-01,
  "obstacle_distance_cost_weight": 0.25e-01,
  "trajectory_smoothness_cost_weight": 0.0,
},
"trap": {
  "path_length_cost_weight": 1.0,
  "goal_distance_cost_weight": 3.0e-02,
  "obstacle_distance_cost_weight": 1.0e-01,
  "trajectory_smoothness_cost_weight": 1.0e+05,
},
"cluttered": {
  "path_length_cost_weight": 1.0,
  "goal_distance_cost_weight": 1.0,
  "obstacle_distance_cost_weight": 1.0,
  "trajectory_smoothness_cost_weight": 1.0,
},
}


""" --- Automated Plan Generation --- """

def get_base_workload():
  global base_args
  return WorkloadConfig(
    metadata=Metadata(
      created_at=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
      plan_name=base_args["plan_name"],
      label="", # populate
      tags=["None"], # populate
      info="None", # populate
      run_planner_profiler=False
    ),
    percept_config=PerceptConfig(
      namespace="oriented_pointmass",
      mode="static",
      scene_config=SceneConfig(
        scene_type="", # populate
        scene_params={} # populate
      ),
      fields_config=FieldsConfig(
        agent_radius=base_args["agent_radius"],
        mass_radius=base_args["mass_radius"],
        potential_detect_shell_rad=base_args["potential_detect_shell_rad"],
        show_processing_delay=False,
        show_requests=False,
        use_cpu=False # populate
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
      delta_t=0.020,
      max_prediction_steps=0, # populate
      planning_frequency=30,
      agent_switch_factor=1.0,
      path_length_cost_weight=0.0, # populate
      goal_distance_cost_weight=0.0, # populate
      obstacle_distance_cost_weight=0.0, # populate
      trajectory_smoothness_cost_weight=0.0, # populate
      publish_trajectory=base_args["publish_trajectory"],
      poses=base_args["poses"],
      agents=[]
    ),
  )

def _generate_poses():
  global experiment_args
  random.seed(42)
  num_poses = experiment_args['num_poses']

  start_orientation = [1.0, 0.0, 0.0, 0.0]
  goal_orientation = [1.0, 0.0, 0.0, 0.0]
  rangex = 0.5
  rangez = 0.5

  poses = []
  for i in range(num_poses):
    start_pos = [random.uniform(-rangex, rangex), -2.0, random.uniform(-rangez, rangez)]
    goal_pos = [random.uniform(-rangex, rangex), 2.0, random.uniform(-rangez, rangez)]
    poses.append({
      "start_pos": start_pos,
      "goal_pos": goal_pos,
      "start_orientation": start_orientation,
      "goal_orientation": goal_orientation
    })

  return poses

def _generate_workload_params():
  global experiment_args, base_args, known_planner_configs
  cpu_options = experiment_args['cpu_options']
  planning_windows = experiment_args['planning_windows']
  poses = _generate_poses()
  delim1 = experiment_args['delim1']
  delim2 = experiment_args['delim2']
  workload_id = 1 # global workload id

  workloads = OrderedDict()
  for scene_id, scene_label in scene_ids.items():
    for j, use_cpu in cpu_options.items():
      for k, planning_window in planning_windows.items():
        for l, pose in enumerate(poses):
          workload = {}

          # workload unique iteration label
          workload['label'] = f"scene{delim2}{scene_id}{delim1}cpu{delim2}{j}{delim1}predsteps{delim2}{k}{delim1}pose{delim2}{l}"
          filename = f"{workload_id}{delim1}{workload['label']}.yaml"

          workload['tags'] = [
            base_args['plan_name'],
            f"scene{delim2}{scene_id}",
            f"cpu{delim2}{j}",
            f"predsteps{delim2}{k}",
            f"pose{delim2}{l}"
          ]

          workload['scene_type'] = scene_config[scene_label]['scene_type']
          workload['scene_params'] = scene_config[scene_label]['scene_params']
          workload['use_cpu'] = use_cpu
          workload['max_prediction_steps'] = planning_window
          workload['poses'] = pose
          workload['agents_config'] = known_agent_configs[scene_label]
          workload['planner_config'] = known_planner_configs[scene_label]

          workloads[filename] = (workload, experiment_args['runtime'])
          workload_id += 1

  return workloads

workload_params_keys = [
  'label',
  'tags',
  'scene_type',
  'scene_params',
  'use_cpu',
  'max_prediction_steps',
  'poses',
  'agents_config',
  'planner_config'
]
    
def _generate_workload_yaml(workload_params):
  missing_keys = [key for key in workload_params_keys if key not in workload_params]
  if missing_keys:
    raise ValueError(f"Missing required parameters: {', '.join(missing_keys)}")
  
  workload = deepcopy(get_base_workload())
  workload.metadata.label = workload_params['label']
  workload.metadata.tags = workload_params['tags']
  workload.percept_config.scene_config.scene_type = workload_params['scene_type']
  workload.percept_config.scene_config.scene_params = workload_params['scene_params']
  workload.percept_config.fields_config.use_cpu = workload_params['use_cpu']
  workload.planner_config.max_prediction_steps = workload_params['max_prediction_steps']
  workload.planner_config.poses = workload_params['poses']
  workload.planner_config.agents = workload_params['agents_config']
  workload.planner_config.path_length_cost_weight = workload_params['planner_config']['path_length_cost_weight']
  workload.planner_config.goal_distance_cost_weight = workload_params['planner_config']['goal_distance_cost_weight']
  workload.planner_config.obstacle_distance_cost_weight = workload_params['planner_config']['obstacle_distance_cost_weight']
  workload.planner_config.trajectory_smoothness_cost_weight = workload_params['planner_config']['trajectory_smoothness_cost_weight']

  return workload.to_yaml()

def main():
  plan_directory = Path(os.path.dirname(__file__))
  plan_index_file = plan_directory / "run.plan"
  os.makedirs(plan_directory / "workloads", exist_ok=True)

  # generate workloads
  workloads = _generate_workload_params()

  plan = OrderedDict({'plan_name': base_args["plan_name"], 'workloads': [], 'runtime': []})	
  for filename, (workload, runtime) in workloads.items():
    workload = _generate_workload_yaml(workload)
    workload_path = plan_directory / "workloads" / filename
    with open(workload_path, "w") as f:
      f.write(workload)

    plan['workloads'].append(f"eval/plans/{plan_directory.name}/workloads/{filename}")
    plan['runtime'].append(runtime)

  # update index file
  with open(plan_index_file, "w") as f:
    yaml.dump(plan, f)

if __name__ == "__main__":
  main()
