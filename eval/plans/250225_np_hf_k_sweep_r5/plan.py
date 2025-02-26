BASE_WORKLOAD_YAML="""
metadata:
  created_at: ""
  plan_name: "250225_np_hf_k_sweep_r5"
  label: ""
  tags:
    - "narrow_passage"
  info: "Benchmark k-gain values for each heuristic_for_narrow_passage"

percept_config:
  mode: "static" # realsense, sim, or static 
  scene_config:
    scene_type: "generated" # static, or generated
    scene_params: 
      scene_generation_type: "narrow_passage"
      scene_generation_params:
        wall1:
          spacing: 0.10
          plane: "xz"
          wall_width: 6.0
          wall_height: 6.0
          wall_center_i: 0.0
          wall_center_j: 0.0
          wall_offset_k: 0.5
          include_hole: true
          hole_center_x: -0.5
          hole_center_y: 0.0
          hole_width: 1.0
          hole_height: 6.0
        wall2:
          spacing: 0.10
          plane: "xz"
          wall_width: 6.0
          wall_height: 6.0
          wall_center_i: 0.0
          wall_center_j: 0.0
          wall_offset_k: 1.0
          include_hole: true
          hole_center_x: 0.5
          hole_center_y: 0.0
          hole_width: 1.0
          hole_height: 6.0
  rviz_config:
    show_rviz: false
    rviz_config_path: "rviz2/planning.rviz"

planner_config:
  experiment_type: "oriented_pointmass"
  loop_frequency: 1000
  service_timeout: 5000
  delta_t: 0.010
  max_prediction_steps: 30
  planning_frequency: 30
  agent_switch_factor: 0.9
  path_length_cost_weight: 1.0
  goal_distance_cost_weight: 1.0
  obstacle_distance_cost_weight: 1.0
  poses:
    start_pos:  [0.0, 0.0, 0.0]
    goal_pos: [0.0, 1.5, 0.0]
    start_orientation: [1.0, 0.0, 0.0, 0.0]
    goal_orientation: [1.0, 0.0, 0.0, 0.0]
"""


PROCEDURAL_FIELDS_CONFIG = """
k_cf_velocity: 0.001
k_cf_obstacle: 0.001
k_cf_goal: 0.001
k_cf_goalobstacle: 0.001
k_cf_random: 0.001
agent_radius: 0.05
mass_radius: 0.05
max_allowable_force: 20.0
publish_force_vector: false
show_processing_delay: false
show_requests: false
"""

PROCEDURAL_AGENT_YAML = """
mass: 1.0
radius: 0.05
max_velocity: 0.1
approach_distance: 0.25
forces:
    attractor_force:
        k_gain: 1.0
        k_stiffness_linear: 1.0
        k_stiffness_angular: 0.01
        k_damping_linear: 2.5
        k_damping_angular: 0.025
"""


import yaml
from pathlib import Path
import os
from copy import deepcopy
from yaml.representer import Representer
from collections import OrderedDict
import numpy as np
import datetime
# Add custom representer to maintain dictionary order
def ordered_dict_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, ordered_dict_representer)

def _generate_workloads(workload_configs:dict):
    plans_directory = Path(os.path.dirname(__file__))
    plan = OrderedDict({'workloads': []})

    for i, (label, workload_config) in enumerate(workload_configs.items()):
        # get the heuristic name
        workload_name = f"{i+1}_{label}.yaml"
        workload_filepath = plans_directory / workload_name
        with open(workload_filepath, "w") as f:
            yaml.dump(workload_config, f, sort_keys=False)
        # Convert to relative path from percept directory by including 'eval/plans/...'
        relative_path = f"eval/plans/{plans_directory.name}/{workload_name}"
        plan['workloads'].append(relative_path)
    
    plan_filepath = plans_directory / "run.plan"
    with open(plan_filepath, "w") as f:
        yaml.dump(plan, f, sort_keys=False)

def _generate_procedural_configs(base_workload_config:dict):
    configs = {}
    heuristics = [
        "velocity_heuristic_force",
        "goal_heuristic_force",
        "obstacle_heuristic_force",
        "goalobstacle_heuristic_force", 
    ]
    k_gains = np.logspace(np.log10(0.000001), np.log10(0.001), 20)
    
    # Initialize base config with empty agents list
    base_workload_config["planner_config"]["agents"] = []

    for heuristic in heuristics:
        print(heuristic)
        # Create agent config
        agent = yaml.safe_load(PROCEDURAL_AGENT_YAML)
        agent["forces"][heuristic] = {
            "k_gain": 1.0,
            "detect_shell_radius": 5.0
        }

        for k_gain in k_gains:
            label = f"{heuristic}_{k_gain:.9f}"
            print(label)

            # Create new config copy for this combination
            config = deepcopy(base_workload_config)
            config["planner_config"]["agents"] = [agent]  # Add single agent
            
            # Set fields config
            fields_config = yaml.safe_load(PROCEDURAL_FIELDS_CONFIG)
            fields_config["k_cf_" + heuristic.split('_')[0]] = float(k_gain)
            config["percept_config"]["fields_config"] = fields_config
            
            config["metadata"]["info"] = f"Benchmark k-gain values for {heuristic}"
            config["metadata"]["tags"].append(heuristic)
            config["metadata"]["label"] = label
            config["metadata"]["created_at"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            configs[label] = config

    return configs

def _generate_plan(base_workload_yaml: str):
    base_workload_config = yaml.safe_load(base_workload_yaml)
    workload_configs = _generate_procedural_configs(base_workload_config)
    _generate_workloads(workload_configs)

def main():
    _generate_plan(BASE_WORKLOAD_YAML)

if __name__ == "__main__":
    main()
