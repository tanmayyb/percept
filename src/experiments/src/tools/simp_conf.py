#!/usr/bin/env python3

import yaml
import re
import sys
import os

class SimpConf:
    def __init__(self, input_file):
        self.raw_data = self._load_file(input_file)
        self.params = self._parse_params()
        self.cost_defs = self._parse_cost_defs()
        
    def _load_file(self, path):
        with open(path, 'r') as f:
            return f.read()

    def _parse_params(self):
        pattern = re.compile(r'^(\w+)\s*=\s*([^#\n|]+)', re.MULTILINE)
        matches = pattern.findall(self.raw_data)
        data = {}
        for key, val in matches:
            has_brackets = '[' in val
            val = val.strip().replace('[', '').replace(']', '').rstrip(',')
            items = [i.strip() for i in val.split(',') if i.strip()]
            parsed_items = []
            
            for item in items:
                if item == '...': continue
                if item == '1': parsed_items.append(True)
                elif item == '0': parsed_items.append(False)
                elif "'" in item or '"' in item: parsed_items.append(item.strip("'\""))
                else:
                    try: parsed_items.append(float(item))
                    except ValueError: parsed_items.append(item)
            
            if len(parsed_items) == 1 and not has_brackets:
                data[key] = parsed_items[0]
            else:
                data[key] = parsed_items
        return data

    def _parse_cost_defs(self):
        defs = {}
        cost_pattern = re.compile(r'^cost:\s*(\w+)\s*\|\s*([\d.e+-]+)\s*\|?(.*)', re.MULTILINE)
        matches = cost_pattern.findall(self.raw_data)
        for c_type, wght, extras in matches:
            p = {'weight': float(wght)}
            if extras.strip():
                for pair in extras.split(','):
                    if '=' in pair:
                        k, v = pair.split('=')
                        p[k.strip()] = float(v.strip())
            defs[c_type] = p
        return defs

    def generate_yaml(self):
        experiment_type = self.params.get('experiment', 'manipulator')
        
        # Override logic: use loop_freq from input.conf if present, else use default
        default_freq = 10000
        loop_frequency = int(self.params.get('loop_freq', default_freq))
        
        config = {
            'loop_frequency': loop_frequency,
            'publishers': ['target', 'pose'],
            'subscribers': [],
            'callback_clients': ['obstacle_distance_cost_client'],
            'callback_servers': [],
            'publisher': {
                'target': {'type': 'gafro_motor', 'topic': 'target', 'callback_queue': 'target'},
                'pose': {'type': 'gafro_motor', 'topic': 'pose', 'callback_queue': 'pose'}
            },
            'callback_client': {
                'obstacle_distance_cost_client': {
                    'type': 'obstacle_distance_cost', 'callback_request': 'get_min_obstacle_distance',
                    'callback_response': 'obstacle_distance_response', 'timeout': 10
                }
            },
            'cf_planner': {
                'n_agents': 0, 
                'agent_type': experiment_type, 
                'delta_t': 0.01,
                'max_prediction_steps': int(self.params.get('p_steps', 100)),
                'planning_frequency': 100,
                'agent_switch_factor': float(self.params.get('k_sw', 1.0)),
                'costs': []
            }
        }

        if experiment_type == 'manipulator':
            config['publishers'].append('robot')
            config['publisher']['robot'] = {
                'type': 'gafro_system', 'description': 'robots/panda/panda.yaml',
                'topic': 'robot', 'callback_queue': 'robot', 'frame': 'world',
                'color': {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 0.0}
            }
            config['callback_servers'] = ['set_goal_server']
            config['callback_server'] = { 'set_goal_server': {'type': 'set_goal_server', 'callback': 'set_goal_callback'} }

        cost_map = {
            'goal': 'goal_distance_cost', 'obs': 'obstacle_distance_cost',
            'path': 'path_length_cost', 'traj': 'trajectory_smoothness_cost'
        }
        
        c_types = self.params.get('cst_type', [])
        c_en = self.params.get('cst_en', [])
        for i, enabled in enumerate(c_en):
            if enabled:
                short_name = c_types[i]
                full_name = cost_map.get(short_name, f"{short_name}_cost")
                config['cf_planner']['costs'].append(full_name)
                config['cf_planner'][full_name] = self.cost_defs.get(short_name, {'weight': 1.0})

        agt_types, agt_en = self.params.get('agt_type', []), self.params.get('agt_en', [])
        config['cf_planner']['n_agents'] = sum(1 for e in agt_en if e)

        type_mapping = {
            'apf': 'apf_heuristic_force', 
            'vel': 'velocity_heuristic_force',
            'gol': 'goal_heuristic_force', 
            'gob': 'goalobstacle_heuristic_force',
            'obs': 'obstacle_heuristic_force', 
            'rdm': 'random_heuristic_force'
        }

        agent_idx = 1
        for i, enabled in enumerate(agt_en):
            if not enabled: continue
            a_key = f'agent_{agent_idx}'
            raw_type = agt_types[i]
            force_type = type_mapping.get(raw_type)
            if not force_type: continue
             
            if force_type not in config['callback_clients']:
                config['callback_clients'].append(force_type)
                config['callback_client'][force_type] = {
                    'type': force_type, 'callback_request': f'get_{force_type}',
                    'callback_response': f'{force_type}_response', 'timeout': 10
                }

            config['cf_planner'][a_key] = {
                'mass': self.params['mass'][i], 'radius': self.params['rad'][i],
                'forces': ['attractor_force', force_type],
                'attractor_force': {
                    'k_gain': self.params['g_attr'][i], 'k_stiffness_linear': self.params['k_sl'][i],
                    'k_stiffness_angular': self.params['k_sa'][i], 'k_damping_linear': self.params['k_dl'][i],
                    'k_damping_angular': self.params['k_da'][i],
                },
                force_type: {
                    'k_gain': self.params['g_repl'][i], 'k_force': self.params['k_f'][i],
                    'detect_shell_radius': self.params['r_det'][i], 'max_allowable_force': self.params['f_max'][i],
                }
            }
            agent_idx += 1

        return yaml.dump(config, sort_keys=False, default_flow_style=False)

from ament_index_python.packages import get_package_share_directory

def main():
    if len(sys.argv) < 2: return
    parser = SimpConf(sys.argv[1])
    
    experiment_type = parser.params.get('experiment', 'manipulator')
    
    try:
        package_share = get_package_share_directory('experiments')
        base_path = os.path.join(package_share, experiment_type)
    except Exception as e:
        print(f"Error: Package 'experiments' not found in workspace: {e}")
        return

    if experiment_type == 'oriented_pointmass':
        output_path = os.path.join(base_path, 'oriented_pointmass_config.yaml')
    else:
        output_path = os.path.join(base_path, 'manipulator_config.yaml')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(parser.generate_yaml())

if __name__ == '__main__':
    main()