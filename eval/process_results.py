#!/usr/bin/env python3
import argparse
import yaml
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from ament_index_python.packages import get_package_share_directory

pio.templates.default = "plotly_white"
pio.renderers.default = "browser"

class MetricsReader:
	def __init__(self, filepath: str, success_time_limit: float = None, acceptable_min_distances_to_target: float = None):
		self.filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
		if not self.filepath.exists():
			raise FileNotFoundError(f"Error: File '{self.filepath}' does not exist!")
		self.success_time_limit = success_time_limit
		self.acceptable_min_distances_to_target = acceptable_min_distances_to_target
		self._load_data()
		
	def _load_data(self):
		with open(self.filepath, 'r') as file:
			documents = list(yaml.safe_load_all(file))
		self.log = documents

		# read from config
		workload_config = documents[0]['workload_config']
		self.workload_config = workload_config
		poses = workload_config['planner_config']['poses']
		agents = workload_config['planner_config']['agents']
		self.FORCE_CONFIG_LABEL = 'forces' if 'forces' in list(agents[0].keys()) else 'force_configs'
		self.start_pos = poses['start_pos']
		self.goal_pos = poses['goal_pos']
		self.start_orientation = poses['start_orientation']
		self.goal_orientation = poses['goal_orientation']
		self.agent_radius = workload_config['percept_config']['fields_config']['agent_radius']
		self.mass_radius = workload_config['percept_config']['fields_config']['mass_radius']

		# read from records
		self.agent_poses = []
		self.target_poses = []
		self.pointclouds = []
		self.best_agent = []
		self.agent_planning_times = []
		self.agent_costs = []
		self.agent_trajectory_smoothness_costs = []
		self.agent_path_length_costs = []
		self.agent_goal_distance_costs = []
		self.agent_obstacle_distance_costs = []
		
		def process_record(record):
			if record is None:
				return None
			result = {
				'timestamp': record.get('timestamp'),
				'data': None,
				'type': record.get('type', '')
			}
			if result['type'] == 'agent_pose':
				result['data'] = record
			elif result['type'] == 'target_pose':
				result['data'] = record
			elif result['type'] == 'pointcloud':
				result['data'] = record
			elif result['type'] == 'best_agent_name':
				result['data'] = self._parse_best_agent_record(record)
			elif result['type'] == 'agent_planning_time':
				result['data'] = record
			elif result['type'] == 'agent_cost':
				result['data'] = record
			elif result['type'] == 'agent_trajectory_smoothness_cost':
				result['data'] = record
			elif result['type'] == 'agent_path_length_cost':
				result['data'] = record
			elif result['type'] == 'agent_goal_distance_cost':
				result['data'] = record
			elif result['type'] == 'agent_obstacle_distance_cost':
				result['data'] = record
			return result

		# Process records in parallel
		with ThreadPoolExecutor() as executor:
			futures = list(tqdm(
				executor.map(process_record, documents[1:]),
				total=len(documents[1:]),
				desc="Processing records"
			))

		# Collect results
		for result in futures:
			if result is not None:                
				if result['data'] is not None:
					if result['type'] == 'agent_pose':
						self.agent_poses.append(result['data'])
					elif result['type'] == 'target_pose':
						self.target_poses.append(result['data'])
					elif result['type'] == 'pointcloud':
						self.pointclouds.append(result['data'])
					elif result['type'] == 'best_agent_name':
						self.best_agent.append(result['data'])
					elif result['type'] == 'agent_planning_time':
						self.agent_planning_times.append(result['data'])
					elif result['type'] == 'agent_cost':
						self.agent_costs.append(result['data'])
					elif result['type'] == 'agent_trajectory_smoothness_cost':
						self.agent_trajectory_smoothness_costs.append(result['data'])
					elif result['type'] == 'agent_path_length_cost':
						self.agent_path_length_costs.append(result['data'])
					elif result['type'] == 'agent_goal_distance_cost':
						self.agent_goal_distance_costs.append(result['data'])
					elif result['type'] == 'agent_obstacle_distance_cost':
						self.agent_obstacle_distance_costs.append(result['data'])
		# convert to pandas dataframe
		self.filename = self.filepath.name
		self.agent_poses = pd.DataFrame(self.agent_poses)
		self.target_poses = pd.DataFrame(self.target_poses)
		self.pointclouds = pd.DataFrame(self.pointclouds)
		self.best_agent = pd.DataFrame(self.best_agent)
		self.agent_planning_times = pd.DataFrame(self.agent_planning_times)
		self.agent_costs = pd.DataFrame(self.agent_costs)
		self.agent_trajectory_smoothness_costs = pd.DataFrame(self.agent_trajectory_smoothness_costs)
		self.agent_path_length_costs = pd.DataFrame(self.agent_path_length_costs)
		self.agent_goal_distance_costs = pd.DataFrame(self.agent_goal_distance_costs)
		self.agent_obstacle_distance_costs = pd.DataFrame(self.agent_obstacle_distance_costs)

		self.info = None


		if len(self.pointclouds) == 0 and self.workload_config['percept_config']['mode'] == 'static':
			if self.workload_config['percept_config']['scene_config']['scene_type'] == 'generated':
				from mp_eval.classes.scene_generator import SceneGenerator
				from mp_eval.classes.workload import SceneConfig
				scene_config = SceneConfig.from_config(self.workload_config['percept_config']['scene_config'])
				scene_generator = SceneGenerator(scene_config, "", "")
				pointcloud_data = scene_generator.get_assets()[0]
			elif self.workload_config['percept_config']['scene_config']['scene_type'] == 'static':
				scene_path = self.workload_config['percept_config']['scene_config']['scene_params']['static_scene_path']
				pkg_share = Path(get_package_share_directory('percept'))
				with open(pkg_share / scene_path, "r") as f:
					pointcloud_data = yaml.load(f, Loader=yaml.CLoader)['obstacles']
			self.pointclouds = pd.DataFrame({"points": [[obs["position"] for obs in pointcloud_data]]})

		self.process_info_and_analysis()


	@staticmethod
	def _parse_best_agent_record(record: dict):
		# Extract the integer agent id from the best_agent_name field.
		return {
			'timestamp': record['timestamp'],
			'agent_id': int(record['best_agent_name'].split('/')[-1].replace('agent_', '')),
			'type': 'best_agent_name'
		}


	def _update_2d_layout(self, fig, title, xaxis_title, yaxis_title, extra=None):
		layout = dict(
			title=title,
			xaxis_title=xaxis_title,
			yaxis_title=yaxis_title,
			width=800,
			height=500,
			xaxis=dict(rangeslider=dict(visible=True))
		)
		if extra:
			layout.update(extra)
		fig.update_layout(**layout)
		return fig
	
	def success_criteria(self, num_collisions, elapsed_motion_runtime, min_distances_to_target):
		time_limit = self.success_time_limit
		acceptable_min_distances_to_target = self.acceptable_min_distances_to_target
		# criteria
		if num_collisions > 0 \
			or (time_limit is not None and elapsed_motion_runtime > time_limit) \
			or (acceptable_min_distances_to_target is not None and min_distances_to_target > acceptable_min_distances_to_target):
			failure_reason = []
			if num_collisions > 0:
				failure_reason.append("Collision")
			if time_limit is not None and elapsed_motion_runtime > time_limit:
				failure_reason.append("Elapsed motion runtime > time limit")
			if acceptable_min_distances_to_target is not None and min_distances_to_target > acceptable_min_distances_to_target:
				failure_reason.append("Minimum distance to target > acceptable limit")
			
			return False, failure_reason
		else:
			return True, None

	def process_info_and_analysis(self):
		start_time = time.time()
		label = self.workload_config['metadata']['label']
		# Convert pointcloud to a NumPy array (shape: [M, 3])
		pointcloud = np.array(self.pointclouds.iloc[0]['points'])
		num_agents = len(self.workload_config['planner_config']['agents'])
		detection_shell_radius = {
			f"agent_{i+1}": next(v for k, v in agent[self.FORCE_CONFIG_LABEL].items() if 'heuristic_force' in k)['detect_shell_radius']
			for i, agent in enumerate(self.workload_config['planner_config']['agents'])
		}

		# Vectorize extraction of agent and target positions, and timestamps.
		# Each "position" field is a dict with keys 'x', 'y', 'z'
		agent_positions = np.array([[pos['x'], pos['y'], pos['z']] for pos in self.agent_poses['position']])
		target_positions = np.array([[pos['x'], pos['y'], pos['z']] for pos in self.target_poses['position']])
		agent_timestamps = self.agent_poses['timestamp'].to_numpy()
		target_timestamps = self.target_poses['timestamp'].to_numpy()

		# Find the closest temporal matches
		# For each agent timestamp, find the index of the closest target timestamp
		target_indices = np.searchsorted(target_timestamps, agent_timestamps)
		# Handle edge cases where searchsorted returns len(target_timestamps)
		target_indices = np.clip(target_indices, 0, len(target_timestamps) - 1)

		# For each target index, check if the previous index is actually closer
		mask = target_indices > 0
		closer_prev = np.abs(target_timestamps[target_indices[mask] - 1] - agent_timestamps[mask]) < \
									np.abs(target_timestamps[target_indices[mask]] - agent_timestamps[mask])
		target_indices[mask] = target_indices[mask] - closer_prev

		# Align the arrays using the matched indices
		agent_positions = agent_positions
		target_positions = target_positions[target_indices]
		timestamps = agent_timestamps

		# Compute distance to target for each pose.
		distance_to_target = np.linalg.norm(agent_positions - target_positions, axis=1)
		min_idx = np.argmin(distance_to_target)
		min_distances_to_target = distance_to_target[min_idx]
		min_distances_to_target_timestamp = timestamps[min_idx]

		# Compute pairwise distances from each agent position to every point in the pointcloud.
		# This yields an array of shape (N, M) where N = number of agent poses and M = number of points.
		dists = np.linalg.norm(agent_positions[:, None, :] - pointcloud[None, :, :], axis=2)
		distances_to_obstacles = dists  # Already in the desired array-of-arrays form.
		min_distances_to_obstacles = dists.min(axis=1)
		max_distances_to_obstacles = dists.max(axis=1)

		# Determine when the agent begins to move (i.e. when the position differs from the experiment start by >0.001)
		experiment_start_pos = agent_positions[0]
		norms_from_start = np.linalg.norm(agent_positions - experiment_start_pos, axis=1)
		motion_indices = np.where(norms_from_start > 0.001)[0]
		if motion_indices.size > 0:
			motion_start_index = motion_indices[0]
			motion_start_pos = agent_positions[motion_start_index]
			motion_start_timestamp = timestamps[motion_start_index]
			# Compute the path length after motion has begun.
			path_length = np.sum(np.linalg.norm(np.diff(agent_positions[motion_start_index:], axis=0), axis=1))
		else:
			motion_start_index = None
			motion_start_pos = None
			motion_start_timestamp = timestamps[0]
			path_length = 0.0

		# Collision detection (vectorized)
		collision_threshold = self.agent_radius + self.mass_radius
		collision_mask = dists < collision_threshold
		num_collisions = int(collision_mask.sum())
		# For each agent pose, repeat the timestamp for every colliding point.
		collision_timestamps = np.concatenate([
			np.full(np.count_nonzero(collision_mask[i]), ts)
			for i, ts in enumerate(timestamps)
		]) if len(timestamps) > 0 else np.array([])
		# Build collision_indices: for each agent pose, add its index repeated once per point in the pointcloud.
		collision_indices = np.repeat(np.arange(len(timestamps)), len(pointcloud))

		# Calculate agent switches and residency times.
		current_agent = None
		last_timestamp = None
		agent_switches = 0
		agent_residency = {}
		for _, row in self.agent_planning_times.iterrows():
			agent_id = f"agent_{row['agent_id']}"
			ts = row['timestamp']
			if current_agent is None:
				current_agent = agent_id
				last_timestamp = ts
				agent_residency[agent_id] = 0.0
			elif agent_id != current_agent:
				agent_switches += 1
				agent_residency[current_agent] += ts - last_timestamp
				if agent_id not in agent_residency:
					agent_residency[agent_id] = 0.0
				current_agent = agent_id
				last_timestamp = ts
		if current_agent is not None and last_timestamp is not None:
			agent_residency[current_agent] += (
				self.agent_planning_times['timestamp'].iloc[-1]
				- self.agent_planning_times['timestamp'].iloc[0]
			)
		total_time = (
			self.agent_planning_times['timestamp'].iloc[-1]
			- self.agent_planning_times['timestamp'].iloc[0]
		)
		agent_residency_pct = {agent: (t / total_time) * 100 for agent, t in agent_residency.items()}

		mean_planning_time = np.mean(self.agent_planning_times['planning_time'].iloc[1:])
		stddev_planning_time = np.std(self.agent_planning_times['planning_time'].iloc[1:])

		experiment_start_timestamp = timestamps.min()
		experiment_end_timestamp = timestamps.max()
		elapsed_experiment_runtime = experiment_end_timestamp - experiment_start_timestamp
		elapsed_motion_runtime = min_distances_to_target_timestamp - motion_start_timestamp

		success, reason = self.success_criteria(num_collisions, elapsed_motion_runtime, min_distances_to_target)

		print(f"File: {self.filename} took {time.time() - start_time:.3f} seconds to process")

		self.info = {
			'label': label,
			'success': str(success),
			'reason': reason,
			'timestamps': timestamps,
			'num_agents': num_agents,
			'num_obstacles': len(pointcloud),
			'num_collisions': num_collisions,
			'collision_timestamps': collision_timestamps,
			'collision_indices': collision_indices,
			'detection_shell_radius': detection_shell_radius,
			'agent_switches': agent_switches,
			'agent_residency_pct': agent_residency_pct,
			'agent_positions': agent_positions,
			'distances_to_obstacles': distances_to_obstacles,
			'min_distances_to_obstacles': min_distances_to_obstacles,
			'max_distances_to_obstacles': max_distances_to_obstacles,
			'distance_to_target': distance_to_target,
			'path_length': path_length,
			'motion_start_timestamp': motion_start_timestamp,
			'motion_start_pos': motion_start_pos,
			'experiment_start_timestamp': experiment_start_timestamp,
			'experiment_end_timestamp': experiment_end_timestamp,
			'elapsed_experiment_runtime': elapsed_experiment_runtime,
			'elapsed_motion_runtime': elapsed_motion_runtime,
			'min_distances_to_target': min_distances_to_target,
			'min_distances_to_target_timestamp': min_distances_to_target_timestamp,
			'mean_planning_time': mean_planning_time,
			'stddev_planning_time': stddev_planning_time,
		}


	def format_stats(self):
		stats = {}
		for key, value in self.info.items():
			if isinstance(value, np.ndarray):
				if key in ['timestamps', 'collision_timestamps', 'collision_indices']:
					continue
				else:
					stats[key] = {
						'min': float(f"{float(np.min(value)):.3f}"),
						'max': float(f"{float(np.max(value)):.3f}"),
						'mean': float(f"{float(np.mean(value)):.3f}"), 
						'std': float(f"{float(np.std(value)):.3f}")
					}
			elif isinstance(value, (int, float, np.float64, np.float32, np.int64, np.int32)):
				stats[key] = float(f"{float(value):.3f}")
			elif isinstance(value, dict):
				# Handle nested dictionaries
				stats[key] = {k: float(f"{float(v):.3f}") if isinstance(v, (int, float, np.float64, np.float32, np.int64, np.int32)) else v 
							for k, v in value.items()}
			else:
				stats[key] = value
		return yaml.dump(stats, sort_keys=False, default_flow_style=False).split('\n')

	def print_stats(self):
		print("\nExperiment Info:")
		print("-" * 50)
		print("\n".join(self.format_stats()))
		print("-" * 50)

	def save_stats(self):
		output_path = self.filepath.parent / f"{self.filepath.stem}.stats"
		with open(output_path, 'w') as f:
			f.write("Experiment Info:\n")
			f.write("-" * 50 + "\n")
			f.write("\n".join(self.format_stats()))
			f.write("\n" + "-" * 50)
		

	def plot_scene(self):
		# Build agent path coordinates and timestamps.
		x_coords = []
		y_coords = []
		z_coords = []
		timestamps = []
		for i, row in self.agent_poses.iterrows():
			x_coords.append(row['position']['x'])
			y_coords.append(row['position']['y'])
			z_coords.append(row['position']['z'])
			timestamps.append(row['timestamp'] - self.info['experiment_start_timestamp'])

		# Extract pointcloud coordinates (assuming one pointcloud record).
		points = self.pointclouds.iloc[0]['points']
		pointcloud_x, pointcloud_y, pointcloud_z = zip(*points)

		agent_trace = go.Scatter3d(
			x=x_coords,
			y=y_coords,
			z=z_coords,
			mode='lines+markers',
			marker=dict(
				size=1,
				color=timestamps,
				colorscale='Turbo',
				showscale=True,
				colorbar=dict(title='Time (s)')
			),
			line=dict(color='blue', width=2),
			name='Agent Path'
		)
		pointcloud_trace = go.Scatter3d(
			x=pointcloud_x,
			y=pointcloud_y,
			z=pointcloud_z,
			mode='markers',
			marker=dict(
				size=2,
				color=pointcloud_z,  # Use Z values for coloring
				colorscale='Turbo',  # Use Turbo colormap
				showscale=False,      # Show the colorbar
				colorbar=dict(title='Z Position')
			),
			name='Pointcloud'
		)
		# Create start and goal markers.
		start_marker = go.Scatter3d(
			x=[self.start_pos[0]],
			y=[self.start_pos[1]],
			z=[self.start_pos[2]],
			mode='markers',
			marker=dict(size=8, color='blue', symbol='diamond'),
			name='Start'
		)
		goal_marker = go.Scatter3d(
			x=[self.goal_pos[0]],
			y=[self.goal_pos[1]],
			z=[self.goal_pos[2]],
			mode='markers',
			marker=dict(size=8, color='green', symbol='diamond'),
			name='Goal'
		)
		
		fig = go.Figure(data=[agent_trace, pointcloud_trace, start_marker, goal_marker])
		fig.update_layout(
			title='Agent Path and Pointcloud Visualization: ' + self.filename,
			scene=dict(
				xaxis_title='X Position',
				yaxis_title='Y Position',
				zaxis_title='Z Position',
				aspectmode='data'
			),
			width=800,
			height=800
		)
		return fig

	def plot_agent_planning_time(self):
		df = self.agent_planning_times
		y_values = df['planning_time']
		q1 = y_values.quantile(0.05)
		q3 = y_values.quantile(0.95)
		y_max = q3 + (q3 - q1) * 1.5

		fig = go.Figure()
		for agent_id in df['agent_id'].unique():
			agent_data = df[df['agent_id'] == agent_id]
			fig.add_trace(go.Scattergl(
				x=agent_data['timestamp'] - self.info['experiment_start_timestamp'],
				y=agent_data['planning_time'],
				mode='lines+markers',
				name=f'Agent {agent_id}',
				line=dict(shape='hv'),
				marker=dict(size=4, opacity=0.6)
			))
		self._update_2d_layout(
			fig,
			title='Agent Planning Time: ' + self.filename,
			xaxis_title='Time (seconds)',
			yaxis_title='Planning Time (seconds)',
			extra={'yaxis': {'range': [0.0, y_max]}}
		)
		return fig

	def plot_best_agent_selection(self):
		df = self.best_agent
		heuristic_mapping = {
			agent_id + 1: f"{agent_id + 1}: {list(agent[self.FORCE_CONFIG_LABEL].keys())[1].split('_')[0]}"
			for agent_id, agent in enumerate(self.workload_config['planner_config']['agents'])
		}
		df['heuristic'] = df['agent_id'].map(heuristic_mapping)
		fig = go.Figure()
		fig.add_trace(go.Scatter(
			x=df['timestamp'] - self.info['experiment_start_timestamp'],
			y=df['heuristic'],
			mode='lines+markers',
			name='Best Agent ID',
			line=dict(shape='hv', color='blue', width=0.5),
			marker=dict(size=2.5, color='blue')
		))
		self._update_2d_layout(
			fig,
			title='Best Agent Selection Over Time: ' + self.filename,
			xaxis_title='Time (seconds)',
			yaxis_title='Agent ID'
		)
		fig.update_layout(
			yaxis=dict(
				tickmode='linear',
				dtick=1,
				ticks='outside',
				ticklen=5
			),
			showlegend=True,
			legend=dict(x=1.1, y=1)
		)
		return fig

	def plot_distance_to_goal(self):

		timestamps = self.info['timestamps'] - self.info['experiment_start_timestamp']
		distances = self.info['distance_to_target']

		fig = go.Figure(data=go.Scatter(
			x=timestamps,
			y=distances,
			mode='lines+markers',
			name='Distance to Goal',
			marker=dict(size=6, color='blue'),
			line=dict(color='blue')
		))
		self._update_2d_layout(
			fig,
			title='Distance to Goal Over Time: ' + self.filename,
			xaxis_title='Time (seconds)',
			yaxis_title='Distance to Goal (meters)'
		)
		return fig

	def plot_distance_to_closest_obstacle(self):
		pointcloud = self.pointclouds.iloc[0]['points']
		timestamps = []
		agent_positions = []
		min_distances = []

		for _, row in self.agent_poses.iterrows():
			timestamps.append(row['timestamp'] - self.info['experiment_start_timestamp'])
			agent_positions.append([row['position']['x'], row['position']['y'], row['position']['z']])
			min_distances.append(np.min(np.linalg.norm(pointcloud - np.array(agent_positions[-1]), axis=1)))

		timestamps = np.array(timestamps)
		agent_positions = np.array(agent_positions) 
		min_distances = np.array(min_distances)
		
		fig = go.Figure(data=go.Scatter(
			x=timestamps,
			y=min_distances,
			mode='lines+markers',
			name='Distance to Closest Obstacle',
			marker=dict(size=6, color='red'),
			line=dict(color='red')
		))
		self._update_2d_layout(
			fig,
			title='Distance to Closest Obstacle Over Time: ' + self.filename,
			xaxis_title='Time (seconds)',
			yaxis_title='Distance to Closest Point (meters)'
		)
		return fig

def parse_args():
	parser = argparse.ArgumentParser(
		description="Process navigation metrics from a YAML file and generate plots."
	)
	parser.add_argument(
		"--filepath",
		type=str,
		required=True,
		help="Path to the metrics YAML file."
	)
	parser.add_argument(
		"--plot",
		action="store_true",
		help="If provided, show all plots."
	)
	return parser.parse_args()

def main():
	args = parse_args()
	filepath = Path(args.filepath) if Path(args.filepath).is_absolute() else Path('./eval/results/') / args.filepath

	try:
		reader = MetricsReader(filepath)
	except FileNotFoundError as e:
		print(e)
		return

	reader.print_stats()

	if args.plot:
		reader.plot_agent_planning_time().show()
		reader.plot_best_agent_selection().show()
		reader.plot_distance_to_goal().show()
		reader.plot_distance_to_closest_obstacle().show()
		reader.plot_scene().show()

if __name__ == "__main__":
	main()
