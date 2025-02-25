#!/usr/bin/env python3
import argparse
import yaml
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from pathlib import Path

pio.templates.default = "plotly_white"
pio.renderers.default = "browser"

class MetricsVisualizer:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
        if not self.filepath.exists():
            raise FileNotFoundError(f"Error: File '{self.filepath}' does not exist!")
        self._load_data()

    def _load_data(self):
        with open(self.filepath, 'r') as file:
            documents = list(yaml.safe_load_all(file))
        self.log = documents
        workload_config = documents[0]['workload_config']
        self.workload_config = workload_config
        poses = workload_config['planner_config']['poses']
        self.start_pos = poses['start_pos']
        self.goal_pos = poses['goal_pos']
        self.start_orientation = poses['start_orientation']
        self.goal_orientation = poses['goal_orientation']

        self.agent_poses = []
        self.target_poses = []
        self.pointclouds = []
        self.best_agent_list = []
        self.agent_planning_time_list = []
        start_timestamp = float('inf')
        end_timestamp = float('-inf')
        for record in documents[1:]:
            if record is not None:
                if 'timestamp' in record:
                    start_timestamp = min(start_timestamp, record['timestamp'])
                    end_timestamp = max(end_timestamp, record['timestamp'])
                record_type = record.get('type', '')
                if record_type == 'agent_pose':
                    self.agent_poses.append(record)
                elif record_type == 'target_pose':
                    self.target_poses.append(record)
                elif record_type == 'pointcloud':
                    self.pointclouds.append(record)
                elif record_type == 'best_agent_name':
                    self.best_agent_list.append(MetricsVisualizer._parse_best_agent_record(record))
                elif record_type == 'agent_planning_time':
                    self.agent_planning_time_list.append(record)
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.elapsed_time = end_timestamp - start_timestamp
        self.filename = self.filepath.name

    @staticmethod
    def _parse_best_agent_record(record: dict):
        # Extract the integer agent id from the best_agent_name field.
        return {
            'timestamp': record['timestamp'],
            'agent_id': int(record['best_agent_name'].split('/')[-1].replace('agent_', '')),
            'type': 'best_agent_name'
        }

    def get_summary(self):
        print('Elapsed Time:', self.elapsed_time)
        print('Start Position:', self.start_pos)
        print('Goal Position:', self.goal_pos)
        print('Start Orientation:', self.start_orientation)
        print('Goal Orientation:', self.goal_orientation)

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

    def plot_scene(self):
        # Build agent path coordinates and timestamps.
        x_coords = [pose['position']['x'] for pose in self.agent_poses]
        y_coords = [pose['position']['y'] for pose in self.agent_poses]
        z_coords = [pose['position']['z'] for pose in self.agent_poses]
        timestamps = [pose['timestamp'] - self.start_timestamp for pose in self.agent_poses]

        # Extract pointcloud coordinates (assuming one pointcloud record).
        points = self.pointclouds[0]['points']
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
            marker=dict(size=2, color='red', symbol='circle'),
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
        df = pd.DataFrame(self.agent_planning_time_list)
        y_values = df['planning_time']
        q1 = y_values.quantile(0.05)
        q3 = y_values.quantile(0.95)
        y_max = q3 + (q3 - q1) * 1.5

        fig = go.Figure()
        for agent_id in df['agent_id'].unique():
            agent_data = df[df['agent_id'] == agent_id]
            fig.add_trace(go.Scattergl(
                x=agent_data['timestamp'] - self.start_timestamp,
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
        df = pd.DataFrame(self.best_agent_list)
        heuristic_mapping = {
            agent_id + 1: f"{agent_id + 1}: {list(agent['forces'].keys())[1].split('_')[0]}"
            for agent_id, agent in enumerate(self.workload_config['planner_config']['agents'])
        }
        df['heuristic'] = df['agent_id'].map(heuristic_mapping)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'] - self.start_timestamp,
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
        agent_positions = np.array([[pose['position']['x'], pose['position']['y'], pose['position']['z']]
                                    for pose in self.agent_poses])
        goal_pos = np.array(self.goal_pos)
        distances = np.linalg.norm(agent_positions - goal_pos, axis=1)
        timestamps = np.array([pose['timestamp'] for pose in self.agent_poses]) - self.start_timestamp

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
        pointcloud = np.array(self.pointclouds[0]['points'])
        timestamps = []
        min_distances = []
        for pose in self.agent_poses:
            agent_pos = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
            distances_to_points = np.linalg.norm(pointcloud - agent_pos, axis=1)
            min_distances.append(np.min(distances_to_points))
            timestamps.append(pose['timestamp'] - self.start_timestamp)
        
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
        visualizer = MetricsVisualizer(filepath)
    except FileNotFoundError as e:
        print(e)
        return

    visualizer.get_summary()

    if args.plot:
        visualizer.plot_agent_planning_time().show()
        visualizer.plot_best_agent_selection().show()
        visualizer.plot_distance_to_goal().show()
        visualizer.plot_distance_to_closest_obstacle().show()
        visualizer.plot_scene().show()

if __name__ == "__main__":
    main()
