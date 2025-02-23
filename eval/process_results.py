#!/usr/bin/env python3
import argparse
import yaml
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

pio.templates.default = "plotly_white"
pio.renderers.default = "browser"

@dataclass
class MetricsData:
    filename: str
    log: List[Dict]
    workload_config: dict
    agent_poses: List[Dict]
    target_poses: List[Dict]
    pointclouds: List[Dict]
    start_timestamp: float
    end_timestamp: float
    elapsed_time: float
    start_pos: List[float]
    goal_pos: List[float]
    start_orientation: List[float]
    goal_orientation: List[float]
    agent_planning_time_list: List[Dict]
    best_agent_list: List[Dict]

    @staticmethod
    def _parse_best_agent_record(record: Dict):
        # Extracts integer agent id from best_agent_name field
        return {
            'timestamp': record['timestamp'],
            'agent_id': int(record['best_agent_name'].split('/')[-1].replace('agent_', '')),
            'type': 'best_agent_name'
        }
    
    @classmethod
    def from_yaml(cls, filepath: Path):
        with open(filepath, 'r') as file:
            documents = list(yaml.safe_load_all(file))

        agent_poses = []
        target_poses = []
        pointclouds = []
        best_agent_list = []
        agent_planning_time_list = []
        start_timestamp = float('inf')
        end_timestamp = float('-inf')
        
        for record in documents[1:]:
            if record is not None:
                if 'timestamp' in record:
                    start_timestamp = min(start_timestamp, record['timestamp'])
                    end_timestamp = max(end_timestamp, record['timestamp'])
                    
                record_type = record.get('type', '')
                if record_type == 'agent_pose':
                    agent_poses.append(record)
                elif record_type == 'target_pose':
                    target_poses.append(record)
                elif record_type == 'pointcloud':
                    pointclouds.append(record)
                elif record_type == 'best_agent_name':
                    best_agent_list.append(cls._parse_best_agent_record(record))
                elif record_type == 'agent_planning_time':
                    agent_planning_time_list.append(record)
        
        elapsed_time = end_timestamp - start_timestamp
        workload_config = documents[0]['workload_config']
        poses = workload_config['planner_config']['poses']
        
        return cls(
            filename=filepath.name,
            log=documents,
            workload_config=workload_config,
            agent_poses=agent_poses,
            target_poses=target_poses,
            pointclouds=pointclouds,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            elapsed_time=elapsed_time,
            start_pos=poses['start_pos'],
            goal_pos=poses['goal_pos'],
            start_orientation=poses['start_orientation'],
            goal_orientation=poses['goal_orientation'],
            agent_planning_time_list=agent_planning_time_list,
            best_agent_list=best_agent_list
        )
    
    def get_summary(self):
        print('Elapsed Time:', self.elapsed_time)
        print('Start Position:', self.start_pos)
        print('Goal Position:', self.goal_pos)
        print('Start Orientation:', self.start_orientation)
        print('Goal Orientation:', self.goal_orientation)


@dataclass
class SceneVisualizer:
    filename: str
    agent_trace: go.Scatter3d
    target_trace: go.Scatter3d
    pointcloud_trace: go.Scatter3d
    fig: go.Figure

    @classmethod
    def from_metrics_data(cls, metrics_data: MetricsData):
        filename = metrics_data.filename
        
        # Build agent path coordinates
        x_coords = [pose['position']['x'] for pose in metrics_data.agent_poses]
        y_coords = [pose['position']['y'] for pose in metrics_data.agent_poses]
        z_coords = [pose['position']['z'] for pose in metrics_data.agent_poses]

        # Extract pointcloud coordinates (assuming one pointcloud record)
        points = metrics_data.pointclouds[0]['points']
        pointcloud_x, pointcloud_y, pointcloud_z = zip(*points)

        agent_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=z_coords,
                colorscale='Viridis',
                showscale=True
            ),
            line=dict(color='blue', width=2),
            name='Agent Path'
        )
        pointcloud_trace = go.Scatter3d(
            x=pointcloud_x,
            y=pointcloud_y,
            z=pointcloud_z,
            mode='markers',
            marker=dict(size=6, color='red', symbol='circle'),
            name='Pointcloud'
        )
        
        fig = go.Figure(data=[agent_trace, pointcloud_trace])
        fig.update_layout(
            title='Agent Path and Pointcloud Visualization: ' + filename,
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                aspectmode='data'
            ),
            width=800,
            height=800
        )
        
        return cls(
            filename=filename,
            agent_trace=agent_trace,
            target_trace=None,  # Not used in current implementation
            pointcloud_trace=pointcloud_trace,
            fig=fig
        )
    
    def show(self):
        self.fig.show()


def plot_agent_planning_time(metrics_data: MetricsData) -> go.Figure:
    df = pd.DataFrame(metrics_data.agent_planning_time_list)
    fig = go.Figure()
    for agent_id in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent_id]
        fig.add_trace(go.Scattergl(
            x=agent_data['timestamp'] - metrics_data.start_timestamp,
            y=agent_data['planning_time'],
            mode='lines+markers',
            name=f'Agent {agent_id}',
            line=dict(shape='hv'),
            marker=dict(size=4, opacity=0.6)
        ))
    fig.update_layout(
        title='Agent Planning Time: ' + metrics_data.filename,
        xaxis_title='Time (seconds)',
        yaxis_title='Planning Time (seconds)',
        yaxis=dict(range=[0.0, 0.10]),
        width=800,
        height=500
    )
    return fig


def plot_best_agent_selection(metrics_data: MetricsData) -> go.Figure:
    df = pd.DataFrame(metrics_data.best_agent_list)
    # Map agent IDs to heuristic strings from workload config
    heuristic_mapping = {
        agent_id + 1: f"{agent_id + 1}: {list(agent['forces'].keys())[1].split('_')[0]}"
        for agent_id, agent in enumerate(metrics_data.workload_config['planner_config']['agents'])
    }
    df['heuristic'] = df['agent_id'].map(heuristic_mapping)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'] - metrics_data.start_timestamp,
        y=df['heuristic'],
        mode='lines+markers',
        name='Best Agent ID',
        line=dict(shape='hv', color='blue', width=0.5),
        marker=dict(size=2.5, color='blue')
    ))
    fig.update_layout(
        title='Best Agent Selection Over Time: ' + metrics_data.filename,
        xaxis_title='Time (seconds)',
        yaxis=dict(
            title='Agent ID',
            tickmode='linear',
            dtick=1,
            ticks='outside',
            ticklen=5
        ),
        width=800,
        height=500,
        showlegend=True,
        legend=dict(x=1.1, y=1)
    )
    return fig


def plot_distance_to_goal(metrics_data: MetricsData) -> go.Figure:
    # Vectorize agent positions and calculate distances to the goal
    agent_positions = np.array([
        [pose['position']['x'], pose['position']['y'], pose['position']['z']]
        for pose in metrics_data.agent_poses
    ])
    goal_pos = np.array(metrics_data.goal_pos)
    distances = np.linalg.norm(agent_positions - goal_pos, axis=1)
    timestamps = np.array([pose['timestamp'] for pose in metrics_data.agent_poses]) - metrics_data.start_timestamp

    fig = go.Figure(data=go.Scatter(
        x=timestamps,
        y=distances,
        mode='lines+markers',
        name='Distance to Goal',
        marker=dict(size=6, color='blue'),
        line=dict(color='blue')
    ))
    fig.update_layout(
        title='Distance to Goal Over Time: ' + metrics_data.filename,
        xaxis_title='Time (seconds)',
        yaxis_title='Distance to Goal (meters)',
        width=800,
        height=500
    )
    return fig


def plot_distance_to_closest_obstacle(metrics_data: MetricsData) -> go.Figure:
    pointcloud = np.array(metrics_data.pointclouds[0]['points'])
    timestamps = []
    min_distances = []
    for pose in metrics_data.agent_poses:
        agent_pos = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
        distances_to_points = np.linalg.norm(pointcloud - agent_pos, axis=1)
        min_distances.append(np.min(distances_to_points))
        timestamps.append(pose['timestamp'] - metrics_data.start_timestamp)
    
    fig = go.Figure(data=go.Scatter(
        x=timestamps,
        y=min_distances,
        mode='lines+markers',
        name='Distance to Closest Obstacle',
        marker=dict(size=6, color='red'),
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Distance to Closest Obstacle Over Time: ' + metrics_data.filename,
        xaxis_title='Time (seconds)',
        yaxis_title='Distance to Closest Point (meters)',
        width=800,
        height=500
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
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' does not exist!")
        return
    
    # Load and summarize metrics data
    metrics_data = MetricsData.from_yaml(filepath)
    metrics_data.get_summary()
    
    # Create scene visualizer instance
    scene_visualizer = SceneVisualizer.from_metrics_data(metrics_data)
    
    # If the plot flag is set, show each figure
    if args.plot:
        plot_agent_planning_time(metrics_data).show()
        plot_best_agent_selection(metrics_data).show()
        plot_distance_to_goal(metrics_data).show()
        plot_distance_to_closest_obstacle(metrics_data).show()
        scene_visualizer.show()


if __name__ == "__main__":
    main()


