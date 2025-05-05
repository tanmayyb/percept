import os
import sys
# sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('/home/dev/percept/eval/'))


from process_results import MetricsReader
import yaml
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from pathlib import Path
pio.templates.default = "plotly_white"
pio.renderers.default = "browser"

plan_name = '250430_tuning_r1'
root_dir = Path('eval')
results_dir = root_dir / 'results' / plan_name
plan_dir = root_dir / 'plans' / plan_name


filepaths = sorted(list(results_dir.glob('*.result')))[:]
results = {}
results_str = {}
for filepath in filepaths:
	try:
		reader = MetricsReader(
			filepath, 
			success_time_limit=60.0, 
			acceptable_min_distances_to_target=0.10
		)
		results[reader.info['label']] = reader
		results_str[reader.info['label']] = reader.format_stats()
	except Exception as e:
		print(f"Error reading {filepath}: {e}")

costs = {}
index = {}
for i, (label, result) in enumerate(results.items()):
	costs[label] = result.agent_costs.groupby("agent_id")["cost"].mean().iloc[0]
	index[label] = i

costs = pd.Series(costs)
index = pd.Series(index)

fig = go.Figure(data=go.Scatter(
	x=index,
	y=costs,
	mode='lines+markers',
	name='Cost',
	marker=dict(size=6, color='red'),
	line=dict(color='red')
))

fig.update_layout(
	title='Cost',
	xaxis_title='Index',
	yaxis_title='Cost'
)

fig.show()


k_force = {}
index = {}
for i, (label, result) in enumerate(results.items()):
	k_force[label] = result.workload_config['planner_config']['agents'][0]['force_configs']['apf_heuristic_force']['k_force']
	index[label] = i
k_force = pd.Series(k_force)

k_force_sorted = k_force.sort_values()
costs_sorted = costs.reindex(k_force_sorted.index)

fig = go.Figure(data=go.Scatter(
	x=k_force_sorted,
	y=costs_sorted,
	mode='lines+markers',
	name='K_force',
	marker=dict(size=6, color='red'),
	line=dict(color='red')
))
fig.update_xaxes(type="log")

fig.update_layout(
	title='K_force',
	xaxis_title='K_force',
	yaxis_title='Cost'
)

fig.show()