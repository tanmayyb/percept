import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import subprocess
import os
import re
import time
from collections import deque, defaultdict
import numpy as np
from datetime import datetime

"""
Launch chrome with the following flag to avoid webgl context issues:
--max-active-webgl-contexts=32
"""

# --- Configuration ---
APP_TITLE = "Cost Monitor"
DEFAULT_ROS_NAMESPACE = "/oriented_pointmass/"
COST_COMPONENTS = [
    "trajectory_smoothness_cost",
    "path_length_cost",
    "goal_distance_cost",
    "obstacle_distance_cost",
    "cost" # Total cost topic
]
LOG_DIR = "results/cost_monitor"
UPDATE_INTERVAL_MS = 2000  # 2 seconds
MAX_DATA_POINTS = 200 # Max points to store and plot per topic
WEIGHT_SUGGESTION_POINTS = 100 # Number of recent points to use for weight suggestion
MA_WINDOW_OPTIONS = [10, 50, 100, 200]  # Available MA window sizes
DEFAULT_MA_WINDOW = 50  # Default MA window size

# --- Global State Management ---
# It's generally better to use dcc.Store for state in Dash, but for subprocesses
# which live outside the Dash context, a global dictionary is a practical approach.
processes = {} # Stores all running subprocesses, keyed by log_file_path
file_last_read_time = defaultdict(float) # To track when each log file was last read
is_paused = False  # Global state to track if updates are paused

# --- Helper Functions ---

def ensure_log_dir():
    """Creates the log directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)

def get_log_file_path(agent_id, component):
    """Constructs the full path for a log file."""
    return os.path.join(LOG_DIR, f"agent_{agent_id}_{component}.log")

def parse_ros_message(content):
    """Parses timestamp and float values from ROS2 topic echo output."""
    try:
        # Find timestamp and data pairs
        pattern = r"\[(.*?)\].*?data:\s*(-?\d+\.?\d*e?-?\d*)"
        matches = re.findall(pattern, content)
        return [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f"), float(val)) for ts, val in matches]
    except (ValueError, IndexError) as e:
        print(f"Error parsing message: {e}")
        return []

def start_ros_listener(agent_id, component, namespace):
    """Starts a 'ros2 topic echo' subprocess for a given topic."""
    log_path = get_log_file_path(agent_id, component)
    topic_name = f"{namespace}agent_{agent_id}/{component}"
    
    # Check if a process for this log file is already running and alive
    if log_path in processes and processes[log_path].poll() is None:
        print(f"Process already running for {topic_name}")
        return # Process is already running
        
    # Command to echo the topic with timestamps and write to the log file
    command = f"sh -c ': > {log_path} && ros2 topic echo {topic_name} --full-length | while read -r line; do echo \"[$(date +\"%Y-%m-%d %H:%M:%S.%N\" | cut -b1-23)] $line\"; done >> {log_path}'"
    print(f"Starting command: {command}")

    try:
        # Start the subprocess
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes[log_path] = proc
        print(f"Started listener for {topic_name}")
        
        # Check if process started successfully
        if proc.poll() is not None:
            print(f"ERROR: Process failed to start for {topic_name}")
    except Exception as e:
        print(f"Error starting listener for {topic_name}: {e}")

def stop_all_listeners():
    """Terminates all running subprocesses."""
    for log_path, proc in processes.items():
        if proc.poll() is None: # If the process is still running
            try:
                proc.terminate()
                proc.wait(timeout=1) # Wait for the process to terminate
                print(f"Stopped listener for {log_path}")
            except subprocess.TimeoutExpired:
                proc.kill() # Force kill if it doesn't terminate gracefully
                print(f"Force-killed listener for {log_path}")
            except Exception as e:
                print(f"Error stopping listener for {log_path}: {e}")
    processes.clear()

def reset_logs():
    """Clears all log files and restarts listeners."""
    stop_all_listeners()
    if os.path.exists(LOG_DIR):
        # Only delete the log files that this app creates
        for agent_id in range(1, 100):  # Reasonable upper limit for agent IDs
            for component in COST_COMPONENTS:
                log_path = get_log_file_path(agent_id, component)
                if os.path.exists(log_path):
                    os.remove(log_path)
    ensure_log_dir()
    # The main callback loop will automatically restart the listeners.
    print("All logs cleared and listeners stopped. They will restart on the next interval.")

def calculate_moving_average(data, window_size=10):
    """Calculate moving average of a data series."""
    if len(data) < window_size:
        return None
    return np.mean(data[-window_size:])

# --- App Initialization ---
ensure_log_dir()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = APP_TITLE

# --- App Layout ---
app.layout = html.Div([
    # dcc.Store components to hold state across callbacks
    dcc.Store(id='n-agents-store', data=1), # Start with 1 agent
    dcc.Store(id='data-store', data={}), # Holds all time-series data

    # Header section
    html.Div([
        html.H1(APP_TITLE.lower().replace(' ', '_'), style={'color': 'white', 'margin': '0'}),
        html.Div([
            dbc.Input(id='namespace-input', value=DEFAULT_ROS_NAMESPACE, type='text', style={'width': '300px', 'marginRight': '15px'}),
            html.Div([
                html.Label("MA Window:", style={'color': 'white', 'marginRight': '10px'}),
                html.Div(
                    dcc.Slider(
                        id='ma-window-slider',
                        min=0,
                        max=len(MA_WINDOW_OPTIONS) - 1,
                        step=1,
                        value=MA_WINDOW_OPTIONS.index(DEFAULT_MA_WINDOW),
                        marks={i: str(size) for i, size in enumerate(MA_WINDOW_OPTIONS)},
                        included=False
                    ),
                    style={'width': '200px', 'marginRight': '15px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '15px'}),
            dbc.Button("+ Add Agent", id="add-agent-btn", color="success", className="me-1"),
            dbc.Button("- Remove Agent", id="remove-agent-btn", color="warning", className="me-1"),
            dbc.Button("Reset Data", id="reset-btn", color="danger", className="me-1"),
            dbc.Button("⏸ Pause", id="pause-btn", color="info", className="me-1"),
            dbc.Button("▶ Play", id="play-btn", color="info", className="me-1", style={'display': 'none'}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'backgroundColor': 'orange', 'padding': '10px', 'display': 'flex',
        'justifyContent': 'space-between', 'alignItems': 'center'
    }),

    # Main content area for agent sections
    html.Div([
        # Total costs card
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(id='total-costs-plot', style={'height': '300px'}),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='trajectory-smoothness-plot', style={'height': '250px'}),
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='path-length-plot', style={'height': '250px'}),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='goal-distance-plot', style={'height': '250px'}),
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='obstacle-distance-plot', style={'height': '250px'}),
                    ], width=6),
                ]),
            ], style={'padding': '10px'}),
            className="mb-3"
        ),
        # Agent sections container
        html.Div(id='agents-container')
    ]),
    
    # Interval component to trigger updates
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_MS, n_intervals=0)
])

# --- Callbacks ---

@app.callback(
    Output('n-agents-store', 'data'),
    Input('add-agent-btn', 'n_clicks'),
    Input('remove-agent-btn', 'n_clicks'),
    State('n-agents-store', 'data'),
    prevent_initial_call=True
)
def update_agent_count(add_clicks, remove_clicks, n_agents):
    """Manages adding and removing agents."""
    ctx = callback_context
    if not ctx.triggered:
        return n_agents

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'add-agent-btn':
        return n_agents + 1
    if button_id == 'remove-agent-btn':
        return max(1, n_agents - 1) # Ensure at least one agent remains
    return n_agents


@app.callback(
    Output('agents-container', 'children'),
    Input('n-agents-store', 'data')
)
def generate_agent_sections(n_agents):
    """Dynamically generates the UI sections for each agent."""
    # Create a row to hold all agent cards horizontally
    agent_cards = []
    for i in range(1, n_agents + 1):
        agent_id = i
        section = dbc.Card(
            dbc.CardBody([
                html.H2(f"agent_{agent_id}", style={'marginBottom': '10px'}),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Component", style={'textAlign': 'left', 'paddingLeft': '10px'}),
                            html.Th("Latest", style={'textAlign': 'left'}),
                            html.Th("MA", style={'textAlign': 'left'})
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td("total_cost", style={'textAlign': 'left', 'paddingLeft': '10px'}),
                                html.Td(html.Div(id={'type': 'latest-cost-total', 'index': agent_id}, style={'textAlign': 'left', 'fontSize': '0.8em', 'color': 'gray'})),
                                html.Td(html.Div(id={'type': 'ma-cost-total', 'index': agent_id}, style={'textAlign': 'left'})),
                            ]),
                            html.Tr([
                                html.Td("trajectory_smoothness", style={'textAlign': 'left', 'paddingLeft': '10px'}),
                                html.Td(html.Div(id={'type': 'latest-cost', 'index': f"{agent_id}-trajectory_smoothness_cost"}, style={'textAlign': 'left', 'fontSize': '0.8em', 'color': 'gray'})),
                                html.Td(html.Div(id={'type': 'ma-cost', 'index': f"{agent_id}-trajectory_smoothness_cost"}, style={'textAlign': 'left'})),
                            ]),
                            html.Tr([
                                html.Td("path_length", style={'textAlign': 'left', 'paddingLeft': '10px'}),
                                html.Td(html.Div(id={'type': 'latest-cost', 'index': f"{agent_id}-path_length_cost"}, style={'textAlign': 'left', 'fontSize': '0.8em', 'color': 'gray'})),
                                html.Td(html.Div(id={'type': 'ma-cost', 'index': f"{agent_id}-path_length_cost"}, style={'textAlign': 'left'})),
                            ]),
                            html.Tr([
                                html.Td("goal_distance", style={'textAlign': 'left', 'paddingLeft': '10px'}),
                                html.Td(html.Div(id={'type': 'latest-cost', 'index': f"{agent_id}-goal_distance_cost"}, style={'textAlign': 'left', 'fontSize': '0.8em', 'color': 'gray'})),
                                html.Td(html.Div(id={'type': 'ma-cost', 'index': f"{agent_id}-goal_distance_cost"}, style={'textAlign': 'left'})),
                            ]),
                            html.Tr([
                                html.Td("obstacle_distance", style={'textAlign': 'left', 'paddingLeft': '10px'}),
                                html.Td(html.Div(id={'type': 'latest-cost', 'index': f"{agent_id}-obstacle_distance_cost"}, style={'textAlign': 'left', 'fontSize': '0.8em', 'color': 'gray'})),
                                html.Td(html.Div(id={'type': 'ma-cost', 'index': f"{agent_id}-obstacle_distance_cost"}, style={'textAlign': 'left'})),
                            ])
                        ])
                    ], style={'width': '100%', 'marginBottom': '10px'}),
                    dbc.Button("suggest_weights", id={'type': 'suggest-weights-btn', 'index': agent_id}, color="primary", className="mt-1", size="sm"),
                    # Collapsible section for weight suggestions
                    dbc.Collapse(
                        html.Div(id={'type': 'weights-content', 'index': agent_id}),
                        id={'type': 'weights-collapse', 'index': agent_id},
                        is_open=False,
                        style={'marginTop': '10px', 'maxHeight': '300px', 'overflowY': 'auto'}
                    ),
                ], style={'padding': '5px', 'height': '100%'}),
            ], style={'padding': '10px'}),
            className="mb-3",
            style={'width': '450px', 'marginRight': '10px'}  # Increased width from 300px to 400px
        )
        agent_cards.append(section)
    
    # Create a row with horizontal scrolling
    return html.Div(
        dbc.Row(
            [dbc.Col(card, width='auto') for card in agent_cards],
            style={'flexWrap': 'nowrap', 'overflowX': 'auto', 'margin': '0'}
        ),
        style={'width': '100%', 'overflowX': 'auto'}
    )


@app.callback(
    Output('data-store', 'data', allow_duplicate=True),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_reset(n_clicks):
    """Resets all logs and data."""
    reset_logs()
    # Clear the data store
    return {}

@app.callback(
    [
        Output('total-costs-plot', 'figure'),
        Output('trajectory-smoothness-plot', 'figure'),
        Output('path-length-plot', 'figure'),
        Output('goal-distance-plot', 'figure'),
        Output('obstacle-distance-plot', 'figure'),
        Output({'type': 'latest-cost', 'index': dash.ALL}, 'children'),
        Output({'type': 'latest-cost-total', 'index': dash.ALL}, 'children'),
        Output({'type': 'ma-cost', 'index': dash.ALL}, 'children'),
        Output({'type': 'ma-cost-total', 'index': dash.ALL}, 'children'),
        Output({'type': 'latest-costs-label', 'index': dash.ALL}, 'children'),
        Output('data-store', 'data')
    ],
    [Input('interval-component', 'n_intervals'),
     Input('ma-window-slider', 'value')],
    State('n-agents-store', 'data'),
    State('namespace-input', 'value'),
    State('data-store', 'data')
)
def update_data_and_plots(n, ma_window_index, n_agents, namespace, stored_data):
    """The main callback to read data, update plots, and manage subprocesses."""
    global is_paused
    
    if is_paused:
        raise dash.exceptions.PreventUpdate
    
    print(f"\nUpdate triggered - Interval: {n}, Agents: {n_agents}")
    
    try:
        ma_window_index = int(ma_window_index) if ma_window_index is not None else MA_WINDOW_OPTIONS.index(DEFAULT_MA_WINDOW)
        ma_window_index = max(0, min(ma_window_index, len(MA_WINDOW_OPTIONS) - 1))
        ma_window = MA_WINDOW_OPTIONS[ma_window_index]
    except (ValueError, TypeError, IndexError):
        print("Invalid MA window index, using default")
        ma_window = DEFAULT_MA_WINDOW
    
    if stored_data is None:
        stored_data = {}
        print("Initialized empty data store")

    # Start/Check ROS Listeners
    for i in range(1, n_agents + 1):
        for component in COST_COMPONENTS:
            start_ros_listener(i, component, namespace)

    # Read new data from logs
    for i in range(1, n_agents + 1):
        for component in COST_COMPONENTS:
            log_path = get_log_file_path(i, component)
            if os.path.exists(log_path):
                mod_time = os.path.getmtime(log_path)
                if mod_time > file_last_read_time[log_path]:
                    print(f"Reading new data from {log_path}")
                    with open(log_path, 'r') as f:
                        content = f.read()
                    
                    new_data = parse_ros_message(content)
                    print(f"Parsed {len(new_data)} values from {component}")
                    
                    if log_path not in stored_data:
                        stored_data[log_path] = deque(maxlen=MAX_DATA_POINTS)
                    
                    stored_data[log_path].clear()
                    stored_data[log_path].extend(new_data)
                    
                    file_last_read_time[log_path] = mod_time
                else:
                    print(f"No new data in {log_path}")
            else:
                print(f"Log file not found: {log_path}")

    # Convert deques to lists for JSON serialization
    serializable_data = {}
    for k, v in stored_data.items():
        try:
            serializable_data[k] = [(ts.isoformat(), val) for ts, val in v]
        except AttributeError:
            # If timestamps are already strings, use them as is
            serializable_data[k] = [(ts, val) for ts, val in v]
        except Exception as e:
            print(f"Error serializing data for {k}: {e}")
            serializable_data[k] = []

    # Prepare outputs for Dash
    ctx = callback_context
    latest_cost_outputs = []
    total_cost_outputs = []
    ma_cost_outputs = []
    ma_total_cost_outputs = []

    latest_cost_ids = [item['id'] for item in ctx.outputs_list[5]]
    total_cost_ids = [item['id'] for item in ctx.outputs_list[6]]
    ma_cost_ids = [item['id'] for item in ctx.outputs_list[7]]
    ma_total_cost_ids = [item['id'] for item in ctx.outputs_list[8]]
    latest_costs_label_ids = [item['id'] for item in ctx.outputs_list[9]]

    latest_cost_map = {f"{id['index']}": "-" for id in latest_cost_ids}
    total_cost_map = {id['index']: "-" for id in total_cost_ids}
    ma_cost_map = {f"{id['index']}": "-" for id in ma_cost_ids}
    ma_total_cost_map = {id['index']: "-" for id in ma_total_cost_ids}

    # Create the total costs plot with all agents
    total_costs_fig = go.Figure()
    max_points = 0
    for i in range(1, n_agents + 1):
        log_path = get_log_file_path(i, 'cost')
        data = list(stored_data.get(log_path, []))
        if data:
            timestamps, values = zip(*data)
            total_costs_fig.add_trace(go.Scattergl(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=f'Agent {i}'
            ))
            max_points = max(max_points, len(data))
            
            # Update total cost display
            latest_total = values[-1]
            total_cost_map[i] = f"{latest_total:.2e}"
            
            # Calculate and display moving average for total cost
            ma = calculate_moving_average(values, ma_window)
            if ma is not None:
                ma_total_cost_map[i] = f"{ma:.2e}"

    total_costs_fig.update_layout(
        title=dict(
            text="Total Costs",
            y=0.95
        ),
        xaxis_title="Time",
        yaxis_title="Cost",
        annotations=[dict(
            text=f"Number of data points: {max_points}",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.9,
            font=dict(size=12)
        )],
        template='plotly_white',
        margin=dict(l=40, r=20, t=60, b=20),
        height=300,
        title_font_size=14,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Create component plots with all agents
    component_plots = {}
    for component in ['trajectory_smoothness_cost', 'path_length_cost', 'goal_distance_cost', 'obstacle_distance_cost']:
        fig = go.Figure()
        for i in range(1, n_agents + 1):
            log_path = get_log_file_path(i, component)
            data = list(stored_data.get(log_path, []))
            if data:
                timestamps, values = zip(*data)
                fig.add_trace(go.Scattergl(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=f'Agent {i}'
                ))
                
                # Update latest cost display
                latest_val = values[-1]
                latest_cost_map[f"{i}-{component}"] = f"{latest_val:.2e}"
                
                # Calculate and display moving average
                ma = calculate_moving_average(values, ma_window)
                if ma is not None:
                    ma_cost_map[f"{i}-{component}"] = f"{ma:.2e}"

        fig.update_layout(
            title=dict(
                text=component.replace('_', ' ').title(),
                y=0.95
            ),
            xaxis_title="Time",
            yaxis_title="Cost",
            template='plotly_white',
            margin=dict(l=40, r=20, t=40, b=20),
            height=250,
            title_font_size=12,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        component_plots[component] = fig

    # Convert maps to lists in the correct order
    latest_cost_outputs = [latest_cost_map[f"{id['index']}"] for id in latest_cost_ids]
    total_cost_outputs = [total_cost_map[id['index']] for id in total_cost_ids]
    ma_cost_outputs = [ma_cost_map[f"{id['index']}"] for id in ma_cost_ids]
    ma_total_cost_outputs = [ma_total_cost_map[id['index']] for id in ma_total_cost_ids]
    latest_costs_labels = ["latest_costs"] * len(latest_costs_label_ids)

    return (
        total_costs_fig,
        component_plots['trajectory_smoothness_cost'],
        component_plots['path_length_cost'],
        component_plots['goal_distance_cost'],
        component_plots['obstacle_distance_cost'],
        latest_cost_outputs,
        total_cost_outputs,
        ma_cost_outputs,
        ma_total_cost_outputs,
        latest_costs_labels,
        serializable_data
    )


@app.callback(
    [Output({'type': 'weights-collapse', 'index': dash.ALL}, 'is_open'),
     Output({'type': 'weights-content', 'index': dash.ALL}, 'children')],
    [Input({'type': 'suggest-weights-btn', 'index': dash.ALL}, 'n_clicks'),
     Input('ma-window-slider', 'value')],
    State('data-store', 'data'),
    prevent_initial_call=True
)
def suggest_weights(n_clicks, ma_window_index, stored_data):
    """Calculates and displays suggested weights in a collapsible section."""
    ctx = callback_context
    if not any(n_clicks) or not ctx.triggered:
        return [False] * len(n_clicks), [""] * len(n_clicks)

    # Get the current MA window size
    try:
        ma_window_index = int(ma_window_index) if ma_window_index is not None else MA_WINDOW_OPTIONS.index(DEFAULT_MA_WINDOW)
        ma_window_index = max(0, min(ma_window_index, len(MA_WINDOW_OPTIONS) - 1))
        ma_window = MA_WINDOW_OPTIONS[ma_window_index]
    except (ValueError, TypeError, IndexError):
        print("Invalid MA window index, using default")
        ma_window = DEFAULT_MA_WINDOW

    button_id = ctx.triggered[0]['prop_id']
    pattern = re.compile(r'{"index":(\d+),"type":"suggest-weights-btn"}')
    match = pattern.search(button_id)
    if not match:
        return [False] * len(n_clicks), [""] * len(n_clicks)
    
    agent_id = int(match.group(1))
    
    # These are the components used for weight calculation
    components_for_weights = [c for c in COST_COMPONENTS if c != 'cost']
    
    averages = {}
    all_data_present = True
    for component in components_for_weights:
        log_path = get_log_file_path(agent_id, component)
        data = list(stored_data.get(log_path, []))
        
        if len(data) < ma_window: # Not enough data for MA calculation
            all_data_present = False
            break
        
        # Extract just the numerical values from the data
        values = [val for _, val in data]
        
        # Calculate moving average
        ma = calculate_moving_average(values, ma_window)
        if ma is None:
            all_data_present = False
            break
            
        averages[component] = abs(ma)

    if not all_data_present:
        return [False] * len(n_clicks), [""] * len(n_clicks)

    # Calculate weights (reciprocal of average, handling zeros)
    weights = {}
    for component, avg in averages.items():
        weights[component] = 1 / (avg + 1e-9) # Add epsilon to avoid division by zero

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {comp: w / total_weight for comp, w in weights.items()}

    # Create content
    content = [
        html.P(f"Based on {ma_window}-point moving average:", style={'marginBottom': '5px', 'fontSize': '0.9em'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Component", style={'textAlign': 'right', 'paddingRight': '10px', 'fontSize': '0.8em'}),
                html.Th("Weight", style={'textAlign': 'right', 'fontSize': '0.8em'})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(comp.replace('_', ' ').title(), style={'textAlign': 'right', 'paddingRight': '10px', 'fontSize': '0.8em'}),
                    html.Td(f"{weight:.2e}", style={'textAlign': 'right', 'fontSize': '0.8em'})
                ]) for comp, weight in normalized_weights.items()
            ])
        ], className="table table-sm", style={'marginBottom': '0'})
    ]

    # Create list of outputs for all agents
    is_open_list = [False] * len(n_clicks)
    content_list = [""] * len(n_clicks)
    
    # Set the values for the triggered agent
    for i, n in enumerate(n_clicks):
        if n is not None:
            is_open_list[i] = True
            content_list[i] = content

    return is_open_list, content_list

@app.callback(
    [Output('pause-btn', 'style'),
     Output('play-btn', 'style')],
    [Input('pause-btn', 'n_clicks'),
     Input('play-btn', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_pause_play(pause_clicks, play_clicks):
    """Toggles between pause and play states."""
    global is_paused
    ctx = callback_context
    if not ctx.triggered:
        return {'display': 'inline-block'}, {'display': 'none'}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'pause-btn':
        is_paused = True
        return {'display': 'none'}, {'display': 'inline-block'}
    else:  # play-btn
        is_paused = False
        return {'display': 'inline-block'}, {'display': 'none'}

# --- Main Execution ---
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8050, debug=True)
    finally:
        # Ensure all subprocesses are cleaned up when the app is closed
        print("\nShutting down... stopping all ROS listeners.")
        stop_all_listeners()
