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
import io
import base64
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

"""
Launch chrome with the following flag to avoid webgl context issues:
--max-active-webgl-contexts=32
Run this program using:
python cost_monitor.py --port 8050
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
BEST_AGENT_TOPIC = "best_agent_name"  # New topic for best agent
LOG_DIR = "results/cost_monitor"
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
UPDATE_INTERVAL_MS = 2000  # 2 seconds
MAX_DATA_POINTS = 200 # Max points to store and plot per topic
WEIGHT_SUGGESTION_POINTS = 100 # Number of recent points to use for weight suggestion
MA_WINDOW_OPTIONS = [10, 50, 100, 200]  # Available MA window sizes
DEFAULT_MA_WINDOW = 50  # Default MA window size

# --- Logging Setup ---
def setup_logger():
    """Sets up the application logger."""
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Clear the log file if it exists
    if os.path.exists(APP_LOG_FILE):
        open(APP_LOG_FILE, 'w').close()
    
    # Create logger
    logger = logging.getLogger('cost_monitor')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        APP_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()
logger.info("Cost Monitor application starting")

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
    logger.debug("Ensured log directory exists")

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
        logger.error(f"Error parsing message: {e}")
        return []

def parse_best_agent_message(content):
    """Parses timestamp and agent name from ROS2 topic echo output."""
    try:
        # Find timestamp and data pairs
        pattern = r"\[(.*?)\].*?data:\s*([^\s]+)"
        matches = re.findall(pattern, content)
        # Strip 'cf_planner/' prefix from agent names
        return [(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f"), str(val).replace('cf_planner/', '')) for ts, val in matches]
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing best agent message: {e}")
        return []

def get_best_agent_log_file_path():
    """Constructs the full path for the best agent log file."""
    return os.path.join(LOG_DIR, f"best_agent.log")

def start_ros_listener(agent_id, component, namespace):
    """Starts a 'ros2 topic echo' subprocess for a given topic."""
    log_path = get_log_file_path(agent_id, component)
    topic_name = f"{namespace}agent_{agent_id}/{component}"
    
    # Check if a process for this log file is already running and alive
    if log_path in processes and processes[log_path].poll() is None:
        logger.debug(f"Process already running for {topic_name}")
        return # Process is already running
        
    # Command to echo the topic with timestamps and write to the log file
    command = f"sh -c ': > {log_path} && ros2 topic echo {topic_name} --full-length | while read -r line; do echo \"[$(date +\"%Y-%m-%d %H:%M:%S.%N\" | cut -b1-23)] $line\"; done >> {log_path}'"
    logger.debug(f"Starting command: {command}")

    try:
        # Start the subprocess
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes[log_path] = proc
        logger.info(f"Started listener for {topic_name}")
        
        # Check if process started successfully
        if proc.poll() is not None:
            logger.error(f"Process failed to start for {topic_name}")
    except Exception as e:
        logger.error(f"Error starting listener for {topic_name}: {e}")

def start_best_agent_listener(namespace):
    """Starts a 'ros2 topic echo' subprocess for the best agent topic."""
    log_path = get_best_agent_log_file_path()
    topic_name = f"{namespace}{BEST_AGENT_TOPIC}"
    
    # Check if a process for this log file is already running and alive
    if log_path in processes and processes[log_path].poll() is None:
        logger.debug(f"Process already running for {topic_name}")
        return # Process is already running
        
    # Command to echo the topic with timestamps and write to the log file
    command = f"sh -c ': > {log_path} && ros2 topic echo {topic_name} --full-length | while read -r line; do echo \"[$(date +\"%Y-%m-%d %H:%M:%S.%N\" | cut -b1-23)] $line\"; done >> {log_path}'"
    logger.debug(f"Starting command: {command}")

    try:
        # Start the subprocess
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes[log_path] = proc
        logger.info(f"Started listener for {topic_name}")
        
        # Check if process started successfully
        if proc.poll() is not None:
            logger.error(f"Process failed to start for {topic_name}")
    except Exception as e:
        logger.error(f"Error starting listener for {topic_name}: {e}")

def stop_all_listeners():
    """Terminates all running subprocesses."""
    for log_path, proc in processes.items():
        if proc.poll() is None: # If the process is still running
            try:
                proc.terminate()
                proc.wait(timeout=1) # Wait for the process to terminate
                logger.info(f"Stopped listener for {log_path}")
            except subprocess.TimeoutExpired:
                proc.kill() # Force kill if it doesn't terminate gracefully
                logger.warning(f"Force-killed listener for {log_path}")
            except Exception as e:
                logger.error(f"Error stopping listener for {log_path}: {e}")
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
    logger.info("All logs cleared and listeners stopped. They will restart on the next interval.")

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
    dcc.Download(id='download-dataframe'), # For downloading the exported data

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
            dbc.Button("📥 Export Data", id="export-btn", color="primary", className="me-1"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    '📂 Upload NPZ',
                ]),
                style={
                    'width': '100px',
                    'height': '36px',
                    'lineHeight': '36px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '0px',
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'cursor': 'pointer'
                },
                multiple=False
            ),
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
                dcc.Graph(id='best-agent-plot', style={'height': '200px'}),  # New best agent plot
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
    logger.info("Resetting all data")
    reset_logs()
    # Clear the data store
    return {}

@app.callback(
    [
        Output('total-costs-plot', 'figure'),
        Output('best-agent-plot', 'figure'),  # New output
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
    
    logger.info(f"\nUpdate triggered - Interval: {n}, Agents: {n_agents}")
    
    # Calculate MA window size
    try:
        ma_window_index = int(ma_window_index) if ma_window_index is not None else MA_WINDOW_OPTIONS.index(DEFAULT_MA_WINDOW)
        ma_window_index = max(0, min(ma_window_index, len(MA_WINDOW_OPTIONS) - 1))
        ma_window = MA_WINDOW_OPTIONS[ma_window_index]
    except (ValueError, TypeError, IndexError):
        logger.info("Invalid MA window index, using default")
        ma_window = DEFAULT_MA_WINDOW
    
    if stored_data is None:
        stored_data = {}
        logger.info("Initialized empty data store")
    
    # Only check pause state for interval updates
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'interval-component.n_intervals':
        if is_paused:
            logger.info("Update prevented due to pause state")
            raise dash.exceptions.PreventUpdate
            
        # Start/Check ROS Listeners
        for i in range(1, n_agents + 1):
            for component in COST_COMPONENTS:
                start_ros_listener(i, component, namespace)
        
        # Start best agent listener
        start_best_agent_listener(namespace)

        # Read new data from logs
        for i in range(1, n_agents + 1):
            for component in COST_COMPONENTS:
                log_path = get_log_file_path(i, component)
                if os.path.exists(log_path):
                    mod_time = os.path.getmtime(log_path)
                    if mod_time > file_last_read_time[log_path]:
                        logger.info(f"Reading new data from {log_path}")
                        with open(log_path, 'r') as f:
                            content = f.read()
                        
                        new_data = parse_ros_message(content)
                        logger.info(f"Parsed {len(new_data)} values from {component}")
                        
                        if log_path not in stored_data:
                            stored_data[log_path] = deque(maxlen=MAX_DATA_POINTS)
                        
                        stored_data[log_path].clear()
                        stored_data[log_path].extend(new_data)
                        
                        file_last_read_time[log_path] = mod_time
                    else:
                        logger.info(f"No new data in {log_path}")
                else:
                    logger.info(f"Log file not found: {log_path}")
        
        # Read best agent data
        best_agent_log_path = get_best_agent_log_file_path()
        if os.path.exists(best_agent_log_path):
            mod_time = os.path.getmtime(best_agent_log_path)
            if mod_time > file_last_read_time[best_agent_log_path]:
                logger.info(f"Reading new data from {best_agent_log_path}")
                with open(best_agent_log_path, 'r') as f:
                    content = f.read()
                
                new_data = parse_best_agent_message(content)
                logger.info(f"Parsed {len(new_data)} values from best agent")
                
                if best_agent_log_path not in stored_data:
                    stored_data[best_agent_log_path] = deque(maxlen=MAX_DATA_POINTS)
                
                stored_data[best_agent_log_path].clear()
                stored_data[best_agent_log_path].extend(new_data)
                
                file_last_read_time[best_agent_log_path] = mod_time
            else:
                logger.info(f"No new data in {best_agent_log_path}")
        else:
            logger.info(f"Log file not found: {best_agent_log_path}")

    # Convert deques to lists for JSON serialization
    serializable_data = {}
    for k, v in stored_data.items():
        try:
            serializable_data[k] = [(ts.isoformat() if isinstance(ts, datetime) else ts, val) for ts, val in v]
        except AttributeError:
            # If timestamps are already strings, use them as is
            serializable_data[k] = [(ts, val) for ts, val in v]
        except Exception as e:
            logger.error(f"Error serializing data for {k}: {e}")
            serializable_data[k] = []

    # Prepare outputs for Dash
    ctx = callback_context
    latest_cost_outputs = []
    total_cost_outputs = []
    ma_cost_outputs = []
    ma_total_cost_outputs = []

    latest_cost_ids = [item['id'] for item in ctx.outputs_list[6]]
    total_cost_ids = [item['id'] for item in ctx.outputs_list[7]]
    ma_cost_ids = [item['id'] for item in ctx.outputs_list[8]]
    ma_total_cost_ids = [item['id'] for item in ctx.outputs_list[9]]
    latest_costs_label_ids = [item['id'] for item in ctx.outputs_list[10]]

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

    # Create best agent plot
    best_agent_fig = go.Figure()
    best_agent_log_path = get_best_agent_log_file_path()
    best_agent_data = list(stored_data.get(best_agent_log_path, []))
    if best_agent_data:
        timestamps, agent_names = zip(*best_agent_data)
        # Create a mapping of agent names to numeric values for plotting
        unique_agents = list(set(agent_names))
        agent_to_num = {agent: i for i, agent in enumerate(unique_agents)}
        numeric_values = [agent_to_num[agent] for agent in agent_names]
        
        # Create step-like visualization using hv lines
        for i in range(len(timestamps) - 1):
            # Add horizontal line
            best_agent_fig.add_trace(go.Scattergl(
                x=[timestamps[i], timestamps[i + 1]],
                y=[numeric_values[i], numeric_values[i]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
            # Add vertical line
            best_agent_fig.add_trace(go.Scattergl(
                x=[timestamps[i + 1], timestamps[i + 1]],
                y=[numeric_values[i], numeric_values[i + 1]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
        
        # Add markers at each point
        best_agent_fig.add_trace(go.Scattergl(
            x=timestamps,
            y=numeric_values,
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                symbol='circle'
            ),
            name='Best Agent'
        ))
        
        # Update y-axis to show agent names
        best_agent_fig.update_layout(
            yaxis=dict(
                ticktext=unique_agents,
                tickvals=list(range(len(unique_agents))),
                tickangle=0
            )
        )

    best_agent_fig.update_layout(
        title=dict(
            text="Current Best Agent",
            y=0.95
        ),
        xaxis_title="Time",
        yaxis_title="Agent",
        template='plotly_white',
        margin=dict(l=40, r=20, t=40, b=20),
        height=200,
        title_font_size=12,
        showlegend=False
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
        best_agent_fig,
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
    [State('data-store', 'data'),
     State({'type': 'weights-content', 'index': dash.ALL}, 'children'),
     State({'type': 'weights-collapse', 'index': dash.ALL}, 'is_open')],
    prevent_initial_call=True
)
def suggest_weights(n_clicks, ma_window_index, stored_data, current_contents, current_states):
    """Calculates and displays suggested weights in a collapsible section."""
    ctx = callback_context
    if not ctx.triggered:
        return current_states, current_contents

    # Get the current MA window size
    try:
        ma_window_index = int(ma_window_index) if ma_window_index is not None else MA_WINDOW_OPTIONS.index(DEFAULT_MA_WINDOW)
        ma_window_index = max(0, min(ma_window_index, len(MA_WINDOW_OPTIONS) - 1))
        ma_window = MA_WINDOW_OPTIONS[ma_window_index]
    except (ValueError, TypeError, IndexError):
        logger.info("Invalid MA window index, using default")
        ma_window = DEFAULT_MA_WINDOW

    # Find which button was clicked
    triggered_id = None
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id']
        if 'suggest-weights-btn' in prop_id:
            pattern = re.compile(r'{"index":(\d+),"type":"suggest-weights-btn"}')
            match = pattern.search(prop_id)
            if match:
                triggered_id = int(match.group(1))

    if triggered_id is None:
        return current_states, current_contents

    # These are the components used for weight calculation
    components_for_weights = [c for c in COST_COMPONENTS if c != 'cost']
    
    averages = {}
    all_data_present = True
    for component in components_for_weights:
        log_path = get_log_file_path(triggered_id, component)
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
        return current_states, current_contents

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

    # Create new lists for outputs
    new_states = current_states.copy()
    new_contents = current_contents.copy()
    
    # Find the index of the triggered agent
    for i, n in enumerate(n_clicks):
        if n is not None:
            new_states[i] = True
            new_contents[i] = content
            break

    return new_states, new_contents

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
        logger.info("Application paused")
        return {'display': 'none'}, {'display': 'inline-block'}
    else:  # play-btn
        is_paused = False
        logger.info("Application resumed")
        return {'display': 'inline-block'}, {'display': 'none'}

@app.callback(
    Output('download-dataframe', 'data'),
    Input('export-btn', 'n_clicks'),
    State('data-store', 'data'),
    State('n-agents-store', 'data'),
    prevent_initial_call=True
)
def export_data(n_clicks, stored_data, n_agents):
    """Exports the current data to a numpy archive file."""
    if not n_clicks or not stored_data:
        raise dash.exceptions.PreventUpdate

    # Create a nested dictionary to store the data
    export_dict = {}
    
    # Process data for each agent and component
    for i in range(1, n_agents + 1):
        agent_data = {}
        for component in COST_COMPONENTS:
            log_path = get_log_file_path(i, component)
            data = list(stored_data.get(log_path, []))
            if data:
                # Convert timestamps and values to numpy arrays
                timestamps = np.array([ts for ts, _ in data])
                values = np.array([val for _, val in data])
                agent_data[component] = {
                    'timestamps': timestamps,
                    'values': values
                }
        if agent_data:
            export_dict[f'agent_{i}'] = agent_data

    # Add best agent data
    best_agent_log_path = get_best_agent_log_file_path()
    best_agent_data = list(stored_data.get(best_agent_log_path, []))
    if best_agent_data:
        timestamps = np.array([ts for ts, _ in best_agent_data])
        agent_names = np.array([name for _, name in best_agent_data])
        export_dict['best_agent'] = {
            'timestamps': timestamps,
            'agent_names': agent_names,
            'metadata': {
                'description': 'Current best agent over time',
                'format': 'agent_X where X is the agent number'
            }
        }
        logger.info(f"Exporting best agent data with {len(timestamps)} points")

    # Add metadata
    export_dict['metadata'] = {
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_agents': n_agents,
        'components': COST_COMPONENTS,
        'has_best_agent_data': best_agent_log_path in stored_data,
        'best_agent_format': 'agent_X'  # Document the format of best agent names
    }

    # Create a buffer to store the numpy archive
    buffer = io.BytesIO()
    
    # Save the dictionary to the buffer
    np.savez_compressed(buffer, **export_dict)
    
    # Get the current timestamp for the filename
    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
    filename = f"{timestamp}-cost_export.npz"
    
    # Return the data for download
    return dcc.send_bytes(buffer.getvalue(), filename)

@app.callback(
    [Output('data-store', 'data', allow_duplicate=True),
     Output('n-agents-store', 'data', allow_duplicate=True),
     Output('interval-component', 'n_intervals')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def handle_upload(contents, filename, current_data):
    """Handles uploaded NPZ files and updates the data store."""
    logger.info(f"Upload callback triggered with filename: {filename}")
    
    if contents is None:
        logger.warning("No contents in upload")
        raise dash.exceptions.PreventUpdate

    # Decode the uploaded file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Load the NPZ file
        data = np.load(io.BytesIO(decoded), allow_pickle=True)
        logger.info(f"Loaded NPZ file with keys: {data.files}")
        
        # Initialize new data store
        new_data = current_data.copy() if current_data is not None else {}
        max_agent_id = 0
        
        # Process each agent's data
        for key in data.files:
            if key == 'metadata' or key == 'best_agent':
                continue
                
            agent_id = int(key.split('_')[1])
            max_agent_id = max(max_agent_id, agent_id)
            logger.debug(f"Processing agent {agent_id}")
            
            agent_data = data[key].item()  # Convert numpy array to dictionary
            for component in COST_COMPONENTS:
                if component in agent_data:
                    log_path = get_log_file_path(agent_id, component)
                    timestamps = agent_data[component]['timestamps']
                    values = agent_data[component]['values']
                    
                    logger.debug(f"Processing {component} with {len(timestamps)} data points")
                    
                    # Convert timestamps to datetime objects if they're strings
                    if isinstance(timestamps[0], str):
                        timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
                    elif isinstance(timestamps[0], np.datetime64):
                        timestamps = [pd.Timestamp(ts).to_pydatetime() for ts in timestamps]
                    
                    # Store the data
                    new_data[log_path] = deque(
                        list(zip(timestamps, values)),
                        maxlen=MAX_DATA_POINTS
                    )
        
        # Process best agent data if present
        if 'best_agent' in data.files:
            best_agent_data = data['best_agent'].item()
            timestamps = best_agent_data['timestamps']
            agent_names = best_agent_data['agent_names']
            
            # Convert timestamps to datetime objects if they're strings
            if isinstance(timestamps[0], str):
                timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            elif isinstance(timestamps[0], np.datetime64):
                timestamps = [pd.Timestamp(ts).to_pydatetime() for ts in timestamps]
            
            # Store the best agent data
            best_agent_log_path = get_best_agent_log_file_path()
            new_data[best_agent_log_path] = deque(
                list(zip(timestamps, agent_names)),
                maxlen=MAX_DATA_POINTS
            )
        
        logger.info(f"Processed data for {max_agent_id} agents")
        logger.debug(f"Number of data points in store: {sum(len(v) for v in new_data.values())}")
        
        # Convert deques to lists and ensure datetime serialization
        serializable_data = {}
        for log_path, data_deque in new_data.items():
            serializable_data[log_path] = [
                (ts.isoformat() if isinstance(ts, datetime) else ts, val)
                for ts, val in data_deque
            ]
        
        # Force a refresh by incrementing the interval counter
        current_interval = dash.callback_context.outputs_list[2].get('value', 0)
        new_interval = current_interval + 1
        logger.debug(f"Current interval: {current_interval}, New interval: {new_interval}")
        
        return serializable_data, max_agent_id, new_interval
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise dash.exceptions.PreventUpdate

# --- Main Execution ---
if __name__ == '__main__':
    try:
        logger.info("Starting Cost Monitor application")
        app.run(host='0.0.0.0', port=8050, debug=False)
    finally:
        # Ensure all subprocesses are cleaned up when the app is closed
        logger.info("Shutting down... stopping all ROS listeners.")
        stop_all_listeners()
