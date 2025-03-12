# Evaluation


## Process Results


### Methods of `MetricsReader` Class

This table lists the methods of the `MetricsReader` class, their indices, processed variables, and a brief description.

| Index | Method                         | Processed Variables                                                                                      | Description |
|-------|--------------------------------|----------------------------------------------------------------------------------------------------------|-------------|
| 1     | `__init__`                     | `filepath`, `success_time_limit`, `acceptable_min_distances_to_target`                                   | Initializes the class, loads data from the YAML file, and sets configuration values. |
| 2     | `_load_data`                    | `log`, `workload_config`, `poses`, `agents`, `FORCE_CONFIG_LABEL`, `start_pos`, `goal_pos`, etc.         | Reads YAML data, extracts configuration parameters, and initializes data structures. |
| 3     | `_parse_best_agent_record`      | `record`                                                                                                 | Extracts and processes the best agent information from a log record. |
| 4     | `_update_2d_layout`             | `fig`, `title`, `xaxis_title`, `yaxis_title`, `extra`                                                    | Updates the layout of 2D Plotly figures. |
| 5     | `success_criteria`              | `num_collisions`, `elapsed_motion_runtime`, `min_distances_to_target`                                    | Determines if a trial was successful based on predefined conditions. |
| 6     | `process_info_and_analysis`     | `label`, `pointcloud`, `num_agents`, `detection_shell_radius`, `agent_positions`, etc.                   | Analyzes experiment data, computes statistics, and detects collisions. |
| 7     | `format_stats`                  | `info`                                                                                                   | Formats experiment statistics for output. |
| 8     | `print_stats`                   | _None_                                                                                                   | Prints the experiment statistics to the console. |
| 9     | `save_stats`                    | `output_path`                                                                                            | Saves experiment statistics to a file. |
| 10    | `plot_scene`                    | `x_coords`, `y_coords`, `z_coords`, `timestamps`, `points`, etc.                                         | Generates a 3D plot of the agent's movement and obstacles. |
| 11    | `plot_agent_planning_time`      | `df`, `y_values`, `q1`, `q3`, `y_max`, `fig`, `agent_id`, `agent_data`                                   | Plots agent planning time over the experiment duration. |
| 12    | `plot_best_agent_selection`     | `df`, `heuristic_mapping`, `fig`                                                                         | Visualizes the best agent selection over time. |
| 13    | `plot_distance_to_goal`         | `timestamps`, `distances`, `fig`                                                                         | Plots the distance of the agent to the goal over time. |
| 14    | `plot_distance_to_closest_obstacle` | `pointcloud`, `timestamps`, `agent_positions`, `min_distances`, `fig`                              | Plots the distance of the agent to the closest obstacle over time. |



