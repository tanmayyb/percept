from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration




def generate_launch_description():

  arg_num_iterations = DeclareLaunchArgument(
    'n',
    default_value='1000',
    description='Number of iterations'
  )

  return LaunchDescription([
    arg_num_iterations,    

    Node(
      package='experiments', # Replace with actual package name
      executable='planner_service_tester',
      name='planner_service_tester',
      # namespace='oriented_pointmass',
      parameters=[{
        'service_topics': [            
          # '/get_apf_heuristic_force',
          # '/get_velocity_heuristic_force',
          # '/get_goalobstacle_heuristic_force',
          
          '/oriented_pointmass/get_apf_heuristic_force',
          '/oriented_pointmass/get_velocity_heuristic_force',
          '/oriented_pointmass/get_goalobstacle_heuristic_force',
          
          # '/oriented_pointmass/get_apf_heuristic_force',
          # '/oriented_pointmass/get_velocity_heuristic_force',
          # '/oriented_pointmass/get_goalobstacle_heuristic_force',
          # '/oriented_pointmass/get_goal_heuristic_force',
          # '/oriented_pointmass/get_min_obstacle_distance',
          # '/oriented_pointmass/get_navigation_function_force',
          # '/oriented_pointmass/get_obstacle_heuristic_force',
          # '/oriented_pointmass/get_random_heuristic_force',
        ],
        'iterations': LaunchConfiguration('n'),
      }],
      # output='screen',
      # arguments=['--ros-args', '--log-level', 'WARN']
    ),
        
    Node(
      package='experiments',
      namespace='oriented_pointmass',
      executable='oriented_pointmass',
      name='oriented_pointmass',
    ),
  ])