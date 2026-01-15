from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    package_name = 'experiments'

    return LaunchDescription([
      Node(
        package='experiments', # Replace with actual package name
        executable='planner_service_tester',
        name='planner_service_tester',
        # namespace='oriented_pointmass',
        parameters=[{
          'service_topics': [            
            '/oriented_pointmass/get_apf_heuristic_force',
            '/oriented_pointmass/get_velocity_heuristic_force',
            '/oriented_pointmass/get_goalobstacle_heuristic_force',
          ],
          'iterations': 1000,
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', 'WARN']
      ),
      Node(
        package='experiments',
        executable='parallel_service_tester',
        name='parallel_service_stress_tester',
        parameters=[{
          'service_topics': [            
            '/oriented_pointmass/get_apf_heuristic_force',
            '/oriented_pointmass/get_velocity_heuristic_force',
            '/oriented_pointmass/get_goalobstacle_heuristic_force',
          ],
          'burst_size': 1000,
          'total_bursts': 1,
        }],
        output='screen',
      ),

    ])