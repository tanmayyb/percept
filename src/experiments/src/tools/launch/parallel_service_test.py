from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    package_name = 'experiments'

    remappings = [
      ('/get_min_obstacle_distance', '/oriented_pointmass/get_min_obstacle_distance'),
      ('/get_random_heuristic_circforce', '/oriented_pointmass/get_random_heuristic_force'),
      ('/get_obstacle_heuristic_circforce', '/oriented_pointmass/get_obstacle_heuristic_force'),
      ('/get_goal_heuristic_circforce', '/oriented_pointmass/get_goal_heuristic_force'),
      ('/get_velocity_heuristic_circforce', '/oriented_pointmass/get_velocity_heuristic_force'),
      ('/get_goalobstacle_heuristic_circforce', '/oriented_pointmass/get_goalobstacle_heuristic_force'),
      ('/get_random_heuristic_circforce', '/oriented_pointmass/get_random_heuristic_force'),
      ('/get_apf_heuristic_circforce', '/oriented_pointmass/get_apf_heuristic_force'),
      ('/get_navigation_function_circforce', '/oriented_pointmass/get_navigation_function_force'),
    ]


    return LaunchDescription([

      # # To test parallelism of Planner Service Tester
      # Node(
      #   package='experiments',
      #   executable='planner_service_tester',
      #   name='planner_service_tester',
      #   # namespace='oriented_pointmass',
      #   parameters=[{
      #     'service_topics': [            
      #       '/oriented_pointmass/get_apf_heuristic_force',
      #       '/oriented_pointmass/get_velocity_heuristic_force',
      #       '/oriented_pointmass/get_goalobstacle_heuristic_force',
      #     ],
      #     'iterations': 1000,
      #   }],
      #   output='screen',
      #   # arguments=['--ros-args', '--log-level', 'WARN']
      # ),

      # To test parallelism of VFE
      Node(
        package='percept_core',
        executable='static_scene_loader.py',
        name='static_scene_loader',
        parameters=[{
          'loop_disable': False,
          'publish_rate': 0.03
        }]
      ),
      Node(
        package='percept_core',
        executable='vf_engine',
        name='vf_engine',
        # output='screen',
        parameters=[{
          'show_processing_delay': False,
          'show_requests': False,
          'show_netforce_output': False
        }],
        remappings=remappings,
		  ),
          
      Node(
        package='experiments',
        executable='parallel_service_tester',
        name='parallel_service_stress_tester',
        parameters=[{
          'service_topics': [            
            '/oriented_pointmass/get_apf_heuristic_force',
            '/oriented_pointmass/get_velocity_heuristic_force',
            # '/oriented_pointmass/get_goalobstacle_heuristic_force',
          ],
          'burst_size': 10000,
          'total_bursts': 10,
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', 'WARN']
      ),

    ])