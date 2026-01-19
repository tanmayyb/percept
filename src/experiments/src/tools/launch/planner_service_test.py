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
      package='experiments',
      namespace='oriented_pointmass',
      executable='oriented_pointmass',
      name='oriented_pointmass',
    ),
      Node(
        package='experiments',
        executable='planner_service_tester',
        name='planner_service_tester',
        parameters=[{
          'service_topics': [            
            '/oriented_pointmass/get_apf_heuristic_force',
            '/oriented_pointmass/get_velocity_heuristic_force',
            '/oriented_pointmass/get_goalobstacle_heuristic_force',
          ],
          'iterations': 1000,
        }],
        output='screen',
      ),

  ])