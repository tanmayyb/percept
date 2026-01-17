from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os.path
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
  arg_show_processing_delay = DeclareLaunchArgument(
    'show_processing_delay',
    default_value='False',
    description='Show processing delay information'
  )

  arg_show_requests = DeclareLaunchArgument(
    'show_requests',
    default_value='False',
    description='Show service request information'
  )

  remappings = [
    ('/get_velocity_heuristic_circforce', '/manipulator/get_velocity_heuristic_force'),
    ('/get_apf_heuristic_circforce', '/manipulator/get_apf_heuristic_force'),
		]

  return LaunchDescription([
    arg_show_processing_delay,
    arg_show_requests,
    Node(
      package='tf2_ros',
      executable='static_transform_publisher',
      arguments = ['--x', '0', '--y', '0', '--z', '0', 
                  '--yaw', '0', '--pitch', '0', '--roll', '0', 
                  '--frame-id', 'world', 
                  '--child-frame-id', 'panda_link0']
    ),
    Node(
      package='percept_core',
      executable='perception_node',
      name='perception_node',
      arguments=['--ros-args', '--log-level', 'WARN']
    ),
		Node(
			package='percept_core',
			executable='vf_engine',
			name='vf_engine',
			# output='screen',
			parameters=[{
				'show_processing_delay': LaunchConfiguration('show_processing_delay'),
				'show_requests': LaunchConfiguration('show_requests'),
				'show_netforce_output': True
			}],
			remappings=remappings,
		),
    Node(
      package='experiments',
      namespace='manipulator',
      executable='manipulator',
      name='manipulator',
      output='screen',
      arguments=['--ros-args', '--log-level', 'WARN']
    ),
  ])
