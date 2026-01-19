from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os.path
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition


def generate_launch_description():
	arg_rviz2_disable = DeclareLaunchArgument(
		'rviz2_disable',
		default_value='False',
		description='Whether to launch RViz'
	)

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
		arg_rviz2_disable,
		arg_show_processing_delay,
		arg_show_requests,
						
		# nodes
		Node(
			package='experiments',
			namespace='oriented_pointmass',
			executable='oriented_pointmass',
			name='oriented_pointmass',
			output='screen'
		),
		Node(
			package='percept_core',
			executable='vf_engine',
			name='vf_engine',
			# output='screen',
			parameters=[{
				'show_processing_delay': LaunchConfiguration('show_processing_delay'),
				'show_requests': LaunchConfiguration('show_requests'),
				'show_netforce_output': False
			}],
			remappings=remappings,
		),

		Node(
      package='percept_core',
      executable='perception_node',
      name='perception_node',
      arguments=['--ros-args', '--log-level', 'WARN']
		),
		
		# conditional nodes
		Node(
			package='rviz2',
			namespace='rviz2',
			executable='rviz2',
			name='rviz',
			condition=IfCondition(PythonExpression(['not ', LaunchConfiguration('rviz2_disable')])),
			arguments=['-d' + os.path.join(get_package_share_directory('experiments'), 
																	'oriented_pointmass', 'oriented_pointmass.rviz')]
		)
	])
