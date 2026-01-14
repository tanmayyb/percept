from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
# from launch.conditions import IfCondition




def generate_launch_description():

	return LaunchDescription([

		# Node(
		# 	package='percept_core',
		# 	executable='vf_engine',
		# 	name='vf_engine',
		# 	parameters=[{
		# 		'show_netforce_output': False,
		# 	}],
		# 	# output='log',
		# 	arguments=['--ros-args', '--log-level', 'WARN']
		# ),

		# Node(
		# 	package='percept_core',
		# 	executable='static_scene_loader.py',
		# 	name='static_scene_loader',
		# 	parameters=[{
		# 		'loop_disable': False,
		# 		'publish_rate': 0.03
		# 	}]
		# ),

		# Node(
		# 	package='experiments',
		# 	executable='vfe_service_tester',
		# 	parameters=[{
		# 		# 'service_topic': '/get_apf_heuristic_circforce'
		# 		'service_topic': '/get_velocity_heuristic_circforce'
		# 	}],
		# 	output='screen'
		# )

		Node(
			package='experiments',
			executable='planner_service_tester',
			parameters=[{
				# 'service_topic': '/get_apf_heuristic_force',
				'service_topic': '/get_velocity_heuristic_force',
				'iterations': 1000
			}],
			output='screen'
		),
		Node(
			package='experiments',
			# namespace='oriented_pointmass',
			executable='oriented_pointmass',
			name='oriented_pointmass',
			output='screen'
		),
	])