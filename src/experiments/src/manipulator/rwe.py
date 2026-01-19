from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import TimerAction


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
    ('/get_min_obstacle_distance', '/manipulator/get_min_obstacle_distance'),
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
			parameters=[{
				'show_processing_delay': False,
				'show_requests': False,
				'show_netforce_output': False,
        'mass_radius': 0.04,
        'agent_radius': 0.04,
          
			}],
			remappings=remappings,
		),
    TimerAction(
      period=2.0,
      actions=[
        Node(
          package='experiments',
          namespace='manipulator',
          executable='manipulator',
          name='manipulator',
          output='screen',
          arguments=['--ros-args', '--log-level', 'WARN']
        )]
    )
  ])
