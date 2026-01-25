from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch.actions import TimerAction



# can only test 1 at a time
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

  vfe_tester_node =   Node(
    package='experiments',
    executable='vfe_service_tester',
    parameters=[{
      'service_topic': '/get_apf_heuristic_circforce',
      # 'service_topic': '/get_velocity_heuristic_circforce',
      # 'service_topic': '/get_obstacle_heuristic_circforce',
    }],
    output='screen'
  )

  return LaunchDescription([
    arg_show_processing_delay,
    arg_show_requests,
    # # For existing scene    
    Node(
    	package='percept_core',
    	executable='static_scene_loader.py',
    	name='static_scene_loader',
    	parameters=[{
    		'loop_disable': False,
    		'publish_rate': 0.03
    	}]
    ),

    TimerAction(
      period=1.5,
      actions=[
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
        ),
      ]
    ),  

    # Node(
    #   package='percept_core',
    #   executable='vf_engine_cpu',
    #   name='vf_engine',
    #   parameters=[{
    #     'show_processing_delay': LaunchConfiguration('show_processing_delay'),
    #     'show_requests': LaunchConfiguration('show_requests'),
    #     'show_netforce_output': False
    #   }],
    # ),
    TimerAction(
        period=2.0,
        actions=[vfe_tester_node]
    )


  ])