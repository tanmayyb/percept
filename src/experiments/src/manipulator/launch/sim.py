from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue


def get_remappings():
    return [
        ('/get_velocity_heuristic_circforce', '/manipulator/get_velocity_heuristic_force'),
        ('/get_apf_heuristic_circforce', '/manipulator/get_apf_heuristic_force'),
        ('/get_min_obstacle_distance', '/manipulator/get_min_obstacle_distance'),
    ]

def get_vf_engine_node():
    return Node(
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
        remappings=get_remappings(),
    )

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

    arg_enable_task_sequencer = DeclareLaunchArgument(
        'tasks',
        default_value='true',
        description='Enable task sequencer node'
    )

    robot_description_content = ParameterValue(
        Command(['cat ', EnvironmentVariable('panda_urdf')]),
        value_type=str
    )

    return LaunchDescription([
        arg_show_processing_delay,
        arg_show_requests,
        arg_enable_task_sequencer,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['--x', '0', '--y', '0', '--z', '0', 
                       '--yaw', '0', '--pitch', '0', '--roll', '0', 
                       '--frame-id', 'world', 
                       '--child-frame-id', 'panda_link0']
        ),
        Node(
              package='robot_state_publisher',
              executable='robot_state_publisher',
              parameters=[{'robot_description': robot_description_content}]
          ),
        Node(
            package='percept_core',
            executable='static_scene_loader.py',
            name='static_scene_loader',
            # arguments=['--ros-args', '--log-level', 'WARN'],
            parameters=[{
              'loop_disable': False,
              'publish_rate': 0.03
            }]
        ),
        TimerAction(
            period=2.0,
            actions=[
                get_vf_engine_node(),
                ]
          ),
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='experiments',
                    namespace='manipulator',
                    executable='manipulator',
                    name='manipulator',
                    output='screen',
                    arguments=['--ros-args', '--log-level', 'WARN'],
                )
            ]
        ),
        Node(
            package='experiments',
            executable='task_sequencer',
            name='task_sequencer',
            output='screen',
            condition=IfCondition(LaunchConfiguration('tasks')),
            parameters=[{
                'rad': 0.05
            }]
        ),
    ])