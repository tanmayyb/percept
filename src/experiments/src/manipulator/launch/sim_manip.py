from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, EnvironmentVariable, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue

def get_remappings():
    return [
        ('/get_apf_heuristic_circforce', '/manipulator/get_apf_heuristic_force'),
        ('/get_velocity_heuristic_circforce', '/manipulator/get_velocity_heuristic_force'),
        ('/get_goal_heuristic_circforce', '/manipulator/get_goal_heuristic_force'),
        ('/get_obstacle_heuristic_circforce', '/manipulator/get_obstacle_heuristic_force'),
        ('/get_goalobstacle_heuristic_circforce', '/manipulator/get_goalobstacle_heuristic_force'),
        ('/get_random_heuristic_circforce', '/manipulator/get_random_heuristic_force'),
        ('/get_min_obstacle_distance', '/manipulator/get_min_obstacle_distance'),
    ]

def get_vf_engine_node():
    return Node(
        package='percept_core',
        executable='vf_engine',
        name='vf_engine',
        parameters=[{
            'point_radius': 0.01
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
        default_value='false',
        description='Enable task sequencer node'
    )

    arg_enable_dynamic_scene = DeclareLaunchArgument(
        'dynamic',
        default_value='true',
        description='Enable dynamic scene loader node'
    )

    robot_description_content = ParameterValue(
        Command(['cat ', EnvironmentVariable('panda_urdf')]),
        value_type=str
    )

    return LaunchDescription([
        arg_show_processing_delay,
        arg_show_requests,
        arg_enable_task_sequencer,
        arg_enable_dynamic_scene,
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
            condition=UnlessCondition(LaunchConfiguration('dynamic')),
            parameters=[{
              'loop_disable': False,
              'publish_rate': 0.03
            }]
        ),
        Node(
            package='percept_core',
            executable='dynamic_scene_loader.py',
            name='dynamic_scene_loader',
            condition=IfCondition(LaunchConfiguration('dynamic')),
            parameters=[{
              'ping_pong': True,
              'frame_rate': 30.0
            }]
        ),
        TimerAction(
            period=0.5,
            actions=[
              get_vf_engine_node(),
            ]
        ),
        TimerAction(
            period=0.75,
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