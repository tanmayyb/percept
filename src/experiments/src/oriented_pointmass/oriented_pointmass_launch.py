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
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    return LaunchDescription([
        enable_rviz_arg,
        Node(
            package='experiments',
            namespace='oriented_pointmass',
            executable='oriented_pointmass',
            name='oriented_pointmass',
            output='screen'
        ),
        Node(
            package='rviz2',
            namespace='rviz2',
            executable='rviz2',
            name='rviz',
            condition=IfCondition(LaunchConfiguration('enable_rviz')),
            arguments=['-d' + os.path.join(get_package_share_directory('experiments'), 'oriented_pointmass', 'oriented_pointmass.rviz')]
        )
    ])
