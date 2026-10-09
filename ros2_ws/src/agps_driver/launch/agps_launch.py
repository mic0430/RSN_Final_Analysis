from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    imu_port_arg = DeclareLaunchArgument('imu_port', default_value='/dev/ttyUSB0')
    gps_port_arg = DeclareLaunchArgument('gps_port', default_value='/dev/ttyACM0')

    imu_node = Node(
        package='agps_driver',
        executable='vn100_driver',
        name='vn100_imu',
        parameters=[{'port': LaunchConfiguration('imu_port')}],
        output='screen',
    )

    gps_node = Node(
        package='agps_driver',
        executable='gps_driver',
        name='bu353_gps',
        parameters=[{
            'port': LaunchConfiguration('gps_port'),
            'baud_rate': 4800,
        }],
        output='screen',
    )

    return LaunchDescription([imu_port_arg, gps_port_arg, imu_node, gps_node])
