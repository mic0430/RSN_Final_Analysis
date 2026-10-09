"""
vn100_driver.py
================
EECE5554 Final Project

ROS2 driver for the VectorNav VN-100 IMU.
Parses $VNYMR sentences over serial and publishes standard sensor_msgs/Imu.

Publishes: /imu  (sensor_msgs/Imu)

Save to: EECE5554/FinalProject/agps_driver/agps_driver/vn100_driver.py
"""

import rclpy                                # ros2 python library
from rclpy.node import Node                 # base class for ros2 nodes
from sensor_msgs.msg import Imu             # standard imu message type
from std_msgs.msg import Header             # message header with timestamp
import serial                               # for reading the usb serial port
import math


class VN100Driver(Node):

    def __init__(self):
        super().__init__('vn100_driver')     # node name shows up in ros2 node list

        # --- get the serial port from launch parameter ---
        self.declare_parameter('port', '/dev/ttyUSB0')
        port = self.get_parameter('port').get_parameter_value().string_value

        # --- open the serial connection ---
        self.get_logger().info(f'opening serial port: {port}')
        self.ser = serial.Serial(
            port=port,
            baudrate=115200,                 # vn-100 default baud rate
            timeout=1.0                      # 1 second read timeout
        )

        # --- set the vn-100 to output at 40 Hz ---
        # this writes to register 7 (async data output frequency)
        self.set_output_rate(40)

        # --- create the publisher ---
        # publishes sensor_msgs/Imu on the /imu topic
        # queue size 10 means it buffers up to 10 messages
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)

        # --- create a timer that reads serial data ---
        # 0.005 seconds = 200 Hz polling rate (fast enough to catch 40 Hz data)
        self.timer = self.create_timer(0.005, self.read_serial)

        self.get_logger().info('vn100 driver started, publishing on /imu at 40 Hz')

    def set_output_rate(self, hz):
        """
        tell the vn-100 to output data at the given rate.
        writes to register 7 with a proper checksum.
        """
        # build the command string without checksum
        cmd_body = f'VNWRG,07,{hz}'

        # compute xor checksum (vn protocol requirement)
        checksum = 0
        for ch in cmd_body:
            checksum ^= ord(ch)

        # format the full command with $ prefix and * before checksum
        full_cmd = f'${cmd_body}*{checksum:02X}\r\n'

        self.ser.write(full_cmd.encode())    # send to the sensor
        self.get_logger().info(f'set output rate to {hz} Hz')

    def read_serial(self):
        """
        called by the timer 200 times per second.
        reads one line from serial. if its a $VNYMR sentence, parse and publish it.
        """
        try:
            # read one line from serial (blocks until newline or timeout)
            raw = self.ser.readline()

            if not raw:                      # timeout, nothing to read
                return

            line = raw.decode('ascii', errors='ignore').strip()

            # we only care about $VNYMR sentences (register 27)
            # format: $VNYMR,yaw,pitch,roll,magX,magY,magZ,accelX,accelY,accelZ,gyroX,gyroY,gyroZ*checksum
            if not line.startswith('$VNYMR'):
                return

            # --- validate checksum ---
            if '*' not in line:
                return

            data_part = line[1:line.index('*')]              # everything between $ and *
            received_cs = line[line.index('*') + 1:]         # checksum after *

            computed_cs = 0
            for ch in data_part:
                computed_cs ^= ord(ch)

            if f'{computed_cs:02X}' != received_cs.upper():
                return                                        # bad checksum, skip

            # --- parse the fields ---
            parts = data_part.split(',')
            if len(parts) < 13:              # need at least 13 fields
                return

            # fields from the $VNYMR sentence (see VN-100 user manual, register 27)
            # parts[0] = "VNYMR"
            yaw   = float(parts[1])          # degrees
            pitch = float(parts[2])          # degrees
            roll  = float(parts[3])          # degrees
            # parts[4,5,6] = magnetometer (gauss) — we dont publish this
            accel_x = float(parts[7])        # m/s^2  (vn-100 outputs in m/s^2)
            accel_y = float(parts[8])        # m/s^2
            accel_z = float(parts[9])        # m/s^2
            gyro_x  = float(parts[10])       # rad/s  (vn-100 outputs in rad/s)
            gyro_y  = float(parts[11])       # rad/s
            gyro_z  = float(parts[12])       # rad/s

            # --- build the Imu message ---
            msg = Imu()

            # header: frame id and timestamp
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'imu_frame'

            # orientation from euler angles (convert degrees to radians)
            # we store as quaternion because thats what sensor_msgs/Imu expects
            q = euler_to_quaternion(
                math.radians(roll),
                math.radians(pitch),
                math.radians(yaw)
            )
            msg.orientation.x = q[0]
            msg.orientation.y = q[1]
            msg.orientation.z = q[2]
            msg.orientation.w = q[3]

            # angular velocity (already in rad/s from the vn-100)
            msg.angular_velocity.x = gyro_x
            msg.angular_velocity.y = gyro_y
            msg.angular_velocity.z = gyro_z

            # linear acceleration (already in m/s^2 from the vn-100)
            msg.linear_acceleration.x = accel_x
            msg.linear_acceleration.y = accel_y
            msg.linear_acceleration.z = accel_z

            # publish it
            self.imu_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'error reading serial: {e}')

    def destroy_node(self):
        """clean up when the node shuts down"""
        if self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def euler_to_quaternion(roll, pitch, yaw):
    """
    convert euler angles (radians) to a quaternion [x, y, z, w].
    this is standard aerospace convention (ZYX rotation order).
    """
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return [x, y, z, w]


def main():
    rclpy.init()                             # start ros2
    node = VN100Driver()                     # create the driver node
    try:
        rclpy.spin(node)                     # keep running until ctrl+c
    except KeyboardInterrupt:
        pass
    node.destroy_node()                      # clean up
    rclpy.shutdown()                         # shut down ros2


if __name__ == '__main__':
    main()
