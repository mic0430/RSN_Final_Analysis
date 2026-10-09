"""
gps_driver.py
Simple GPS driver for BU-353S4. Parses $GPGGA sentences.
Publishes sensor_msgs/NavSatFix on /gps/fix.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
import serial


class GPSDriver(Node):

    def __init__(self):
        super().__init__('gps_driver')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 4800)

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud_rate').get_parameter_value().integer_value

        self.get_logger().info(f'opening gps on {port} at {baud} baud')
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=1.0)

        self.pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.timer = self.create_timer(0.05, self.read_serial)

        self.get_logger().info('gps driver started, publishing on /gps/fix')

    def read_serial(self):
        try:
            raw = self.ser.readline()
            if not raw:
                return

            line = raw.decode('ascii', errors='ignore').strip()

            if not line.startswith('$GPGGA'):
                return

            parts = line.split(',')
            if len(parts) < 10:
                return

            fix_quality = int(parts[6]) if parts[6] else 0

            if fix_quality == 0 or parts[2] == '' or parts[4] == '':
                msg = NavSatFix()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'gps_frame'
                msg.status.status = -1
                msg.latitude = 0.0
                msg.longitude = 0.0
                self.pub.publish(msg)
                return

            lat_raw = float(parts[2])
            lat_deg = int(lat_raw / 100)
            lat_min = lat_raw - lat_deg * 100
            lat = lat_deg + lat_min / 60.0
            if parts[3] == 'S':
                lat = -lat

            lon_raw = float(parts[4])
            lon_deg = int(lon_raw / 100)
            lon_min = lon_raw - lon_deg * 100
            lon = lon_deg + lon_min / 60.0
            if parts[5] == 'W':
                lon = -lon

            alt = float(parts[9]) if parts[9] else 0.0

            msg = NavSatFix()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'gps_frame'
            msg.status.status = 0
            msg.latitude = lat
            msg.longitude = lon
            msg.altitude = alt

            self.pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'gps error: {e}')

    def destroy_node(self):
        if self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = GPSDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
