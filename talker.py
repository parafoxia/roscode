import argparse
import math
from collections import defaultdict

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float64, String
from tf_conversions import transformations

COLOURS_PRE_CROP = {
    "yellow": (20, 50),
    "green": (50, 100),
    "blue": (100, 150),
}
COLOURS_POST_CROP = {
    "red": (0, 5),
}


class Talker:
    """The main talker class.

    Args:
        view (bool): Whether to display the robot's persective.

    Attributes:
        bridge (CvBridge): The OpenCV bridge for ROS.
        image_subscriber (Subscriber): A subscriber for image information.
        scan_subscriber (Subscriber): A subscriber for scan information (used for distance).
        odom_subscriber (Subscriber): A subscriber for odometry information.
        colour_publisher (Publisher): A publisher for information about colours the robot can see.
        dist_publisher (Publisher): A publisher for the robot's distance from the wall in front of it.
        yaw_publisher (Publisher): A publisher for the robot's current yaw value.
    """

    def __init__(self, view):
        self.view = view
        self.bridge = CvBridge()

        self.image_subscriber = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.image_callback
        )
        self.scan_subscriber = rospy.Subscriber(
            "/scan", LaserScan, self.scan_callback
        )
        self.odom_subscriber = rospy.Subscriber(
            "/odom", Odometry, self.odom_callback
        )

        self.colour_publisher = rospy.Publisher(
            "/result/colours", String, queue_size=1
        )
        self.dist_publisher = rospy.Publisher(
            "/result/distance", Float64, queue_size=1
        )
        self.yaw_publisher = rospy.Publisher(
            "/result/yaw", Float64, queue_size=1
        )

    def get_visible_colours(self, image):
        """Gets information on all visible blocks of colour, and the amount of each.

        Args:
            image (Image): A snapshot of the robot's current view.

        Returns:
            dict: The colour information, where the key is the colour, and the value is a list of all blocks.
        """

        colours = defaultdict(list)

        for colour, (lower, upper) in COLOURS_PRE_CROP.items():
            threshold = cv2.inRange(
                image, np.array((lower, 150, 50)), np.array((upper, 255, 255))
            )
            contours = cv2.findContours(
                threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )[1]
            for contour in contours:
                area = cv2.contourArea(contour)
                colours[colour].append(area)

        image = image[300:480, 100:540]

        for colour, (lower, upper) in COLOURS_POST_CROP.items():
            threshold = cv2.inRange(
                image, np.array((lower, 150, 50)), np.array((upper, 255, 255))
            )
            contours = cv2.findContours(
                threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )[1]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2500:
                    colours[colour].append(area)

        return dict(colours)

    def get_min_distance_to_wall(self, data):
        """Gets the minimum distance from the wall

        Args:
            data (Scan): The scan data from ROS.

        Returns:
            float: The minimum of all provided distances.
        """

        min_range = data.range_max
        for r in data.ranges[300:340]:
            if r < min_range:
                min_range = r
        return min_range

    def get_max_distance_to_wall(self, data):
        """Gets the maximum distance from the wall

        Args:
            data (LaserScan): The scan data from ROS.

        Returns:
            float: The maximum of all provided distances.
        """

        max_range = data.range_min
        for r in data.ranges[300:340]:
            if r > max_range:
                max_range = r
        return max_range

    # def get_distance_to_wall(self, data):
    #     return np.max(data.ranges[300:340])

    def get_orientation(self, orientation):
        """Get the current yaw angle in degrees.

        Args:
            orientation (Orientation): The orientation data from ROS.

        Returns:
            float: The yaw in degrees.
        """

        yaw = transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )[2]
        return math.degrees(yaw)

    def image_callback(self, data):
        """Takes image data and publishes colour information.

        Args:
            data (Image): The image data from ROS.
        """

        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        colours = self.get_visible_colours(hsv_image)

        s = String()
        s.data = yaml.dump(colours, indent=2)
        self.colour_publisher.publish(s)

        if self.view:
            cv2.namedWindow("Turtlevision", 1)
            cv2.imshow("Turtlevision", image)
            cv2.waitKey(1)

    def scan_callback(self, data):
        """Publishes distance information.

        Args:
            data (LaserScan): The laserscan information from ROS.
        """

        f = Float64()
        f.data = self.get_max_distance_to_wall(data)
        self.dist_publisher.publish(f)

    def odom_callback(self, data):
        """Publishes yaw information.

        Args:
            data (Odometry): The odometry information from ROS.
        """

        f = Float64()
        f.data = self.get_orientation(data.pose.pose.orientation)
        self.yaw_publisher.publish(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the talker node.")
    parser.add_argument(
        "-v", "--view", help="View what the robot sees.", action="store_true"
    )
    args = parser.parse_args()

    rospy.init_node("talker", anonymous=True)
    t = Talker(args.view)
    rospy.spin()
    cv2.destroyAllWindows()
