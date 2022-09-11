from __future__ import division

import argparse
import math
import time
import sys

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String

COLOUR_IN_FRONT_THRESHOLD = 20000      # How much of a colour has to be seen to be considered "a square in front"
FULL_SQUARE = 1                        # The width of one square (or tile)
GREEN_IN_RANGE_THRESHOLD = 5000        # How much green has to be seen in front for the robot to go straight for
HALF_SQUARE = FULL_SQUARE / 2          # The width of half one square (or tile)
HIGH_TURN_SPEED = 0.5                  # High tuen speed
IN_LOOP_POSITION_THRESHOLD = 0.25      # Positional similarity threshold for history comparisons
IN_LOOP_YAW_THRESHOLD = 15             # Rotational similarity threshold for history comparisons
LOW_TURN_SPEED = 0.25                  # Low turn speed
MIN_COLOUR_DETECTION_THRESHOLD = 1000  # How much of a colour has to be seen for it to be detected
MIN_WALL_DISTANCE = 0.4                # How close a wall has to be to be considered "too close"
TARGET_CLOSENESS_THRESHOLD = 0.55      # The closest the robot can get to a wall while approaching
UNTRAVERSABLE_THRESHOLD = 1            # How far a wall has to be to be considered "in the way"


class Directions:
    """A helper class for direction decisions.

    Args:
        repeat (bool): Whether to use adjusted biases on a repeat step. Used to try and break out of loops. Defaults to False.

    Attributes:
        straight (int): The score for going straight.
        right (int): The score for going right.
        back (int): The score for going back.
        left (int): The score for going left.
        cycle (tuple[str, str, str, str]): A tuple containing all the possible directions in the order they should be scanned.
        best (str): A property that calculates the best direction to proceeed.
    """

    __slots__ = ("straight", "right", "back", "left", "cycle")

    def __init__(self, repeat=False):
        # Use biases to set priorities. For example, we want to go left if we can and if there's no
        # blue or green in any other direction. Therefore, we set a higher bias to go left, so it
        # acts as a default of sorts.
        if repeat:
            self.straight = 0.3
            self.right = 0.2
            self.back = 0.0
            self.left = 0.1
        else:
            self.straight = 0.2
            self.right = 0.1
            self.back = 0.0
            self.left = 0.3

        self.cycle = ("straight", "right", "back", "left")

    def __repr__(self):
        return "^ %s | > %s | v %s | < %s" % (
            self.straight, self.right, self.back, self.left
        )

    def __getitem__(self, key):
        return getattr(self, self.cycle[key])

    def __setitem__(self, key, value):
        setattr(self, self.cycle[key], value)

    @property
    def best(self):
        """Compares the scores for all directions and returns the best in which to proceed.

        Returns:
            str: The best direction to proceed.
        """

        return max(
            (i for i in self.__dict__.items() if i[0] != "cycle"),
            key=lambda x: x[1],
        )[0]


class History:
    """A helper class for recording history.

    Attributes:
        pos (Twist): The robot's position at this point.
        yaw (float): The robot's yaw at this point.
    """

    __slots__ = ("pos", "yaw")

    def __init__(self, pos, yaw):
        self.pos = pos
        self.yaw = yaw


class Listener:
    """The main listener class.

    Args:
        skip_scan (bool): Whether to scan for initial hints. This should only be used when debugging.

    Attributes:
        colour_subscriber (Subscriber): A subscriber for information about colours the robot can see.
        dist_subscriber (Subscriber): A subscriber for the robot's distance from the wall in front of it.
        yaw_subscriber (Subscriber): A subscriber for the robot's current yaw value.
        odom_subscriber (Subscriber): A subscriber for odometry information.
        velocity_publisher (Publisher): A publisher for the robot's movement.
        yaw (float): The robot's yaw.
        dist (float): The distance between the robot and the wall in front.
        colours (dict[str, list[float]]): A dictionary of all the colour sections -- and the amount of each -- the robot can see.
        pos (Twist): The robot's current position.
        speed (float): The robot's movement speed.
        turn_speed (float): The robot's turn speed.
        turn_error_threshold (float): The amount in degrees the robot is allowed to deviate from a given target angle.
        move_error_threshold (float): The amount in metres the robot is allowed to overshoot or undershoot a target distance.
        history (list[History]): A list containing the position and yaw of the robot in every step.
    """

    def __init__(self, skip_scan=False):
        self.colour_subscriber = rospy.Subscriber(
            "/result/colours", String, callback=self.colour_callback
        )
        self.dist_subscriber = rospy.Subscriber(
            "/result/distance", Float64, callback=self.dist_callback
        )
        self.yaw_subscriber = rospy.Subscriber(
            "/result/yaw", Float64, callback=self.yaw_callback
        )
        self.odom_subscriber = rospy.Subscriber(
            "/odom", Odometry, callback=self.odom_callback
        )

        self.velocity_publisher = rospy.Publisher(
            "/mobile_base/commands/velocity", Twist, queue_size=1
        )

        self.yaw = 0.0
        self.dist = 0.0
        self.colours = {}
        self.pos = Twist()

        self.speed = 0.2
        self.turn_speed = LOW_TURN_SPEED
        self.turn_error_threshold = 2.5
        self.move_error_threshold = 0.05
        self.history = []

        rospy.sleep(1)  # Give rospy a chance to catch up before running.
        self.run(skip_scan)

    def get_nearest_right_angle(self, angle):
        """Gets the nearest right angle to `angle`.

        Args:
            angle (float): The angle to get the nearest right angle to.

        Returns:
            int: The nearest right angle to `angle`.
        """

        ra = round(angle / 90) * 90
        if ra == -180:
            # -180 is not necessary, and causes problems
            ra = 180
        return ra

    def resolve_angle(self, angle):
        """Returns the angle as its equivalent between -180 and 180.

        Args:
            angle (float): The angle to resolve.

        Returns:
            float: The resolved angle.
        """

        if angle < -179:
            return angle + 360
        elif angle > 180:
            return angle - 360
        return angle

    def get_best_turn_direction(self, target):
        """Gets the best direction to turn toward the target angle.

        Args:
            target (int | float): The target angle.

        Returns:
            int: The direction to turn.
        """

        def _cyclic_array():
            y = round(self.yaw)
            arr = np.arange(y - 180, y + 180)
            arr[arr > 180] -= 360
            arr[arr < -179] += 360
            return arr

        arr = _cyclic_array()
        if np.where(arr == target)[0][0] < 180:
            return -1
        return 1

    def turn_towards(self, target, direction=0):
        """Turns the robot by a degree towards a target angle. This should be used in a for or while loop.

        Args:
            target (int | float): The target angle.
            direction (int): The direction to turn in. If none is given, it is automatically calculated.

        Returns:
            bool: Whether the robot needed to be turned. Should be used to determine when to break from the loop.
        """

        t = Twist()
        d = direction or self.get_best_turn_direction(target)

        if not (
            target - self.turn_error_threshold
            <= round(self.yaw)
            <= target + self.turn_error_threshold
        ):
            angle = self.turn_speed * d
            t.angular.z += angle
            self.velocity_publisher.publish(t)
            return True

        return False

    def move(self, target):
        """Moves the robot by `turn_speed` metres forward. This should be used in a for or while loop.

        Args:
            target (int): The target distance to the wall in front of the robot.

        Returns:
            bool: Whether the robot needed to be moved. Should be used to determine when to break from the loop.
        """

        t = Twist()

        if self.dist > target:
            t.linear.x += self.speed
            self.velocity_publisher.publish(t)
            return True

        return False

    def move_back(self, dist):
        """Moves the robot by `turn_speed` metres backwards. This should be used in a for or while loop.

        Args:
            dist (float): The amount to move backwards.

        Returns:
            bool: Whether the robot needed to be moved. Should be used to determine when to break from the loop.
        """

        t = Twist()

        if self.dist - dist <= 0.1:
            t.linear.x -= self.speed
            self.velocity_publisher.publish(t)
            return True

        return False

    def max_of(self, colour):
        """Returns the area of the largest section of the given colour.

        Args:
            colour (str): The colour to look for.

        Returns:
            float: The largest area of the given colour.
        """

        return max(self.colours.get(colour, [0]))

    def go_to_starting_square(self):
        """Heads to the starting square.
        """

        print "Heading to starting square..."

        target_dist = self.dist - HALF_SQUARE
        while self.move(target_dist):
            continue

        target_angle = self.get_nearest_right_angle(
            self.resolve_angle(self.yaw + 90)
        )
        direction = self.get_best_turn_direction(target_angle)
        while self.turn_towards(target_angle):
            continue

        target_dist = self.dist - HALF_SQUARE
        while self.move(target_dist):
            continue

    def scan_for_hints(self):
        """Scan for initial hints.
        """

        print "Scanning for hints..."
        found = False

        for _ in xrange(4):
            for _ in xrange(3):
                target_angle = self.get_nearest_right_angle(
                    self.resolve_angle(self.yaw - 90)
                )
                hint_yaws = []

                while self.turn_towards(target_angle, -1):
                    if self.max_of("blue") > MIN_COLOUR_DETECTION_THRESHOLD:
                        hint_yaws.append(self.yaw)

                if hint_yaws:
                    hint_yaw = hint_yaws[0]

                    target_angle = self.get_nearest_right_angle(
                        self.resolve_angle(hint_yaw)
                    )
                    direction = self.get_best_turn_direction(target_angle)

                    while self.turn_towards(target_angle):
                        continue

                    break

            target_dist = self.dist - FULL_SQUARE
            while self.move(target_dist):
                continue

            if hint_yaws:
                return

    def search_for_exit(self):
        """Searches for the exit (the green square).

        In every step, the robot:
        1. Checks it's not in a loop; if it is, it sets the directional biases differently
        2. Logs its current position and yaw to their respective histories
        3. Looks around to determine which direction is best
            a. Different colours adjust the scores differently (for example, green is +2).
            b. If it sees its too close to a wall, it moves away slightly.
            c. If there is enough green in a particular direction, it sets an endpoint flag.
        4. Determines the best direction to proceed
        5. Turns to head in that direction
        6. Moves (a maximum of 1m) forward in that direction.

        Once on the green, the method returns.
        """

        step = 1
        found_green = False
        ILPT = IN_LOOP_POSITION_THRESHOLD
        ILYT = IN_LOOP_YAW_THRESHOLD

        def in_loop():
            for h in self.history:
                if (
                    (h.pos.x - ILPT < self.pos.x < h.pos.x + ILPT)
                    and (h.pos.y - ILPT < self.pos.y < h.pos.y + ILPT)
                ):
                    if h.yaw - ILYT < self.yaw < h.yaw + ILYT:
                        return True
                    return False
            return False

        while True:
            print "\nSTEP %s" % step

            # if self.max_of("green") > GREEN_IN_RANGE_THRESHOLD:
            #     print "Heading for green..."
            #     target_angle = self.get_nearest_right_angle(
            #         self.resolve_angle(self.yaw)
            #     )
            #     direction = self.get_best_turn_direction(target_angle)
            #     while self.turn_towards(target_angle, direction):
            #         continue
            #     while self.move(HALF_SQUARE * 1.5):
            #         continue
            #     return

            # Check if robot is in a loop, and adjust biases if so
            if in_loop():
                print "Using adjusted biases to break from loop."
                directions = Directions(repeat=True)
            else:
                directions = Directions()

            self.history.append(History(self.pos, self.yaw))

            # Check directions
            print "Looking around..."
            for i in xrange(len(directions.cycle)):
                dist = self.dist
                if dist < MIN_WALL_DISTANCE:
                    print "Moving away from wall..."
                    while self.move_back(dist):
                        continue

                if dist < UNTRAVERSABLE_THRESHOLD:
                    directions[i] -= 1
                elif i != 2:
                    green = self.max_of("green")
                    if green > COLOUR_IN_FRONT_THRESHOLD:
                        found_green = True
                    if green > MIN_COLOUR_DETECTION_THRESHOLD:
                        directions[i] += 2
                    if self.max_of("blue") > MIN_COLOUR_DETECTION_THRESHOLD:
                        directions[i] += 1
                    if self.max_of("red") > COLOUR_IN_FRONT_THRESHOLD:
                        directions[i] -= 0.5

                if i < 3:
                    target_angle = self.get_nearest_right_angle(
                        self.resolve_angle(self.yaw - 90)
                    )
                    direction = self.get_best_turn_direction(target_angle)
                    while self.turn_towards(target_angle, direction):
                        continue

            # Decide on best way to proceed
            best = directions.best
            print "Scores: [%r]\nGoing %s..." % (directions, best)

            if best == "straight":
                target_angle = self.get_nearest_right_angle(
                    self.resolve_angle(self.yaw - 90)
                )
                direction = self.get_best_turn_direction(target_angle)
                while self.turn_towards(target_angle, direction):
                    continue
            elif best == "right":
                target_angle = self.get_nearest_right_angle(
                    self.resolve_angle(self.yaw + 180)
                )
                direction = self.get_best_turn_direction(target_angle)
                while self.turn_towards(target_angle, direction):
                    continue
            elif best == "back":
                target_angle = self.get_nearest_right_angle(
                    self.resolve_angle(self.yaw + 90)
                )
                direction = self.get_best_turn_direction(target_angle)
                while self.turn_towards(target_angle, direction):
                    continue

            # Move in selected direction
            target_dist = max(
                TARGET_CLOSENESS_THRESHOLD,
                self.dist - FULL_SQUARE
            )
            print "Moving %sm..." % round(self.dist - target_dist, 3)
            while self.move(target_dist):
                continue

            if found_green:
                return

            step += 1

    def run(self, skip_scan):
        """Run all methods and track how long it took.

        Args:
            skip_scan (bool): Whether to skip the initial scanning phase.
        """

        start = time.time()
        if skip_scan:
            print "WARNING: Skipping initial hint scan!"
        else:
            self.go_to_starting_square()
            self.scan_for_hints()
        self.search_for_exit()

        print "\nCOMPLETE! Time taken: %im %is" % (
            divmod(time.time() - start, 60)
        )
        sys.exit(0)

    def colour_callback(self, data):
        """Updates the `colours` attribute.

        Args:
            data (String) The data received from the talker.
        """

        self.colours = yaml.load(data.data)

    def dist_callback(self, data):
        """Updates the `dist` attribute.

        Args:
            data (Float) The data received from the talker.
        """

        self.dist = data.data

    def yaw_callback(self, data):
        """Updates the `yaw` attribute.

        Args:
            data (Float) The data received from the talker.
        """

        self.yaw = data.data

    def odom_callback(self, data):
        """Updates the `pos` attribute.

        Args:
            data (Odometry) The data received from the talker.
        """

        self.pos = data.pose.pose.position


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the listener node.")
    parser.add_argument(
        "--skip-scan",
        help="Skip scanning for initial hints.",
        action="store_true",
    )
    args = parser.parse_args()

    rospy.init_node("listener", anonymous=True)
    l = Listener(args.skip_scan)
    rospy.spin()
