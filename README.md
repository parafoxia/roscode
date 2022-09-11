# CMP-AMR-2021 Assignment

## Explanation

The robot initially moves to the centre of a tile, then proceeds to scan for hints.
If it finds one, it turns toward it – whether it does or not, it then starts scanning square by square.
It uses a mix of initial biases and hints to produce direction scores, then proceeds in the best direction.
If it’s too close to a wall, it will back up.
It treats green highly, and red the same as a wall (when it’s directly in front of it).
It stops upon reaching the green tile.

## Getting the bot going

This section will go over how to get the robot moving through the mazes.
You will need a [Catkin workspace](https://github.com/LCAS/teaching/wiki/First-Turtlebot-coding#setting-up-your-local-ros-workspace) set up before you continue.

It will need the following message types:

- geometry_msgs
- nav_msgs
- sensor_msgs
- std_msgs

### Running the robot's code

In order to run the robot, you need to execute three commands.
These should *all* be in separate tabs.

The first is the Gazebo simulation (# represents a number between 1 and 3 inclusive):

```bash
roslaunch uol_turtlebot_simulator maze#.launch
```

The next is the talker script.
By default, this does not show the robot's view, but you can pass the `--view` flag to enable that:

```bash
cd catkin_ws/src/path/to/scripts
python talker.py
# or...
python talker.py --view
```

Finally, you can run the listener script to get the robot to move.
You can skip the initial scanning phase by passing the `--skip-scan` flag, but you should only use this for debugging:

```bash
python listener.py
# or...
python listener.py --skip-scan
```
