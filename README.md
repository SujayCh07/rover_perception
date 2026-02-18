🚀 Lunar Rover Perception – Height-Based Costmap (ROS 2 Jazzy)

End-to-end perception MVP for a lunar rover:

ZED Stereo Camera
    ↓
PointCloud2
    ↓
TF2 Transform → odom
    ↓
Height-based 2D Grid
    ↓
nav_msgs/OccupancyGrid
    ↓
Nav2 / RViz2


This project implements a minimal, readable, and modular height-threshold costmap pipeline.

📦 Package Structure
rover_perception/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
├── rover_perception/
│   ├── cloud_to_target_frame.py
│   ├── height_costmap.py
│   ├── mock_cloud_pub.py
└── README.md

🖥 System Requirements

Ubuntu 22.04

ROS 2 Jazzy

ZED ROS2 Wrapper

RViz2

TF2

NVIDIA Jetson (Orin / Thor class recommended)

Install ROS Jazzy if needed:

sudo apt install ros-jazzy-desktop


Install RViz if missing:

sudo apt install ros-jazzy-rviz2

🔧 Setup Instructions
1️⃣ Create Workspace
mkdir -p ~/rover_ws/src
cd ~/rover_ws/src
git clone https://github.com/YOUR_USERNAME/rover_perception.git

2️⃣ Build
cd ~/rover_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash


Add sourcing to your ~/.bashrc for convenience if desired.

🎥 Running the Pipeline

You need 3 terminals.

In every new terminal:

source /opt/ros/jazzy/setup.bash
source ~/rover_ws/install/setup.bash

🟢 Terminal 1 – Launch ZED
ros2 launch zed_wrapper zed_camera.launch.py


Verify cloud exists:

ros2 topic list | grep cloud


You should see:

/zed/zed_node/point_cloud/cloud_registered

🔵 Terminal 2 – Transform Cloud to Odom
ros2 run rover_perception cloud_to_target_frame \
--ros-args \
-r /point_cloud/cloud_registered:=/zed/zed_node/point_cloud/cloud_registered


This transforms:

zed_left_camera_frame → odom


Verify:

ros2 topic hz /cloud_in_target_frame

🟣 Terminal 3 – Run Height Costmap
ros2 run rover_perception height_costmap


Verify:

ros2 topic hz /height_costmap

👁 Viewing in RViz2

Launch:

rviz2

Set:

Fixed Frame → odom

Add Displays:

PointCloud2 → /cloud_in_target_frame

Map → /height_costmap

TF

Grid (optional)

If cloud does not appear:

Change Reliability to Best Effort in PointCloud2 settings.

🧠 How It Works
cloud_to_target_frame.py

Subscribes to ZED PointCloud2

Uses TF2 to transform into odom

Publishes /cloud_in_target_frame

height_costmap.py

Subscribes to /cloud_in_target_frame

Discretizes XY into grid

Stores max Z per cell

Applies simple height threshold:

if cell_height > threshold:
    occupied = 100
else:
    free = 0


Publishes nav_msgs/OccupancyGrid

📐 Frame Conventions

ZED optical frame:

X right
Y down
Z forward


Target frame:

odom


All costmap calculations are performed in the target frame.

🧪 Debugging
Check cloud publishing
ros2 topic hz /zed/zed_node/point_cloud/cloud_registered

Check transform
ros2 run tf2_ros tf2_echo odom zed_left_camera_frame

Check output topics
ros2 topic info /cloud_in_target_frame -v
ros2 topic info /height_costmap -v

📈 Current Status

✅ ZED cloud publishing
✅ TF transform working
✅ Real-time RViz visualization
✅ Height threshold costmap publishing
