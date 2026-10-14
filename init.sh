conda activate legs
# ROS2 Humble comes from RoboStack inside the legs env, so `conda activate legs`
# already sets up the ROS environment — do NOT also source a system ROS install
# built for another Python (e.g. Jazzy on py3.12; this env is py3.10). If you instead
# use a system ROS2 Humble install, re-add: source /opt/ros/humble/setup.bash
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
