FROM osrf/ros:humble-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV WORKSPACE=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
    vim \
    nano \
    # ROS2 packages for Assignment 1 (Boustrophedon)
    ros-humble-turtlesim \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-reconfigure \
    # ROS2 packages for Assignment 2 (Cart-Pole)
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-interfaces \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    ros-humble-xacro \
    # ROS2 packages for Assignment 3 (Drone)
    ros-humble-tf2-ros \
    ros-humble-cv-bridge \
    # Additional tools
    python3-opencv \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo Garden (for Ubuntu 22.04)
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && \
    apt-get install -y gz-garden && \
    rm -rf /var/lib/apt/lists/*

# Update rosdep (init already done in base image)
RUN rosdep update

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    control \
    transforms3d \
    torch \
    torchvision \
    gymnasium

# Create workspace structure
WORKDIR ${WORKSPACE}
RUN mkdir -p ${WORKSPACE}/src

# Set up environment sourcing
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> ~/.bashrc && \
    echo "export PX4_AUTOPILOT_PATH=~/PX4-Autopilot" >> ~/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=\${GZ_SIM_RESOURCE_PATH}:${WORKSPACE}/src/terrain_mapping_drone_control/models" >> ~/.bashrc

# RUN cd ${WORKSPACE}/src && \
#     git clone https://github.com/PX4/px4_msgs.git && \
#     cd ${WORKSPACE} && \
#     . /opt/ros/${ROS_DISTRO}/setup.sh && \
#     colcon build --packages-select px4_msgs

# RUN apt-get update && apt-get install -y \
#     libsqlite3-dev \
#     libpcl-dev \
#     libopencv-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# RUN cd ${WORKSPACE}/src && \
#     git clone https://github.com/introlab/rtabmap.git && \
#     cd rtabmap/build && \
#     cmake .. && \
#     make -j$(nproc) && \
#     make install && \
#     ldconfig

# RUN cd ${WORKSPACE}/src && \
#     git clone https://github.com/introlab/rtabmap_ros.git -b humble-devel && \
#     cd ${WORKSPACE} && \
#     . /opt/ros/${ROS_DISTRO}/setup.sh && \
#     colcon build --packages-select rtabmap_ros

# RUN cd ~ && \
#     git clone https://github.com/PX4/PX4-Autopilot.git && \
#     cd PX4-Autopilot && \
#     git checkout 9ac03f03eb && \
#     bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools && \
#     make px4_sitl gz_x500

# Set working directory to workspace src
WORKDIR ${WORKSPACE}/src
# Welcome message script
RUN echo '#!/bin/bash\n\
echo "Workspace: ${WORKSPACE}"\n\
echo "ROS Distribution: ${ROS_DISTRO}"\n\
' > /usr/local/bin/welcome.sh && \
    chmod +x /usr/local/bin/welcome.sh && \
    echo "/usr/local/bin/welcome.sh" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
