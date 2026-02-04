#!/bin/bash

# Run script for Space Robotics and AI Docker environment

set -e

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Enable X11 forwarding on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xhost +local:docker 2>/dev/null || echo "Warning: Could not enable X11 forwarding"
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "Starting container with docker-compose..."
    docker-compose up -d
    docker-compose exec space-robotics-dev bash
elif docker compose version &> /dev/null; then
    docker compose up -d
    echo "Container started successfully!"
    docker compose exec space-robotics-dev bash
else
    echo "docker-compose not found, using docker run..."

    # Detect OS and set display accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        DISPLAY_VAR="${DISPLAY}"
        X11_MOUNT="-v /tmp/.X11-unix:/tmp/.X11-unix:rw"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        DISPLAY_VAR="host.docker.internal:0"
        X11_MOUNT=""
    else
        DISPLAY_VAR="host.docker.internal:0"
        X11_MOUNT=""
    fi

    docker run -it --rm \
        --name srai-workspace \
        --network host \
        -e DISPLAY="${DISPLAY_VAR}" \
        -e QT_X11_NO_MITSHM=1 \
        ${X11_MOUNT} \
        -v "$(pwd)/assignments:/workspace/src:rw" \
        space-robotics-ai:humble
fi
