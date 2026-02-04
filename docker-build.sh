#!/bin/bash

#script for Space Robotics and AI Docker environment

set -e
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    docker-compose build
elif docker compose version &> /dev/null; then
    docker compose build
else
    docker build -t space-robotics-ai:humble .
fi
