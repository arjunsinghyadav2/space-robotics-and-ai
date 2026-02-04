#!/bin/bash
#line endings fix for all assignments windows issue

# Install dos2unix if not available
if ! command -v dos2unix &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq dos2unix
fi
# Convert line endings from CRLF to LF for .py and .sh files
find /workspace/src -name "*.py" -type f -exec dos2unix {} \; 2>/dev/null
find /workspace/src -name "*.sh" -type f -exec dos2unix {} \; 2>/dev/null

find /workspace/src -name "*.py" -type f -exec chmod +x {} \;
find /workspace/src -name "*.sh" -type f -exec chmod +x {} \;
