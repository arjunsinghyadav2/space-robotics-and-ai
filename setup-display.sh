#!/bin/bash
#X11 display for GUI applications; Set DISPLAY for current session
export DISPLAY=host.docker.internal:0

# Add to bashrc for future sessions
if ! grep -q "DISPLAY=host.docker.internal:0" ~/.bashrc; then
    echo 'export DISPLAY=host.docker.internal:0' >> ~/.bashrc
fi

echo "Current DISPLAY: $DISPLAY"


#if X11 is working
if command -v xclock &> /dev/null; then
    echo "Running xclock test (close the window that appears)..."
    timeout 3 xclock 2>/dev/null && echo "✓ X11 is working!" || echo "⚠ X11 test inconclusive"
else
    echo "Installing x11-apps for testing..."
    apt-get update -qq && apt-get install -y -qq x11-apps
    echo "Try running: xclock"
fi
