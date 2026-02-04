#!/bin/bash
cd /workspace
colcon build --packages-select first_order_boustrophedon_navigator --symlink-install
sed -i 's/\r$//' /workspace/install/first_order_boustrophedon_navigator/lib/first_order_boustrophedon_navigator/boustrophedon_controller
source install/setup.bash
echo "Build complete"
