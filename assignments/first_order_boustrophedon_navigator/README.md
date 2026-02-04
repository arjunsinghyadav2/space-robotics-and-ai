# First-Order Boustrophedon Navigator

A PD-controlled lawnmower survey pattern using ROS2 Turtlesim. The turtle executes uniform horizontal sweeps with alternating direction, minimizing cross-track error via tuned PD gains.

## Running with Docker

All dependencies are included in the course Docker image.

```bash
# Start the container
docker-compose up -d
docker-compose exec space-robotics-dev bash

# Inside the container
cd /workspace/src
./rebuild-and-fix.sh
ros2 launch first_order_boustrophedon_navigator boustrophedon.launch.py
```

To tune parameters live, open a second terminal:
```bash
docker-compose exec space-robotics-dev bash
source /workspace/install/setup.bash
ros2 run rqt_reconfigure rqt_reconfigure
```

## What Was Implemented

### 1. Waypoint Generation Fix
The original `generate_waypoints()` used `len(waypoints) % 2` to alternate sweep direction. Since waypoints are always added in pairs, this value is always even the pattern never alternated, causing diagonal returns between every pass. Replaced with a `going_right` boolean flag that toggles each iteration.

### 2. Turn-in-Place Control
The original controller computed linear and angular velocity simultaneously. This caused the turtle to drift forward while turning, producing curved transitions between sweeps. Added a threshold check if `abs(angular_error) > 0.1 rad`, linear velocity is set to zero. The turtle turns in place first, then drives straight.

### 3. Velocity Capping
- Linear velocity clamped to `[0.0, 2.0]` — prevents reversing on waypoint overshoot
- Angular velocity clamped to `[-2.0, 2.0]` — prevents jerky snapping during turns

### 4. PD Controller Tuning
| Parameter | Original | Tuned | Rationale |
|-----------|----------|-------|-----------|
| `Kp_linear` | 10.0 | 2.0 | 10.0 caused overshooting at waypoints. 2.0 gives smooth deceleration |
| `Kd_linear` | 0.1 | 0.3 | Higher damping smooths velocity on straight segments |
| `Kp_angular` | 5.0 | 4.0 | Slightly reduced to lower oscillation at corners |
| `Kd_angular` | 0.2 | 1.0 | 0.2 was critically underdamped. 1.0 gives near-critical damping on turns |
| `spacing` | 1.0 | 1.0 | Unchanged, changing spacing did not provide better coverage or pattern |

## Final Parameters (`config/boustrophedon_params.yaml`)

```yaml
Kp_linear:  2.0
Kd_linear:  0.3
Kp_angular: 4.0
Kd_angular: 1.0
spacing:    1.0
```

## Monitoring Performance

```bash
# Cross-track error values
ros2 topic echo /cross_track_error

# Plot error and velocity live
ros2 run rqt_plot rqt_plot
# Topics: /cross_track_error, /turtle1/cmd_vel/linear/x, /turtle1/cmd_vel/angular/z
```

Plots (cross-track error, trajectory, velocity profiles) are automatically saved as PNG files when the pattern completes.

