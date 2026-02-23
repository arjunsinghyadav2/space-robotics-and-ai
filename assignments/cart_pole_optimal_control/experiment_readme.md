### LQR Controller Tuning Reflection
Starting Point - The Default Parameters
The assignment provides a baseline controller with:
Q = diag([1.0, 1.0, 10.0, 10.0])  R = 0.1
These are arbitrary round numbers. They say nothing about the physical system - a 1m cart error is penalized equally to a 1 rad/s angular velocity error, which makes no physical sense. This was the first problem to address.

#### Iteration 1 - Uniform Scale Sweep (6,727 runs via GPU)
What I did: 
Scaled the entire Q matrix up and down uniformly (q_scale 0.4-2.2) while varying R (r_scale 0.2-1.8).

What I found:
100% pass rate across all combinations - the system is robust
Best config: highest Q scale (2.2) + lowest R (0.2) - cart=0.32m, theta=5.9Deg
The counter-intuitive result: more aggressive Q + less R penalty - less actual control effort spent (23.3N-20.6N)
Why that happens: A lazy controller (low Q, high R) lets deviations grow, then needs large catch-up forces. An aggressive controller (high Q, low R) corrects immediately before deviations build, so the corrections are always small.

What it missed:
This sweep only scaled Q uniformly. It never asked: what if I care more about the pole angle than the cart position, or vice versa? That's the competing-objective trade-off the assignment specifically asks for.

#### Iteration 2 - Competing Objectives Sweep (6,727 runs)
What I did:
 Independently scaled the cart states [x, x_dot] and angle states [theta, theta_dot] to reveal the trade-off.

What I found at fixed r_scale=1.0:
Priority	Q_angle / Q_cart	Cart displacement	Pole angle
Cart-heavy (ratio=0.18)	0.18	0.81m	19.3Deg
Balanced (ratio=1.0)	1.0	0.76m	16.6deg
Angle-heavy (ratio=5.5)	5.5	1.04m	14.8Deg
The trade-off is real: prioritizing pole angle lets the cart drift; prioritizing cart lets the pole tilt. However, at high overall Q scale (2.2, 2.2), the trade-off disappears - the controller is aggressive enough to protect both simultaneously. The ratio only matters when total gain is low.

assignment connection: This directly addresses the requirement to analyze how Q matrix weights affect different states and document trade-offs between objectives.

#### Iteration 3 - Bryson's Rule as Baseline Major Failure
What I tried: Replace the arbitrary Q baseline with Bryson's Rule - the principled method:


Q[i] = 1 / (max_acceptable_deviation[i])**2

With: x_max=2.5m, x_dot_max=5m/s, theta_max=0.3rad, theta_dot_max=2rad/s
Q = [0.16, 0.04, 11.11, 0.25]
What happened: 96% failure rate. Even the best config had cart=1.94m and theta=41Deg - barely surviving.

Why it failed: Bryson's rule hardcoded q_theta / q_x = 69.4x. The controller was so focused on the pole angle that it ignored cart drift. The earthquake kept pushing the cart sideways and the controller barely noticed until it was near the wall.

The lesson: Bryson's Rule is only as good as the tolerances you feed it. Choosing theta_max = 0.3 rad (17Deg) and x_max = 2.5 m (the full physical range) implicitly told the controller that the pole falling is 69 times more catastrophic per unit than the cart drifting. In an earthquake scenario where both are attacked simultaneously, that is wrong.

#### Iteration 4: Direct Bryson Tolerance Sweep 12,096 runs
What I did: 
Instead of picking one set of tolerances, sweep them directly. Let the data tell you which tolerances work.

x_tol: 0.5m - 2.5m  (8 values)
xdot_tol: 1.0 - 5.0 m/s (6 values)
theta_tol: 0.20 - 0.50 rad (6 values)
thetadot_tol: 0.5 - 2.8 rad/s (6 values)
r_scale: 0.5 - 1.8  (7 values)

What I found:
x_tol	Pass rate
0.5m (tight)	21%
2.5m (loose)	8%
Tightening x_tol is the most powerful lever. Setting x_tol=0.5m raises Q[x] from 0.16 to 4.0 - the controller fights cart drift 25× harder than with the naive Bryson baseline.

#### Best configuration found:
x_tol=0.5m, xdot_tol=1.0, theta_tol=0.2rad, thetadot_tol=0.5
- Q = [4.0, 1.0, 25.0, 4.0]  R = 0.05
  q_theta / q_x = 6.25x  (vs 69x from naive Bryson)
Final performance:

Survival: 400s (full duration)
Max cart displacement: 1.41m (within ±2.5m limit)
Max pole angle: 31.9Deg (within 45Deg limit)
Avg control effort: 23.9N
Key Learnings
The Q ratio matters more than the Q magnitude.
The ratio q_theta / q_x determines what the controller defends. The naive Bryson ratio of 69x destroys cart stability. The swept optimum of ~6x balances both objectives under earthquake disturbance.
Tight tolerances = aggressive controller = better performance.
Counterintuitively, telling the controller keep the cart within 0.5m (even though it physically can't) forces high Q[x] that actually reduces observed cart displacement from 1.9m to 1.4m.

Low R is consistently better.
Across all sweeps, lower R (less penalty on control effort) always wins. The system has a 15N earthquake and needs an aggressive response. Penalizing force usage hurts more than it saves.
Bryson's Rule requires physical insight, not just physical limits.
Setting tolerances to the hardware limits (x_max=2.5m) gives a controller calibrated for don't hit the wall - too relaxed for disturbance rejection. The tolerances should reflect desired performance, not just survival boundaries. The score function shapes what best means. 
score = survival - 0.5·cart - 0.02·theta - 0.03·effort
Cart displacement is penalized 25× more than theta per degree/meter. This is why tight x_tol matters most - the score function rewards reducing cart displacement the most.

#### Final Parameters
Q = diag([4.0, 1.0, 25.0, 4.0])  # x_tol=0.5m, theta_tol=0.2rad (Bryson-derived)
R = 0.05, r_scale=0.5 - aggressive control
These were selected through a systematic GPU-accelerated sweep of 12,096 Bryson-tolerance combinations, validated across 3 random earthquake seeds, choosing the configuration with the highest worst-case score.