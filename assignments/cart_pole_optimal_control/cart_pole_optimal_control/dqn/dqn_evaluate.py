from dqn_agent import DQNAgent
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
import json
from pathlib import Path

# Initialize environment (headless by default).
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0] + 1  # 4 original + earthquake force
action_dim = env.action_space.n

# Load trained model
agent = DQNAgent(state_dim, action_dim)
_root = Path(__file__).resolve().parent
model_filename = _root / "dqn_cartpole_earthquake.pth"
agent.q_network.load_state_dict(
    torch.load(str(model_filename), map_location=torch.device("cpu"), weights_only=True)
)
agent.q_network.eval()

# Earthquake Force Parameters
num_waves = 5
freq_range = [0.5, 4.0]  # Frequency range in Hz
base_amplitude = 0  # Base force amplitude in N
env_timestep = 0.02  # Default time step for CartPole environment
frequencies = np.random.uniform(freq_range[0], freq_range[1], num_waves)
phase_shifts = np.random.uniform(0, 2 * np.pi, num_waves)

def generate_earthquake_force(time):
    """Generate earthquake-like force using superposition of sine waves."""
    force = 0.0
    for freq, phase in zip(frequencies, phase_shifts):
        amplitude = base_amplitude * np.random.uniform(0.8, 1.2)
        force += amplitude * np.sin(2 * np.pi * freq * time + phase)
    force += np.random.normal(0, base_amplitude * 0.1)  # Add noise
    return force

# Run evaluation episodes
num_episodes = 10
show_plots = False
episode_rows = []
for episode in range(1, num_episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    time_step = 0
    angle_deviation = deque(maxlen=1000)
    cart_position = deque(maxlen=1000)
    control_effort = deque(maxlen=1000)
    earthquake_force = deque(maxlen=1000)

    for t in range(1000):
        earthquake = generate_earthquake_force(time_step * env_timestep)  # Store in a separate variable
        state_with_force = np.append(state[:4], earthquake)
        action = agent.select_action(state_with_force, evaluate=True)
        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Apply earthquake as external acceleration on cart velocity.
        dyn_state = env.unwrapped.state
        total_mass = env.unwrapped.masscart + env.unwrapped.masspole
        dyn_state[1] += (earthquake / total_mass) * env_timestep
        env.unwrapped.state = dyn_state
        next_state = np.array(dyn_state, dtype=np.float32)

        violated = (
            abs(next_state[0]) > env.unwrapped.x_threshold
            or abs(next_state[2]) > env.unwrapped.theta_threshold_radians
        )
        done = terminated or truncated or violated

        cart_x = abs(next_state[0])
        pole_theta = abs(next_state[2])
        cart_position.append(cart_x)
        angle_deviation.append(pole_theta)
        control_effort.append(10 if action == 1 else -10)
        earthquake_force.append(earthquake)

        state = next_state
        time_step += 1
        total_reward += reward

        if done:
            break

    print(f"Evaluation Episode {episode}, Total Reward: {total_reward}")
    episode_rows.append(
        {
            "episode": episode,
            "survival_steps": time_step,
            "total_reward": float(total_reward),
            "max_cart_m": float(max(cart_position) if len(cart_position) > 0 else 0.0),
            "max_theta_rad": float(max(angle_deviation) if len(angle_deviation) > 0 else 0.0),
            "avg_abs_u": float(np.mean(np.abs(control_effort)) if len(control_effort) > 0 else 0.0),
})

    if show_plots:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(cart_position, label="Cart Position (m)", color="blue")
        axs[0, 0].set_title("Cart Position")
        axs[0, 0].legend()

        axs[0, 1].plot(angle_deviation, label="Pole Angle Deviation (rad)", color="red")
        axs[0, 1].set_title("Pole Angle Deviation")
        axs[0, 1].legend()

        axs[1, 0].plot(earthquake_force, label="Earthquake Force (N)", color="green")
        axs[1, 0].set_title("Earthquake Disturbance")
        axs[1, 0].legend()

        axs[1, 1].plot(control_effort, label="Control Force (N)", color="magenta")
        axs[1, 1].set_title("Control Effort")
        axs[1, 1].legend()
        plt.tight_layout()
        plt.show()

env.close()

with open(_root / "dqn_eval_episodes.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["episode", "survival_steps", "total_reward", "max_cart_m", "max_theta_rad", "avg_abs_u"],
    )
    writer.writeheader()
    writer.writerows(episode_rows)

summary = {
    "episodes": num_episodes,
    "pass_rate": float(np.mean([r["survival_steps"] >= 500 for r in episode_rows])) if episode_rows else 0.0,
    "survival_steps_mean": float(np.mean([r["survival_steps"] for r in episode_rows])) if episode_rows else 0.0,
    "reward_mean": float(np.mean([r["total_reward"] for r in episode_rows])) if episode_rows else 0.0,
    "max_cart_mean_m": float(np.mean([r["max_cart_m"] for r in episode_rows])) if episode_rows else 0.0,
    "max_theta_mean_rad": float(np.mean([r["max_theta_rad"] for r in episode_rows])) if episode_rows else 0.0,
    "avg_abs_u_mean": float(np.mean([r["avg_abs_u"] for r in episode_rows])) if episode_rows else 0.0,
}
with open(_root / "dqn_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
