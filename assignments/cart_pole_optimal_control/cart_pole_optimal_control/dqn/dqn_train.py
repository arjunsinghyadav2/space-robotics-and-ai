from dqn_agent import DQNAgent
import gymnasium as gym
import numpy as np
import random
import torch
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")

# Initialize environment
env = gym.make("CartPole-v1")
# Aligned RL termination limits with LQR assignment limits for fair comparison.
env.unwrapped.x_threshold = 2.5
env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
state_dim = env.observation_space.shape[0] + 1  # Now 5 states (4 original + 1 earthquake force)
action_dim = env.action_space.n

# Initialize DQN Agent (decay is per-episode, not per-step)
agent = DQNAgent(state_dim, action_dim, lr=5e-4, decay=0.9995, min_epsilon=0.05)

# Earthquake Force Parameters
num_waves = 5
freq_range = [0.5, 4.0]  # Frequency range in Hz
base_amplitude = 15  # Base force amplitude in N
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

# Training Loop
num_episodes = 20000
total_rewards = []
steps_per_episode = []
epsilon_values = []
_root = Path(__file__).resolve().parent
model_filename = _root / "dqn_cartpole_earthquake.pth"
log_filename = _root / "dqn_training_log.csv"

time_step = 0
for episode in range(1, num_episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for t in range(1000):
        earthquake_force = generate_earthquake_force(time_step * env_timestep)

        # Append earthquake force to the state
        state_with_force = np.append(state, earthquake_force)  

        action = agent.select_action(state_with_force, evaluate=False)
        next_state, _, terminated, truncated, _ = env.step(action)
        # Apply earthquake as external acceleration on cart velocity.
        # Position integrates naturally on the next env.step().
        dyn_state = env.unwrapped.state
        total_mass = env.unwrapped.masscart + env.unwrapped.masspole
        dyn_state[1] += (earthquake_force / total_mass) * env_timestep
        env.unwrapped.state = dyn_state
        next_state = np.array(dyn_state, dtype=np.float32)

        violated = (
            abs(next_state[0]) > env.unwrapped.x_threshold
            or abs(next_state[2]) > env.unwrapped.theta_threshold_radians
        )

        done = terminated or truncated or violated

        # **Custom Reward Function**
        if done:
            reward = -10.0  # Strong death penalty
        else:
            alive_reward = 1.0
            angle_bonus = 0.5 * (1.0 - abs(next_state[2]) / env.unwrapped.theta_threshold_radians)
            position_bonus = 0.5 * (1.0 - abs(next_state[0]) / env.unwrapped.x_threshold)
            velocity_penalty = -0.1 * abs(next_state[3])  # Discourage oscillation
            reward = alive_reward + angle_bonus + position_bonus + velocity_penalty

        # Append earthquake force to the next state
        next_state_with_force = np.append(next_state, earthquake_force)

        agent.store_transition(state_with_force, action, reward, next_state_with_force, done)
        if time_step % 4 == 0:
            agent.train()

        state = next_state
        time_step += 1
        total_reward += reward
        steps += 1

        if done:
            break

    agent.decay_epsilon()
    agent.update_target_model()
    total_rewards.append(total_reward)
    steps_per_episode.append(steps)
    epsilon_values.append(agent.epsilon) 
    print(f"Episode {episode}, Total Reward: {total_reward:.4f}, Steps: {steps}, Exploration Rate (ε): {agent.epsilon:.6f}")

# Save model at the end of training
agent.save_model(str(model_filename))

with open(log_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward", "steps", "epsilon"])
    for i, (r, s, e) in enumerate(zip(total_rewards, steps_per_episode, epsilon_values), start=1):
        writer.writerow([i, f"{r:.6f}", s, f"{e:.6f}"])
print(f"Training log saved as {log_filename}")

env.close()

# Plot Training Performance
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(total_rewards, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(steps_per_episode, label="Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epsilon_values, label="Exploration Rate (ε)")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()

plt.tight_layout()
plt.savefig(_root / "dqn_training_curve.png", dpi=100)
print(f"Training curve saved to {_root / 'dqn_training_curve.png'}")
