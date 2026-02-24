import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Define Neural Network for Q-function
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False):
        super(QNetwork, self).__init__()
        self.continuous = continuous
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim if not continuous else 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) if self.continuous else self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, epsilon=1.0, min_epsilon=0.1, decay=0.9999, continuous=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.batch_size = 256
        self.memory = deque(maxlen=100000)
        self.continuous = continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-Network & Target Network
        self.q_network = QNetwork(state_dim, action_dim, continuous).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, continuous).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        """ Selects action using epsilon-greedy strategy for discrete actions or outputs a continuous value. """
        if not self.continuous:
            if not evaluate and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.q_network(state_tensor)).item()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.q_network(state_tensor).item() * 10  # Scale output for continuous force

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device) if self.continuous else torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q-values and target Q-values
        q_values = self.q_network(states).gather(1, actions.long()) if not self.continuous else self.q_network(states)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0] if not self.continuous else self.target_network(next_states)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

    def decay_epsilon(self):
        """changes to stable exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def update_target_model(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filename="dqn_cartpole_trained.pth"):
        torch.save(self.q_network.state_dict(), filename)
        print(f"Model saved as {filename} (device: {self.device})")

    def load_model(self, filename="dqn_cartpole_trained.pth"):
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.q_network.eval()
        print(f"Model loaded from {filename}")
