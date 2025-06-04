# DQN pour optimiser la consommation énergétique d'un bâtiment

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Exemple : charge un modèle Keras ou Sklearn
# from tensorflow.keras.models import load_model
# model = load_model("modele_conso.h5")
# def model_predict(state):
#     return model.predict(np.array(state).reshape(1, -1))[0]

def model_predict(state):
    # Exemple factice
    heating = 20 * (1 - state[1]) + 5 * state[0]
    cooling = 10 * (state[3]) + 5 * (1 - state[2])
    return np.array([heating, cooling])

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (torch.FloatTensor(s), torch.LongTensor(a), torch.FloatTensor(r), torch.FloatTensor(s2), torch.FloatTensor(d))
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, model_predict, gamma=0.99, lr=1e-3):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.model_predict = model_predict

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 9)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return int(self.q_net(state_tensor).argmax().item())

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        s, a, r, s2, d = self.replay_buffer.sample(self.batch_size)
        q_vals = self.q_net(s)
        target_q_vals = self.target_net(s2).max(1)[0]
        targets = r + self.gamma * target_q_vals * (1 - d)
        q_vals_action = q_vals.gather(1, a.unsqueeze(1)).squeeze()
        loss = (q_vals_action - targets.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def apply_action(state, action_index):
    step = 0.01
    state = state.copy()
    param_idx = action_index // 2
    direction = 1 if action_index % 2 == 0 else -1
    state[param_idx] += direction * step
    return np.clip(state, 0, 1)

agent = DQNAgent(state_dim=5, action_dim=10, model_predict=model_predict)

n_episodes = 200
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
best_designs = []

for episode in range(n_episodes):
    state = np.random.uniform(0, 1, size=5)
    total_reward = 0
    for step in range(50):
        action = agent.select_action(state, epsilon)
        next_state = apply_action(state, action)
        energy = model_predict(next_state)
        reward = -np.sum(energy)
        agent.replay_buffer.push(state, action, reward, next_state, False)
        agent.train_step()
        state = next_state
        total_reward += reward

    best_designs.append((total_reward, state.copy(), model_predict(state)))
    best_designs = sorted(best_designs, key=lambda x: x[0], reverse=True)[:50]
    agent.update_target()
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

# Visualisation des meilleurs designs
data = []
for reward, state, (heating, cooling) in best_designs:
    data.append([reward] + list(state) + [heating, cooling])

columns = ["total_reward", "overall_height", "relative_compactness", "wall_area", "glazing_area", "orientation", "heating", "cooling"]
df = pd.DataFrame(data, columns=columns)
print(df.head(10))

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(df["glazing_area"], df["heating"], label="Heating", color='red')
plt.scatter(df["glazing_area"], df["cooling"], label="Cooling", color='blue')
plt.xlabel("Glazing Area")
plt.ylabel("Load")
plt.legend()
plt.title("Impact du Glazing Area")
plt.grid(True)
plt.show()

# Radar plot
categories = ["overall_height", "relative_compactness", "wall_area", "glazing_area", "orientation"]
N = len(categories)

plt.figure(figsize=(8,6))
for i in range(3):
    values = df.loc[i, categories].tolist()
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
    plt.polar(angles, values, label=f"Design {i+1}")
plt.title("Top 3 Designs")
plt.legend()
plt.show()
        