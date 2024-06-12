import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32,fc3_units=16,fc4_units=8):
        
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

class DQNAgent:
    def __init__(self, observation_space, action_space, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, buffer_size=10000):
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(observation_space, action_space).to(self.device)
        self.target_dqn = DQN(observation_space, action_space).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.replay_buffer = []

    def flatten_state(self, state):
        flattened = []
        for key, value in state.items():
            if isinstance(value, (list, np.ndarray)):
                flattened.extend(np.array(value).flatten())
            else:
                flattened.append(value)
        return np.array(flattened, dtype=np.float32)

    def select_action(self, state):
        state = self.flatten_state(state)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.dqn(state)
                return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        state = self.flatten_state(state)
        next_state = self.flatten_state(next_state)
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
            
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = random.sample(self.replay_buffer, self.batch_size)
        batch = [np.array(e) for e in zip(*transitions)]
        
        states = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch[1], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch[3], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device)
        
        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.functional.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
