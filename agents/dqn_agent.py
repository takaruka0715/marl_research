import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from .network import DuelingDQN

class DQNAgent:
    """DQN エージェント"""
    def __init__(self, state_dim, action_dim, lr=0.0001, shared_buffer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        self.use_shared_buffer = shared_buffer is not None
        if self.use_shared_buffer:
            self.memory = shared_buffer
        else:
            self.memory = deque(maxlen=50000)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.batch_size = 128
        self.update_counter = 0
    
    def select_action(self, state):
        """ε-greedy 行動選択"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """経験を保存"""
        experience = (state, action, reward, next_state, done)
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
        """モデル学習ステップ"""
        if len(self.memory) < self.batch_size:
            return 0
        
        if self.use_shared_buffer:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.scheduler.step()
        
        return loss.item()
    
    def update_target_network(self):
        """ターゲットネットワーク更新"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """探索率低下"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
