# ================================================
# FILE: agents/dqn_agent.py
# ================================================
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from .network import DuelingDQN
from .vdn import VDNNetwork, VDNTargetNetwork

class DQNAgent:
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

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
        
    def _apply_action_mask(self, q_values, states):
        dx = states[:, 50] * 15.0
        dy = states[:, 51] * 15.0
        dist = torch.abs(dx.round()) + torch.abs(dy.round())
        invalid_mask = dist > 1.1
        q_values[invalid_mask, 4:] = -1e9
        return q_values

    def select_action(self, state, **kwargs):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            dummy_q = torch.zeros(1, self.action_dim).to(self.device)
            masked_q = self._apply_action_mask(dummy_q, state_tensor)
            valid_actions = (masked_q[0] > -1e8).nonzero(as_tuple=True)[0].tolist()
            return random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = self._apply_action_mask(q_values, state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
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
            next_actions = self.q_network(next_states)
            next_actions = self._apply_action_mask(next_actions, next_states).max(1)[1]
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
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class VDNAgent:
    def __init__(self, state_dim, action_dim, num_agents=2, lr=0.0001, shared_buffer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        self.q_network = VDNNetwork(state_dim, action_dim, num_agents).to(self.device)
        self.target_network = VDNTargetNetwork(state_dim, action_dim, num_agents).to(self.device)
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

    def _apply_action_mask(self, q_values, states):
        dx = states[:, 50] * 15.0
        dy = states[:, 51] * 15.0
        dist = torch.abs(dx.round()) + torch.abs(dy.round())
        invalid_mask = dist > 1.1
        q_values[invalid_mask, 4:] = -1e9
        return q_values
    
    def select_actions(self, states_dict, **kwargs):
        actions = {}
        for agent_name, state in states_dict.items():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if random.random() < self.epsilon:
                dummy_q = torch.zeros(1, self.action_dim).to(self.device)
                masked_q = self._apply_action_mask(dummy_q, state_tensor)
                valid_actions = (masked_q[0] > -1e8).nonzero(as_tuple=True)[0].tolist()
                actions[agent_name] = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    agent_idx = int(agent_name.split('_')[1])
                    q_local = self.q_network.get_local_q(agent_idx, state_tensor)
                    q_local = self._apply_action_mask(q_local, state_tensor)
                    actions[agent_name] = q_local.argmax().item()
        return actions
    
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
    
    def store_transition(self, state_dict, action_dict, reward_dict, next_state_dict, done_dict):
        experience = (state_dict, action_dict, reward_dict, next_state_dict, done_dict)
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        if self.use_shared_buffer:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        state_dicts, action_dicts, reward_dicts, next_state_dicts, done_dicts = zip(*batch)
        agent_names = list(state_dicts[0].keys())
        
        states_by_agent = {}
        actions_by_agent = {}
        rewards_by_agent = {}
        next_states_by_agent = {}
        dones_by_agent = {}

        for agent in agent_names:
            states_by_agent[agent] = torch.FloatTensor(np.array([d[agent] for d in state_dicts])).to(self.device)
            actions_by_agent[agent] = torch.LongTensor([d[agent] for d in action_dicts]).to(self.device)
            rewards_by_agent[agent] = torch.FloatTensor([d[agent] for d in reward_dicts]).to(self.device)
            next_states_by_agent[agent] = torch.FloatTensor(np.array([d[agent] for d in next_state_dicts])).to(self.device)
            dones_by_agent[agent] = torch.FloatTensor([d[agent] for d in done_dicts]).to(self.device)

        states_list = [states_by_agent[agent] for agent in agent_names]
        q_locals = self.q_network(states_list)
        
        q_selected_sum = torch.zeros(self.batch_size).to(self.device)
        for i, agent in enumerate(agent_names):
            q_val = q_locals[i].gather(1, actions_by_agent[agent].unsqueeze(1)).squeeze()
            q_selected_sum += q_val
            
        next_states_list = [next_states_by_agent[agent] for agent in agent_names]
        
        with torch.no_grad():
            q_locals_next = self.target_network(next_states_list)
            q_next_max_sum = torch.zeros(self.batch_size).to(self.device)
            
            for i in range(self.num_agents):
                # ★ターゲットQ値にもマスクを適用
                q_locals_next[i] = self._apply_action_mask(q_locals_next[i], next_states_list[i])
                q_next_max_sum += q_locals_next[i].max(1)[0]
            
            total_rewards = torch.zeros(self.batch_size).to(self.device)
            for agent in agent_names:
                total_rewards += rewards_by_agent[agent]
            
            all_dones = torch.stack(list(dones_by_agent.values())).min(dim=0)[0] 
            q_target = total_rewards + (1 - all_dones) * self.gamma * q_next_max_sum

        loss = nn.SmoothL1Loss()(q_selected_sum, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.scheduler.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)