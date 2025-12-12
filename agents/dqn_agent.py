import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from .network import DuelingDQN
from .vdn import VDNNetwork, VDNTargetNetwork

class DQNAgent:
    """独立DQNエージェント"""
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


class VDNAgent:
    """Value Decomposition Networks (VDN) を用いたマルチエージェント"""
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
    
    def select_actions(self, states_dict):
        """
        全エージェントの行動を選択
        Args:
            states_dict: Dict[agent_name, state] (各エージェントの状態)
        Returns:
            Dict[agent_name, action]
        """
        actions = {}
        
        for agent_name, state in states_dict.items():
            if random.random() < self.epsilon:
                actions[agent_name] = random.randint(0, self.action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    # エージェントのインデックスを取得（agent_0 -> 0, agent_1 -> 1）
                    agent_idx = int(agent_name.split('_')[1])
                    q_local = self.q_network.local_q_networks[agent_idx](state_tensor)
                    actions[agent_name] = q_local.argmax().item()
        
        return actions
    
    def store_transition(self, state_dict, action_dict, reward_dict, next_state_dict, done_dict):
        """
        マルチエージェント経験を保存
        Args:
            state_dict: Dict[agent_name, state]
            action_dict: Dict[agent_name, action]
            reward_dict: Dict[agent_name, reward]
            next_state_dict: Dict[agent_name, next_state]
            done_dict: Dict[agent_name, done]
        """
        experience = (state_dict, action_dict, reward_dict, next_state_dict, done_dict)
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
        """VDN モデル学習ステップ"""
        if len(self.memory) < self.batch_size:
            return 0
        
        if self.use_shared_buffer:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        state_dicts, action_dicts, reward_dicts, next_state_dicts, done_dicts = zip(*batch)
        
        # バッチ整形（各エージェントごとにテンソル化）
        agent_names = list(state_dicts[0].keys())
        
        states_by_agent = {agent: [] for agent in agent_names}
        actions_by_agent = {agent: [] for agent in agent_names}
        rewards_by_agent = {agent: [] for agent in agent_names}
        next_states_by_agent = {agent: [] for agent in agent_names}
        dones_by_agent = {agent: [] for agent in agent_names}
        
        for state_dict, action_dict, reward_dict, next_state_dict, done_dict in zip(
            state_dicts, action_dicts, reward_dicts, next_state_dicts, done_dicts):
            for agent in agent_names:
                states_by_agent[agent].append(state_dict[agent])
                actions_by_agent[agent].append(action_dict[agent])
                rewards_by_agent[agent].append(reward_dict[agent])
                next_states_by_agent[agent].append(next_state_dict[agent])
                dones_by_agent[agent].append(done_dict[agent])
        
        # テンソル化
        for agent in agent_names:
            states_by_agent[agent] = torch.FloatTensor(np.array(states_by_agent[agent])).to(self.device)
            actions_by_agent[agent] = torch.LongTensor(actions_by_agent[agent]).to(self.device)
            rewards_by_agent[agent] = torch.FloatTensor(rewards_by_agent[agent]).to(self.device)
            next_states_by_agent[agent] = torch.FloatTensor(np.array(next_states_by_agent[agent])).to(self.device)
            dones_by_agent[agent] = torch.FloatTensor(dones_by_agent[agent]).to(self.device)
        
        # VDN による Q 値計算
        states_list = [states_by_agent[agent] for agent in agent_names]
        next_states_list = [next_states_by_agent[agent] for agent in agent_names]
        
        # 現在のQ値（全体）
        q_tot, q_locals = self.q_network(states_list)
        
        # 各エージェントの Q 値選択
        q_selected = torch.zeros(self.batch_size).to(self.device)
        for i, agent in enumerate(agent_names):
            q_selected += q_locals[i].gather(1, actions_by_agent[agent].unsqueeze(1)).squeeze()
        
        # ターゲットQ値計算
        with torch.no_grad():
            q_tot_next, _ = self.target_network(next_states_list)
            q_target = torch.zeros(self.batch_size).to(self.device)
            
            for i, agent in enumerate(agent_names):
                q_target += rewards_by_agent[agent]
                q_target += (1 - dones_by_agent[agent]) * self.gamma * q_tot_next.max(1)[0]
        
        # 損失計算
        loss = nn.SmoothL1Loss()(q_selected, q_target)
        
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
