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
    """Value Decomposition Networks (VDN) を用いたマルチエージェント (Attention対応版)"""
    def __init__(self, state_dim, action_dim, num_agents=2, lr=0.0001, shared_buffer=None, agent_config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_config = agent_config
        
        # agent_config を渡して初期化 (Attention設定を含むため)
        self.q_network = VDNNetwork(state_dim, action_dim, num_agents, agent_config).to(self.device)
        self.target_network = VDNTargetNetwork(state_dim, action_dim, num_agents, agent_config).to(self.device)
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
    
    def select_actions(self, states_dict, task_features=None):
        """全エージェントの行動を選択"""
        actions = {}
        for agent_name, state in states_dict.items():
            if random.random() < self.epsilon:
                actions[agent_name] = random.randint(0, self.action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    # Task FeatureのTensor化 (Batch=1)
                    tf_tensor = None
                    if task_features is not None:
                         tf_tensor = torch.FloatTensor(task_features).to(self.device)
                    
                    agent_idx = int(agent_name.split('_')[1])
                    # get_local_q 経由で呼び出す (TaskFeature対応)
                    q_local = self.q_network.get_local_q(agent_idx, state_tensor, tf_tensor)
                    actions[agent_name] = q_local.argmax().item()
        return actions
    
    def store_transition(self, state_dict, action_dict, reward_dict, next_state_dict, done_dict, task_features=None, next_task_features=None):
        """マルチエージェント経験を保存 (Task Features含む)"""
        # 経験タプルに task_features を追加
        experience = (state_dict, action_dict, reward_dict, next_state_dict, done_dict, task_features, next_task_features)
        
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
        """VDN モデル学習ステップ (Attention対応)"""
        if len(self.memory) < self.batch_size:
            return 0
        
        if self.use_shared_buffer:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        # 経験タプルの展開 (task_features が追加されている)
        state_dicts, action_dicts, reward_dicts, next_state_dicts, done_dicts, task_features_list, next_task_features_list = zip(*batch)
        
        agent_names = list(state_dicts[0].keys())
        
        # データをテンソル化して辞書にまとめる
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
        
        # Task Features のバッチ化
        tf_batch = None
        ntf_batch = None
        
        # task_features_list の中身が None でないか確認
        if task_features_list[0] is not None:
            # task_features_list は [ (MaxOrders, 3), ... ] のリスト
            tf_batch = torch.FloatTensor(np.array(task_features_list)).to(self.device)
        
        if next_task_features_list[0] is not None:
            ntf_batch = torch.FloatTensor(np.array(next_task_features_list)).to(self.device)

        # ----------------------------------------------------
        # 1. 現在の Q_tot の計算
        # ----------------------------------------------------
        states_list = [states_by_agent[agent] for agent in agent_names]
        
        # task_features を渡す
        q_locals = self.q_network(states_list, tf_batch)
        
        q_selected_sum = torch.zeros(self.batch_size).to(self.device)
        for i, agent in enumerate(agent_names):
            q_val = q_locals[i].gather(1, actions_by_agent[agent].unsqueeze(1)).squeeze()
            q_selected_sum += q_val
            
        # ----------------------------------------------------
        # 2. ターゲット Q_tot の計算
        # ----------------------------------------------------
        next_states_list = [next_states_by_agent[agent] for agent in agent_names]
        
        with torch.no_grad():
            # next_task_features を渡す
            q_locals_next = self.target_network(next_states_list, ntf_batch)
            
            q_next_max_sum = torch.zeros(self.batch_size).to(self.device)
            for i in range(self.num_agents):
                q_next_max_sum += q_locals_next[i].max(1)[0]
            
            total_rewards = torch.zeros(self.batch_size).to(self.device)
            for agent in agent_names:
                total_rewards += rewards_by_agent[agent]
            
            all_dones = torch.stack(list(dones_by_agent.values())).min(dim=0)[0] 
            q_target = total_rewards + (1 - all_dones) * self.gamma * q_next_max_sum

        # ----------------------------------------------------
        # 3. ロス計算と更新
        # ----------------------------------------------------
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