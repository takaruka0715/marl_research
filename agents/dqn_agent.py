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

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    # 【追加】モデル読み込み
    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval() # 推論モードへ
    
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
    """Value Decomposition Networks (VDN) を用いたマルチエージェント (修正版)"""
    def __init__(self, state_dim, action_dim, num_agents=2, lr=0.0001, shared_buffer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # 修正: VDNNetworkはリストを返すようになった
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
        """全エージェントの行動を選択"""
        actions = {}
        for agent_name, state in states_dict.items():
            if random.random() < self.epsilon:
                actions[agent_name] = random.randint(0, self.action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    agent_idx = int(agent_name.split('_')[1])
                    q_local = self.q_network.get_local_q(agent_idx, state_tensor) # 共通メソッドを使用
                    actions[agent_name] = q_local.argmax().item()
        return actions
    
    # 【追加】モデル保存
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    # 【追加】モデル読み込み
    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
    
    def store_transition(self, state_dict, action_dict, reward_dict, next_state_dict, done_dict):
        """マルチエージェント経験を保存"""
        experience = (state_dict, action_dict, reward_dict, next_state_dict, done_dict)
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
        """VDN モデル学習ステップ (修正版)"""
        if len(self.memory) < self.batch_size:
            return 0
        
        if self.use_shared_buffer:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        
        state_dicts, action_dicts, reward_dicts, next_state_dicts, done_dicts = zip(*batch)
        
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

        # ----------------------------------------------------
        # 1. 現在の Q_tot の計算
        # ----------------------------------------------------
        states_list = [states_by_agent[agent] for agent in agent_names]
        q_locals = self.q_network(states_list)  # [Agent1_Q, Agent2_Q, ...]
        
        # 各エージェントが「実際に選んだ行動」のQ値を取り出す
        q_selected_sum = torch.zeros(self.batch_size).to(self.device)
        
        for i, agent in enumerate(agent_names):
            # gather: [Batch, ActionDim] -> [Batch, 1]
            q_val = q_locals[i].gather(1, actions_by_agent[agent].unsqueeze(1)).squeeze()
            q_selected_sum += q_val
            
        # ----------------------------------------------------
        # 2. ターゲット Q_tot の計算
        # ----------------------------------------------------
        next_states_list = [next_states_by_agent[agent] for agent in agent_names]
        
        with torch.no_grad():
            q_locals_next = self.target_network(next_states_list)
            
            # VDNのターゲット: Σ(Max Q_i(s'))
            # 各エージェントごとに最大のQ値を選んでから合計する
            q_next_max_sum = torch.zeros(self.batch_size).to(self.device)
            for i in range(self.num_agents):
                q_next_max_sum += q_locals_next[i].max(1)[0]
            
            # 報酬の合計 (Global Reward)
            total_rewards = torch.zeros(self.batch_size).to(self.device)
            for agent in agent_names:
                total_rewards += rewards_by_agent[agent]
            
            # 終了判定 (全エージェントが終了したら終了とみなす、または環境依存)
            # ここでは「全員終わったらDone」として計算する論理積(AND)をとるか、
            # 「誰か一人でも失敗したら」なら論理和(OR)をとります。
            # 通常、Gymのようなステップ実行なら done は同時刻に発生するため、代表して agent_0 を見るか、論理積をとります。
            # 安全のため「全員がFalseのときだけ継続」(= 1 - all_done) とします。
            
            # batch単位で、全員が done=True なら all_dones=1
            all_dones = torch.stack(list(dones_by_agent.values())).min(dim=0)[0] 
            
            # Target = Σr + γ * (1 - all_done) * Σ(max Q_next)
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

    # (update_target_network, decay_epsilon は変更なし)
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)