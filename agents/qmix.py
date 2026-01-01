import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .vdn import VDNNetwork  # ローカルQネットワークはVDNと共通の構造を使用

class QMixer(nn.Module):
    """
    QMIXの混合ネットワーク
    各エージェントのQ値とグローバル状態(S)を入力とし、Q_totを出力する
    """
    def __init__(self, n_agents, state_dim, mixing_embed_dim=64):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # ハイパーネットワーク: 状態Sから混合ネットワークの重みとバイアスを生成する
        # 重みは非負にする必要があるため、絶対値またはELU+1を適用する
        self.hyper_w_1 = nn.Linear(state_dim, mixing_embed_dim * n_agents)
        self.hyper_w_final = nn.Linear(state_dim, mixing_embed_dim)

        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: [batch_size, 1, n_agents] 各エージェントの選択アクションのQ値
        states: [batch_size, state_dim] グローバル状態
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.matmul(agent_qs, w1) + b1)

        # Final layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = self.hyper_b_final(states)
        b_final = b_final.view(-1, 1, 1)

        # Compute Q_tot
        y = torch.matmul(hidden, w_final) + b_final
        q_tot = y.view(bs, -1)
        return q_tot

class QMIXAgent:
    """QMIXエージェント管理クラス"""
    def __init__(self, state_dim, action_dim, global_state_dim, num_agents=2, lr=0.0001, shared_buffer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.global_state_dim = global_state_dim

        # ローカルQネットワーク (VDNと共通のリスト形式)
        self.q_network = VDNNetwork(state_dim, action_dim, num_agents).to(self.device)
        self.target_network = VDNNetwork(state_dim, action_dim, num_agents).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Mixerネットワーク
        self.mixer = QMixer(num_agents, global_state_dim).to(self.device)
        self.target_mixer = QMixer(num_agents, global_state_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        self.memory = shared_buffer if shared_buffer is not None else deque(maxlen=50000)
        self.use_shared_buffer = shared_buffer is not None

        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.batch_size = 128
        self.update_counter = 0

    def select_actions(self, states_dict):
        actions = {}
        for agent_name, state in states_dict.items():
            if random.random() < self.epsilon:
                actions[agent_name] = random.randint(0, self.action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    agent_idx = int(agent_name.split('_')[1])
                    q_local = self.q_network.get_local_q(agent_idx, state_tensor) # n_network を q_network に変更
                    actions[agent_name] = q_local.argmax().item()
        return actions

    def store_transition(self, s_dict, a_dict, r_dict, ns_dict, d_dict, global_s, global_ns):
        experience = (s_dict, a_dict, r_dict, ns_dict, d_dict, global_s, global_ns)
        
        # バッファが SharedReplayBuffer クラス（addメソッド持ち）か判定して使い分ける
        if hasattr(self.memory, 'add'):
            self.memory.add(experience)
        else:
            self.memory.append(experience)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = self.memory.sample(self.batch_size) if self.use_shared_buffer else random.sample(self.memory, self.batch_size)
        s_dicts, a_dicts, r_dicts, ns_dicts, d_dicts, g_states, g_ns_states = zip(*batch)

        agent_names = list(s_dicts[0].keys())
        g_states = torch.FloatTensor(np.array(g_states)).to(self.device)
        g_ns_states = torch.FloatTensor(np.array(g_ns_states)).to(self.device)

        # ローカルQ値の計算
        states_list = [torch.FloatTensor(np.array([d[agent] for d in s_dicts])).to(self.device) for agent in agent_names]
        q_locals = self.q_network(states_list)
        
        # 選択された行動のQ値を抽出
        chosen_q_list = []
        for i, agent in enumerate(agent_names):
            actions = torch.LongTensor([d[agent] for d in a_dicts]).to(self.device).unsqueeze(1)
            chosen_q = q_locals[i].gather(1, actions)
            chosen_q_list.append(chosen_q)
        
        chosen_q_tensor = torch.cat(chosen_q_list, dim=1).unsqueeze(1) # [batch, 1, n_agents]
        q_tot = self.mixer(chosen_q_tensor, g_states)

        # ターゲットQ値の計算
        with torch.no_grad():
            ns_list = [torch.FloatTensor(np.array([d[agent] for d in ns_dicts])).to(self.device) for agent in agent_names]
            target_q_locals = self.target_network(ns_list)
            
            # 各エージェントの最大Q値
            max_q_list = [q.max(dim=1, keepdim=True)[0] for q in target_q_locals]
            max_q_tensor = torch.cat(max_q_list, dim=1).unsqueeze(1)
            
            target_q_tot = self.target_mixer(max_q_tensor, g_ns_states)
            
            # 全体報酬の計算
            total_rewards = torch.FloatTensor([sum(d.values()) for d in r_dicts]).to(self.device).unsqueeze(1)
            all_dones = torch.FloatTensor([all(d.values()) for d in d_dicts]).to(self.device).unsqueeze(1)
            
            y_target = total_rewards + (1 - all_dones) * self.gamma * target_q_tot

        loss = nn.SmoothL1Loss()(q_tot, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=10)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % 100 == 0: self.scheduler.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)