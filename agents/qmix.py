# ================================================
# FILE: agents/qmix.py
# ================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .vdn import VDNNetwork

class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=64):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Linear(state_dim, mixing_embed_dim * n_agents)
        self.hyper_w_final = nn.Linear(state_dim, mixing_embed_dim)

        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.matmul(agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = self.hyper_b_final(states)
        b_final = b_final.view(-1, 1, 1)

        y = torch.matmul(hidden, w_final) + b_final
        q_tot = y.view(bs, -1)
        return q_tot

class QMIXAgent:
    def __init__(self, state_dim, action_dim, global_state_dim, num_agents=2, lr=0.0001, shared_buffer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.global_state_dim = global_state_dim

        self.q_network = VDNNetwork(state_dim, action_dim, num_agents).to(self.device)
        self.target_network = VDNNetwork(state_dim, action_dim, num_agents).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

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
        self.gamma = 0.99
        self.batch_size = 128
        self.update_counter = 0

    def save_model(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'mixer': self.mixer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.q_network.eval()
        self.mixer.eval()

    def _apply_action_mask(self, q_values, states):
        """状態のインデックス50, 51からカウンターとの距離を逆算し、離れていれば受取をマスク"""
        dx = states[:, 50] * 15.0
        dy = states[:, 51] * 15.0
        dist = torch.abs(dx.round()) + torch.abs(dy.round())
        invalid_mask = dist > 1.1 # カウンター隣接マス以外は無効
        q_values[invalid_mask, 4:] = -1e9
        return q_values

    def _apply_avail_actions(self, q_values, avail_actions):
        if avail_actions is None:
            return q_values

        mask = torch.as_tensor(avail_actions, dtype=torch.bool, device=self.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and q_values.shape[0] > 1:
            mask = mask.expand(q_values.shape[0], -1)

        return q_values.masked_fill(~mask, -1e9)

    def select_actions(self, states_dict, **kwargs):
        avail_actions = kwargs.get('avail_actions') or {}
        actions = {}
        for agent_name, state in states_dict.items():
            agent_avail_actions = avail_actions.get(agent_name) if isinstance(avail_actions, dict) else None
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if random.random() < self.epsilon:
                # ランダム行動時もマスクを適用
                dummy_q = torch.zeros(1, self.action_dim).to(self.device)
                masked_q = self._apply_action_mask(dummy_q, state_tensor)
                masked_q = self._apply_avail_actions(masked_q, agent_avail_actions)
                valid_actions = (masked_q[0] > -1e8).nonzero(as_tuple=True)[0].tolist()
                if not valid_actions:
                    valid_actions = list(range(self.action_dim))
                actions[agent_name] = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    agent_idx = int(agent_name.split('_')[1])
                    q_local = self.q_network.get_local_q(agent_idx, state_tensor) 
                    q_local = self._apply_action_mask(q_local, state_tensor)
                    q_local = self._apply_avail_actions(q_local, agent_avail_actions)
                    actions[agent_name] = q_local.argmax().item()
        return actions

    def store_transition(self, s_dict, a_dict, r_dict, ns_dict, d_dict, global_s, global_ns):
        experience = (s_dict, a_dict, r_dict, ns_dict, d_dict, global_s, global_ns)
        if hasattr(self.memory, 'add'):
            self.memory.add(experience)
        else:
            self.memory.append(experience)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # ★PER用のサンプリング (beta=0.4 は学習初期の標準値)
        if self.use_shared_buffer and hasattr(self.memory, 'update_priorities'):
            batch, indices, weights = self.memory.sample(self.batch_size, beta=0.4)
            weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        else:
            if hasattr(self.memory, 'sample'):
                batch = self.memory.sample(self.batch_size)
            else:
                batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights_tensor = torch.ones(self.batch_size, 1).to(self.device)

        s_dicts, a_dicts, r_dicts, ns_dicts, d_dicts, g_states, g_ns_states = zip(*batch)

        agent_names = list(s_dicts[0].keys())
        g_states = torch.FloatTensor(np.array(g_states)).to(self.device)
        g_ns_states = torch.FloatTensor(np.array(g_ns_states)).to(self.device)

        states_list = [torch.FloatTensor(np.array([d[agent] for d in s_dicts])).to(self.device) for agent in agent_names]
        q_locals = self.q_network(states_list)
        
        chosen_q_list = []
        for i, agent in enumerate(agent_names):
            actions = torch.LongTensor([d[agent] for d in a_dicts]).to(self.device).unsqueeze(1)
            chosen_q = q_locals[i].gather(1, actions)
            chosen_q_list.append(chosen_q)
        
        chosen_q_tensor = torch.cat(chosen_q_list, dim=1).unsqueeze(1)
        q_tot = self.mixer(chosen_q_tensor, g_states)

        with torch.no_grad():
            ns_list = [torch.FloatTensor(np.array([d[agent] for d in ns_dicts])).to(self.device) for agent in agent_names]
            target_q_locals = self.target_network(ns_list)
            
            # ★ターゲットQ値にもマスクを適用し、過大評価の爆発を防ぐ
            for i in range(self.num_agents):
                target_q_locals[i] = self._apply_action_mask(target_q_locals[i], ns_list[i])
            
            max_q_list = [q.max(dim=1, keepdim=True)[0] for q in target_q_locals]
            max_q_tensor = torch.cat(max_q_list, dim=1).unsqueeze(1)
            
            target_q_tot = self.target_mixer(max_q_tensor, g_ns_states)
            
            total_rewards = torch.FloatTensor([sum(d.values()) for d in r_dicts]).to(self.device).unsqueeze(1)
            all_dones = torch.FloatTensor([all(d.values()) for d in d_dicts]).to(self.device).unsqueeze(1)
            
            y_target = total_rewards + (1 - all_dones) * self.gamma * target_q_tot

        # ★重要: 重み付きLossの計算と優先度の更新
        loss_elements = F.smooth_l1_loss(q_tot, y_target, reduction='none')
        loss = (weights_tensor * loss_elements).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=10)
        self.optimizer.step()

        # ★TD誤差を使ってバッファの優先度を更新 (誤差が大きい=驚きが大きい=復習すべき)
        if indices is not None:
            td_errors = torch.abs(q_tot - y_target).detach().cpu().numpy().squeeze()
            # 微小な定数 1e-6 を足して優先度が完全に0になるのを防ぐ
            # bs=1 の場合などのため1次元配列を保証する
            if td_errors.ndim == 0:
                td_errors = np.array([td_errors])
            self.memory.update_priorities(indices, td_errors + 1e-6)

        self.update_counter += 1
        if self.update_counter % 100 == 0: self.scheduler.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
