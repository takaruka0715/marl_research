import numpy as np
import torch
import random
from env import RestaurantEnv
from agents import DQNAgent, VDNAgent, SharedReplayBuffer
from agents.tar2 import TAR2Network, collate_trajectories
from .curriculum import Curriculum

class Trainer:
    """マルチエージェント学習トレーナー（DQN/VDN + TAR2 対応）"""
    
    # 修正: use_tar2 を引数に追加
    def __init__(self, num_episodes=30000, use_shared_replay=True, use_vdn=False, use_tar2=False, config=None):
        self.num_episodes = num_episodes
        self.use_shared_replay = use_shared_replay
        self.use_vdn = use_vdn
        self.use_tar2 = use_tar2  # 修正: 引数から直接設定
        self.config = config
        self.curriculum = Curriculum()
        
        self.agents = {}
        self.episode_rewards = {}
        self.avg_rewards = {}
        self.served_stats = {}
    
    def train(self):
        """学習ループ実行"""
        action_dim = 4
        
        # 環境初期化 (状態次元取得用)
        temp_env = RestaurantEnv(layout_type='empty', local_obs_size=5, config=self.config)
        state_dim = temp_env.observation_space('agent_0').shape[0]
        print(f"State Dimension: {state_dim}")
        print(f"System: {'VDN' if self.use_vdn else 'Independent DQN'} | TAR2: {'ON' if self.use_tar2 else 'OFF'}")
        
        # TAR2 初期化
        if self.use_tar2:
            self.tar2 = TAR2Network(state_dim, action_dim, num_agents=2)
            self.tar2_buffer = []

        # バッファ・エージェント初期化
        shared_buffer = SharedReplayBuffer(capacity=50000) if self.use_shared_replay else None
        
        if self.use_vdn:
            self.agents = {
                'vdn': VDNAgent(state_dim, action_dim, num_agents=2, shared_buffer=shared_buffer)
            }
        else:
            self.agents = {
                agent_name: DQNAgent(state_dim, action_dim, shared_buffer=shared_buffer) 
                for agent_name in temp_env.possible_agents
            }
        
        self.episode_rewards = {agent: [] for agent in temp_env.possible_agents}
        self.avg_rewards = {agent: [] for agent in temp_env.possible_agents}
        self.served_stats = {agent: [] for agent in temp_env.possible_agents}
        
        current_env = None
        current_stage = None
        
        for episode in range(self.num_episodes):
            stage = self.curriculum.get_stage(episode)
            
            if stage != current_stage:
                prev_desc = current_stage['description'] if current_stage else "None"
                current_stage = stage
                print(f"\n{'='*70}")
                print(f"=== Curriculum: {prev_desc} -> {stage['description']} ===")
                print(f"=== Episode {episode} / {self.num_episodes} ===")
                print(f"{'='*70}")
                
                current_env = RestaurantEnv(
                    layout_type=stage['layout'],
                    enable_customers=stage['customers'],
                    customer_spawn_interval=stage['spawn_interval'],
                    local_obs_size=5,
                    coop_factor=0.5,
                    config=self.config
                )
                
                if episode > 0:
                    if self.use_vdn:
                        self.agents['vdn'].epsilon = 0.6
                    else:
                        for agent_name in self.agents:
                            self.agents[agent_name].epsilon = 0.6

            # ----------------------------------------------------
            # 1. データ収集フェーズ (学習なしで走らせる)
            # ----------------------------------------------------
            trajectory_data = self._run_episode_collect_only(current_env, stage)
            
            # 報酬記録
            total_r = trajectory_data['total_reward']
            for agent_name in current_env.possible_agents:
                self.episode_rewards[agent_name].append(total_r / 2)
                self.avg_rewards[agent_name].append(np.mean(self.episode_rewards[agent_name][-50:]))
                self.served_stats[agent_name].append(current_env.served_count[agent_name])

            # ----------------------------------------------------
            # 2. TAR2 報酬再計算フェーズ
            # ----------------------------------------------------
            shaped_rewards = None
            if self.use_tar2:
                # バッファに追加
                self.tar2_buffer.append(trajectory_data)
                
                # TAR2モデルの学習 (バッチが溜まったら)
                if len(self.tar2_buffer) >= 32:
                    b_states, b_actions, b_rewards, _ = collate_trajectories(self.tar2_buffer, self.tar2.device)
                    tar2_loss = self.tar2.update(b_states, b_actions, b_rewards)
                    self.tar2_buffer = []

                # 現在のエピソードの報酬を再分配 (推論)
                s, a, r_tot, _ = collate_trajectories([trajectory_data], self.tar2.device)
                f_s = s[:, -1, :, :]
                
                with torch.no_grad():
                    scores, _ = self.tar2(s, a, f_s)
                    shaped_tensor = self.tar2.get_redistributed_rewards(scores, r_tot)
                    shaped_rewards = shaped_tensor.squeeze(0).cpu().numpy() # (T, N)
            
            # ----------------------------------------------------
            # 3. エージェント学習フェーズ (再計算された報酬を使用)
            # ----------------------------------------------------
            self._store_and_train_agents(trajectory_data, shaped_rewards)

            # Epsilon減衰
            if self.use_vdn:
                self.agents['vdn'].decay_epsilon()
            else:
                for agent_name in current_env.possible_agents:
                    self.agents[agent_name].decay_epsilon()
            
            # 定期更新
            if episode % 10 == 0:
                if self.use_vdn:
                    self.agents['vdn'].update_target_network()
                else:
                    for agent_name in current_env.possible_agents:
                        self.agents[agent_name].update_target_network()
            
            # ログ表示
            if episode % 100 == 0:
                avg_0 = self.avg_rewards['agent_0'][-1]
                served_0 = np.mean(self.served_stats['agent_0'][-50:])
                eps = self.agents['vdn'].epsilon if self.use_vdn else self.agents['agent_0'].epsilon
                tar2_msg = f" | TAR2 Shaped" if self.use_tar2 else ""
                print(f"Ep {episode:4d} | AvgReward: {avg_0:6.1f} | Served: {served_0:.1f} | ε={eps:.3f}{tar2_msg}")
        
        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, current_env

    def _run_episode_collect_only(self, env, stage):
        """学習を行わず、全ステップのデータを収集して返す"""
        env.reset()
        
        states_seq = []
        actions_seq = []
        rewards_seq = []
        dones_seq = []
        
        states = {agent: env.observe(agent) for agent in env.possible_agents}
        
        if stage['layout'] == 'empty' and len(env.seats) > 0:
            if np.random.random() < 0.3:
                order_pos = random.choice(env.seats)
                if order_pos not in env.active_orders:
                    env.active_orders.append(order_pos)
                env.ready_dishes += 1
        
        episode_reward_sum = 0
        agents_order = env.possible_agents 
        
        for step in range(600):
            step_states = []
            step_actions = []
            step_rewards = []
            step_dones = []
            
            current_actions = {}
            for agent_name in agents_order:
                state = states[agent_name]
                step_states.append(state)
                
                if env.truncations.get(agent_name, False):
                    action = 0
                else:
                    if self.use_vdn:
                        actions_dict = self.agents['vdn'].select_actions(states)
                        action = actions_dict[agent_name]
                    else:
                        action = self.agents[agent_name].select_action(state)
                
                current_actions[agent_name] = action
                step_actions.append(action)

            for agent_name in agents_order:
                if env.agent_selection == agent_name:
                    env.step(current_actions[agent_name])
            
            for agent_name in agents_order:
                next_obs = env.observe(agent_name)
                states[agent_name] = next_obs
                
                r = env.rewards.get(agent_name, 0)
                d = env.truncations.get(agent_name, False)
                
                step_rewards.append(r)
                step_dones.append(d)
                
                episode_reward_sum += r

            states_seq.append(np.array(step_states))
            actions_seq.append(np.array(step_actions))
            rewards_seq.append(np.array(step_rewards))
            dones_seq.append(np.array(step_dones))

            if all(env.truncations.values()):
                break
        
        return {
            'states': np.array(states_seq),
            'actions': np.array(actions_seq),
            'rewards': np.array(rewards_seq),
            'dones': np.array(dones_seq),
            'total_reward': episode_reward_sum
        }

    def _store_and_train_agents(self, trajectory, shaped_rewards):
        """収集したデータと報酬を使ってバッファ保存と学習を行う"""
        T = len(trajectory['states'])
        agents_order = ['agent_0', 'agent_1']
        
        for t in range(T - 1):
            s_t = trajectory['states'][t]
            a_t = trajectory['actions'][t]
            ns_t = trajectory['states'][t+1]
            d_t = trajectory['dones'][t]
            
            if shaped_rewards is not None:
                r_t = shaped_rewards[t]
            else:
                r_t = trajectory['rewards'][t]

            s_dict = {name: s_t[i] for i, name in enumerate(agents_order)}
            a_dict = {name: a_t[i] for i, name in enumerate(agents_order)}
            r_dict = {name: r_t[i] for i, name in enumerate(agents_order)}
            ns_dict = {name: ns_t[i] for i, name in enumerate(agents_order)}
            d_dict = {name: bool(d_t[i]) for i, name in enumerate(agents_order)}
            
            if self.use_vdn:
                self.agents['vdn'].store_transition(s_dict, a_dict, r_dict, ns_dict, d_dict)
                self.agents['vdn'].train()
            else:
                for i, name in enumerate(agents_order):
                    self.agents[name].store_transition(
                        s_dict[name], a_dict[name], r_dict[name], ns_dict[name], d_dict[name]
                    )
                    self.agents[name].train()