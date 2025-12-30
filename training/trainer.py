import numpy as np
import torch
import random
from envs import RestaurantEnv
from agents import DQNAgent, VDNAgent, SharedReplayBuffer
from agents.tar2 import TAR2Network, collate_trajectories
from .curriculum import Curriculum
from agents.qmix import QMIXAgent

class Trainer:
    """マルチエージェント学習トレーナー（DQN/VDN + TAR2 + 適応型カリキュラム 対応）"""
    
    # 修正: use_tar2 を引数に追加
    def __init__(self, num_episodes=30000, use_shared_replay=True, use_qmix=False, use_vdn=False, use_tar2=False, config=None):
        self.num_episodes = num_episodes
        self.use_shared_replay = use_shared_replay
        self.use_vdn = use_vdn
        self.use_tar2 = use_tar2  # 修正: 引数から直接設定
        self.use_qmix = use_qmix # 追加
        self.config = config
        
        # 適応型カリキュラムを使用
        self.curriculum = Curriculum()
        
        self.agents = {}
        self.episode_rewards = {}
        self.avg_rewards = {}
        self.served_stats = {}

        # TAR2用バッファ (追加)
        self.tar2 = None
        self.tar2_buffer = []
    
    def train(self):
        """学習ループ実行"""
        action_dim = 4
        
        # --- 初期環境設定（状態次元取得用） ---
        # [cite_start]適応型カリキュラムから初期ステージを取得 [cite: 68]
        current_stage = self.curriculum.get_current_stage()
        
        # 環境初期化
        current_env = RestaurantEnv(
            layout_type=current_stage['layout'],
            enable_customers=current_stage['customers'],
            customer_spawn_interval=current_stage['spawn_interval'],
            local_obs_size=5,
            config=self.config
        )

        state_dim = current_env.observation_space('agent_0').shape[0]

        algo_name = "Independent DQN"
        if self.use_qmix: algo_name = "QMIX"
        elif self.use_vdn: algo_name = "VDN"

        print(f"State Dimension: {state_dim}")
        print(f"System: {algo_name} | TAR2: {'ON' if self.use_tar2 else 'OFF'}")
        
        # TAR2 初期化
        if self.use_tar2:
            self.tar2 = TAR2Network(state_dim, action_dim, num_agents=2)
            self.tar2_buffer = []

        # バッファ・エージェント初期化
        shared_buffer = SharedReplayBuffer(capacity=50000) if self.use_shared_replay else None
        
        if self.use_qmix:
            # グローバル状態は全エージェントの観測を結合したものとする例
            global_state_dim = state_dim * 2 
            self.agents = {
                'qmix': QMIXAgent(state_dim, action_dim, global_state_dim, num_agents=2, shared_buffer=shared_buffer)
            }
        elif self.use_vdn:
            self.agents = {
                'vdn': VDNAgent(state_dim, action_dim, num_agents=2, shared_buffer=shared_buffer)
            }
        else:
            self.agents = {
                agent_name: DQNAgent(state_dim, action_dim, shared_buffer=shared_buffer) 
                for agent_name in current_env.possible_agents
            }
        
        self.episode_rewards = {agent: [] for agent in current_env.possible_agents}
        self.avg_rewards = {agent: [] for agent in current_env.possible_agents}
        self.served_stats = {agent: [] for agent in current_env.possible_agents}
        
        # ステージ滞在カウンター
        stage_episode_count = 0
        
        print(f"\n{'='*70}")
        print(f"=== STARTING STAGE: {current_stage['description']} ===")
        print(f"{'='*70}")
        
        for episode in range(self.num_episodes):
            
            # ----------------------------------------------------
            # 0. 適応型カリキュラムの進行判定 (閾値/タイムアウト)
            # ----------------------------------------------------
            # agent_0 の平均報酬を代表値として使用
            current_served_performance = 0
            if len(self.served_stats['agent_0']) > 0:
                recent_indices = range(-min(50, len(self.served_stats['agent_0'])), 0)
                # 全エージェントのカウントを合計する
                total_served_list = [
                    sum(self.served_stats[agent][i] for agent in current_env.possible_agents)
                    for i in recent_indices
                ]
                current_served_performance = np.mean(total_served_list)
            else:
                current_served_performance = 0

            # check_progression に配膳数の平均を渡すように変更 [cite: 143]
            should_proceed, reason = self.curriculum.check_progression(
                current_served_performance, 
                stage_episode_count
            )

            if should_proceed:
                # 次のステージへ進む
                new_stage = self.curriculum.get_current_stage()
                
                print(f"\n{'='*70}")
                print(f"🔄 CURRICULUM PROGRESSION")
                print(f"   From: {current_stage['description']}")
                print(f"   To:   {new_stage['description']}")
                print(f"   Why:  {reason}")
                print(f"   Perf: {current_served_performance:.1f} (Target: {current_stage['threshold']})")
                print(f"{'='*70}")
                
                # [cite_start]新しいステージ設定で環境を再構築 [cite: 79]
                current_stage = new_stage
                current_env = RestaurantEnv(
                    layout_type=current_stage['layout'],
                    enable_customers=current_stage['customers'],
                    customer_spawn_interval=current_stage['spawn_interval'],
                    local_obs_size=5,
                    coop_factor=0.5,
                    config=self.config
                )
                
                # 滞在カウンターリセット
                stage_episode_count = 0
                
                # 探索率(epsilon)のリセット（環境が変わったので再探索させる）
                reset_epsilon = 0.8
                if self.use_vdn:
                    self.agents['vdn'].epsilon = max(self.agents['vdn'].epsilon, reset_epsilon)
                else:
                    for agent_name in self.agents:
                        self.agents[agent_name].epsilon = max(self.agents[agent_name].epsilon, reset_epsilon)

            # ----------------------------------------------------
            # 1. データ収集フェーズ (学習なしで走らせる)
            # ----------------------------------------------------
            trajectory_data = self._run_episode_collect_only(current_env)
            stage_episode_count += 1
            
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
            elif getattr(self, 'use_qmix', False): # ここを追加
                self.agents['qmix'].decay_epsilon()
            else:
                for agent_name in current_env.possible_agents:
                    self.agents[agent_name].decay_epsilon()
            
            # [cite_start]定期更新 [cite: 90]
            # 定期更新
            if episode % 10 == 0:
                if self.use_vdn:
                    self.agents['vdn'].update_target_network()
                elif getattr(self, 'use_qmix', False): # 追加
                    self.agents['qmix'].update_target_network()
                else:
                    for agent_name in current_env.possible_agents:
                        self.agents[agent_name].update_target_network()
            
            # ログ表示
            if episode % 100 == 0:
                # 1. 平均報酬（直近50エピソードの移動平均）の取得 
                # エージェントごとの平均報酬を合計してチーム全体の成果とする
                avg_0 = self.avg_rewards['agent_0'][-1] if self.avg_rewards['agent_0'] else 0
                avg_1 = self.avg_rewards['agent_1'][-1] if self.avg_rewards['agent_1'] else 0
                team_avg_reward = avg_0 + avg_1

                # 2. 配膳数の統計（直近50エピソード） 
                served_a0 = np.mean(self.served_stats['agent_0'][-50:])
                served_a1 = np.mean(self.served_stats['agent_1'][-50:])
                total_served = served_a0 + served_a1
                
                # 3. 探索率とTAR2の状態取得
                if self.use_vdn:
                    eps = self.agents['vdn'].epsilon
                elif getattr(self, 'use_qmix', False):
                    eps = self.agents['qmix'].epsilon
                else:
                    eps = self.agents['agent_0'].epsilon
                tar2_msg = " | TAR2 Shaped" if self.use_tar2 else ""
                
                # ログ表示の更新
                print(f"Ep {episode:4d} | StgEp: {stage_episode_count:4d} | "
                      f"AvgReward: {team_avg_reward:6.1f} | "  # ← ここに復活させました
                      f"Total Served: {total_served:4.1f} (A0:{served_a0:.1f}, A1:{served_a1:.1f}) | "
                      f"ε={eps:.3f}{tar2_msg}")
        
        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, current_env

    def _run_episode_collect_only(self, env):
        """学習を行わず、全ステップのデータを収集して返す"""
        env.reset()
        
        states_seq = []
        actions_seq = []
        rewards_seq = []
        dones_seq = []
        global_states_seq = []  # QMIX用のグローバル状態

        # [cite_start]グローバル状態（全エージェントの観測結合）を生成する関数 [cite: 387]
        def get_global_state(obs_dict):
            return np.concatenate([obs_dict[a] for a in env.possible_agents])

        # [cite_start]初期観測の取得 [cite: 388]
        states = {agent: env.observe(agent) for agent in env.possible_agents}
        
        # [cite_start]初期グローバル状態を取得 [cite: 388]
        current_global_state = get_global_state(states)
        
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
                
                # [cite_start]行動選択 [cite: 390, 391, 392, 393]
                if env.truncations.get(agent_name, False):
                    action = 0
                else:
                    if self.use_vdn:
                        actions_dict = self.agents['vdn'].select_actions(states)
                        action = actions_dict[agent_name]
                    elif getattr(self, 'use_qmix', False): # QMIX対応
                        actions_dict = self.agents['qmix'].select_actions(states)
                        action = actions_dict[agent_name]
                    else:
                        action = self.agents[agent_name].select_action(state)
                
                current_actions[agent_name] = action
                step_actions.append(action)

            # [cite_start]環境の更新 [cite: 394]
            for agent_name in agents_order:
                if env.agent_selection == agent_name:
                    env.step(current_actions[agent_name])
            
            # [cite_start]次の状態の観測と報酬記録 [cite: 395, 396]
            for agent_name in agents_order:
                next_obs = env.observe(agent_name)
                states[agent_name] = next_obs
                
                r = env.rewards.get(agent_name, 0)
                d = env.truncations.get(agent_name, False)
                
                step_rewards.append(r)
                step_dones.append(d)
                episode_reward_sum += r

            # [cite_start]現在のステップの情報を保存し、次のグローバル状態を更新 [cite: 388]
            global_states_seq.append(current_global_state)
            current_global_state = get_global_state(states)

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
            'global_states': np.array(global_states_seq), # 学習に使用 [cite: 398]
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

            # QMIX用のグローバル状態
            g_s_t = trajectory['global_states'][t]
            g_ns_t = trajectory['global_states'][t+1]
            
            if shaped_rewards is not None:
                r_t = shaped_rewards[t]
            else:
                r_t = trajectory['rewards'][t]

            s_dict = {name: s_t[i] for i, name in enumerate(agents_order)}
            a_dict = {name: a_t[i] for i, name in enumerate(agents_order)}
            r_dict = {name: r_t[i] for i, name in enumerate(agents_order)}
            ns_dict = {name: ns_t[i] for i, name in enumerate(agents_order)}
            d_dict = {name: bool(d_t[i]) for i, name in enumerate(agents_order)}

            # --- アルゴリズムごとに排他的に実行するよう整理 ---
            if getattr(self, 'use_qmix', False):
                # 1. QMIXの場合
                self.agents['qmix'].store_transition(s_dict, a_dict, r_dict, ns_dict, d_dict, g_s_t, g_ns_t)
                self.agents['qmix'].train()
            elif self.use_vdn:
                # 2. VDNの場合
                self.agents['vdn'].store_transition(s_dict, a_dict, r_dict, ns_dict, d_dict)
                self.agents['vdn'].train()
            else:
                # 3. 独立DQNの場合
                for i, name in enumerate(agents_order):
                    self.agents[name].store_transition(
                        s_dict[name], a_dict[name], r_dict[name], ns_dict[name], d_dict[name]
                    )
                    self.agents[name].train()