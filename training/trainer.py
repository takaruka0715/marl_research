# ================================================
# FILE: training/trainer.py
# ================================================
import numpy as np
import torch
import random
import os
from envs import RestaurantEnv
from agents import DQNAgent, VDNAgent, SharedReplayBuffer
from agents.tar2 import TAR2Network, collate_trajectories
from .curriculum import Curriculum
from agents.qmix import QMIXAgent

class Trainer:
    """マルチエージェント学習トレーナー（ParallelEnv対応版）"""
    
    def __init__(self, num_episodes=30000, use_shared_replay=True, use_qmix=False, use_vdn=False, use_tar2=False, config=None):
        self.num_episodes = num_episodes
        self.use_shared_replay = use_shared_replay
        self.use_vdn = use_vdn
        self.use_tar2 = use_tar2
        self.use_qmix = use_qmix
        self.config = config
        
        # 適応型カリキュラム
        self.curriculum = Curriculum()
        
        self.agents = {}
        self.episode_rewards = {}
        self.avg_rewards = {}
        self.served_stats = {}
        
        # 【追加】評価指標の記録用リスト
        self.collision_rates = []
        self.avg_wait_times = []

        # TAR2用バッファ
        self.tar2 = None
        self.tar2_buffer = []

        # ★ログディレクトリの作成
        self.log_dir = "training_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    # ▼▼▼ ログ出力用メソッド追加 ▼▼▼
    def _log(self, msg, stage_idx, to_console=False):
        """ログをファイルに出力し、必要に応じてコンソールにも出力する"""
        filepath = os.path.join(self.log_dir, f"stage_{stage_idx}.log")
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if to_console:
            print(msg)
    
    def train(self):
        """学習ループ実行"""
        
        # ★知識継承のための重要修正: 
        # アクション次元を最初から最大値（移動4 + 料理種類3 = 7）に固定します。
        # これにより、ステージ移行時にネットワークの構造を変える必要がなくなります。
        max_food_types = 3
        action_dim = 4 + max_food_types 
        
        # 初期ステージ取得
        current_stage = self.curriculum.get_current_stage()
        # 環境内の客が注文する種類数はカリキュラムに従う
        nft = current_stage.get('num_food_types', 1)
        
        # 環境初期化
        current_env = RestaurantEnv(
            layout_type=current_stage['layout'],
            enable_customers=current_stage['customers'],
            customer_spawn_interval=current_stage['spawn_interval'],
            local_obs_size=5,
            min_customer_dist=current_stage.get('min_customer_dist', 0),
            max_customer_dist=current_stage.get('max_customer_dist', float('inf')),
            num_food_types=nft, 
            config=self.config
        )

        state_dim = current_env.observation_space('agent_0').shape[0]

        algo_name = "Independent DQN"
        if self.use_qmix: algo_name = "QMIX"
        elif self.use_vdn: algo_name = "VDN"

        stage_idx = self.curriculum.current_stage_idx
        
        msg1 = f"State Dimension: {state_dim}"
        msg2 = f"System: {algo_name} | TAR2: {'ON' if self.use_tar2 else 'OFF'}"
        self._log(msg1, stage_idx, to_console=True)
        self._log(msg2, stage_idx, to_console=True)
        
        # TAR2 初期化
        if self.use_tar2:
            self.tar2 = TAR2Network(state_dim, action_dim, num_agents=2)
            self.tar2_buffer = []

        # バッファ・エージェント初期化
        shared_buffer = SharedReplayBuffer(capacity=50000) if self.use_shared_replay else None
        
        # --- エージェントの初期構築 (ここで1回だけ行う) ---
        if self.use_qmix:
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
        
        stage_episode_count = 0
        
        start_msg = f"\n{'='*70}\n=== STARTING STAGE: {current_stage['description']} ===\n{'='*70}"
        self._log(start_msg, stage_idx, to_console=True)
        
        for episode in range(self.num_episodes):
            
            # --- カリキュラム進行判定 ---
            current_served_performance = 0
            if len(self.served_stats['agent_0']) > 0:
                recent_indices = range(-min(50, len(self.served_stats['agent_0'])), 0)
                total_served_list = [
                    sum(self.served_stats[agent][i] for agent in current_env.possible_agents)
                    for i in recent_indices
                ]
                current_served_performance = np.mean(total_served_list)

            should_proceed, reason = self.curriculum.check_progression(
                current_served_performance, 
                stage_episode_count
            )

            if should_proceed:
                new_stage = self.curriculum.get_current_stage()
                prog_msg = (f"\n{'='*70}\n"
                            f"🔄 CURRICULUM PROGRESSION\n"
                            f"   From: {current_stage['description']}\n"
                            f"   To:   {new_stage['description']}\n"
                            f"   Why:  {reason}\n"
                            f"   Perf: {current_served_performance:.1f}\n"
                            f"{'='*70}")
                self._log(prog_msg, stage_idx, to_console=True)
                
                current_stage = new_stage
                nft = current_stage.get('num_food_types', 1)
                stage_idx = self.curriculum.current_stage_idx
                
                # 環境の再構築 (新しい料理設定と距離設定を反映)
                current_env = RestaurantEnv(
                    layout_type=current_stage['layout'],
                    enable_customers=current_stage['customers'],
                    customer_spawn_interval=current_stage['spawn_interval'],
                    local_obs_size=5,
                    min_customer_dist=current_stage.get('min_customer_dist', 0),
                    max_customer_dist=current_stage.get('max_customer_dist', float('inf')),
                    num_food_types=nft,
                    coop_factor=self.config.coop_factor,
                    config=self.config
                )
                
                # ★重要：エージェントの再初期化（self.agents = ...）を削除しました。
                # これにより、前のステージで得た「歩き方」や「配膳の概念」を保持したまま
                # 新しい料理（Action 5, 6等）の学習に移行できます。
                
                stage_episode_count = 0
                # ステージ移行時は少し探索率を戻して、新しい要素（新しい料理）を学ばせる
                reset_epsilon = 0.8
                
                if self.use_vdn:
                    self.agents['vdn'].epsilon = max(self.agents['vdn'].epsilon, reset_epsilon)
                elif getattr(self, 'use_qmix', False):
                    self.agents['qmix'].epsilon = max(self.agents['qmix'].epsilon, reset_epsilon)
                else:
                    for agent_name in self.agents:
                        self.agents[agent_name].epsilon = max(self.agents[agent_name].epsilon, reset_epsilon)

            # --- 1. データ収集フェーズ (ParallelEnv対応) ---
            trajectory_data = self._run_episode_collect_only(current_env)
            stage_episode_count += 1
            
            # 報酬記録
            total_r = trajectory_data['total_reward']
            for agent_name in current_env.possible_agents:
                agent_idx = current_env.possible_agents.index(agent_name)
                agent_total_r = np.sum(trajectory_data['rewards'][:, agent_idx])
                
                self.episode_rewards[agent_name].append(agent_total_r)
                self.avg_rewards[agent_name].append(np.mean(self.episode_rewards[agent_name][-50:]))
                self.served_stats[agent_name].append(current_env.served_count[agent_name])

            # --- 評価指標の計算と記録 ---
            total_collisions = sum(current_env.collision_count.values())
            total_agent_steps = current_env.num_moves * len(current_env.possible_agents)
            collision_rate = total_collisions / total_agent_steps if total_agent_steps > 0 else 0.0
            self.collision_rates.append(collision_rate)

            if hasattr(current_env, 'completed_wait_times') and len(current_env.completed_wait_times) > 0:
                avg_wait = np.mean(current_env.completed_wait_times)
            else:
                avg_wait = 0.0
            self.avg_wait_times.append(avg_wait)

            # --- 2. TAR2 報酬再計算フェーズ ---
            shaped_rewards = None
            if self.use_tar2:
                self.tar2_buffer.append(trajectory_data)
                
                if len(self.tar2_buffer) >= 32:
                    b_states, b_actions, b_rewards, _ = collate_trajectories(self.tar2_buffer, self.tar2.device)
                    self.tar2.update(b_states, b_actions, b_rewards)
                    self.tar2_buffer = []

                s, a, r_tot, _ = collate_trajectories([trajectory_data], self.tar2.device)
                f_s = s[:, -1, :, :]
                
                with torch.no_grad():
                    scores, _ = self.tar2(s, a, f_s)
                    shaped_tensor = self.tar2.get_redistributed_rewards(scores, r_tot)
                    shaped_rewards = shaped_tensor.squeeze(0).cpu().numpy() # (T, N)
            
            # --- 3. エージェント学習フェーズ ---
            self._store_and_train_agents(trajectory_data, shaped_rewards)

            # Epsilon減衰
            if self.use_vdn:
                self.agents['vdn'].decay_epsilon()
            elif getattr(self, 'use_qmix', False):
                self.agents['qmix'].decay_epsilon()
            else:
                for agent_name in current_env.possible_agents:
                    self.agents[agent_name].decay_epsilon()
            
            # Target Network更新
            if episode % 10 == 0:
                if self.use_vdn:
                    self.agents['vdn'].update_target_network()
                elif getattr(self, 'use_qmix', False):
                    self.agents['qmix'].update_target_network()
                else:
                    for agent_name in current_env.possible_agents:
                        self.agents[agent_name].update_target_network()
            
            # ログ出力
            if episode % 100 == 0:
                avg_0 = self.avg_rewards['agent_0'][-1] if self.avg_rewards['agent_0'] else 0
                avg_1 = self.avg_rewards['agent_1'][-1] if self.avg_rewards['agent_1'] else 0
                team_avg_reward = avg_0 + avg_1

                served_a0 = np.mean(self.served_stats['agent_0'][-50:])
                served_a1 = np.mean(self.served_stats['agent_1'][-50:])
                total_served = served_a0 + served_a1
                
                if self.use_vdn:
                    eps = self.agents['vdn'].epsilon
                elif getattr(self, 'use_qmix', False):
                    eps = self.agents['qmix'].epsilon
                else:
                    eps = self.agents['agent_0'].epsilon # Independentの場合は代表表示
                
                tar2_msg = " | TAR2 Shaped" if self.use_tar2 else ""
                
                # ★修正: 以前の使い慣れたフォーマットを復元し、コンソールとファイル両方に出力
                log_msg = (f"Ep {episode:4d} | StgEp: {stage_episode_count:4d} | "
                           f"AvgReward: {team_avg_reward:6.1f} | "
                           f"Total Served: {total_served:4.1f} (A0:{served_a0:.1f}, A1:{served_a1:.1f}) | "
                           f"CollRate: {collision_rate:.3f} | Wait: {avg_wait:.1f} | "
                           f"ε={eps:.3f}{tar2_msg}")
                self._log(log_msg, stage_idx, to_console=True)

                # ★デバッグ用: 100エピソードごとに配膳ヒートマップをファイルのみに描画
                heatmap_msg = "\n=== Delivery Heatmap (Total Served Count per Seat) ===\n"
                for y in range(current_env.grid_size):
                    row_str = ""
                    for x in range(current_env.grid_size):
                        count = current_env.total_delivery_heatmap[x, y]
                        if count > 0:
                            row_str += f"[{count:4d}] " # 配膳実績あり
                        elif current_env.counter_pos and (x, y) == current_env.counter_pos:
                            row_str += " [Cntr] "     # カウンター
                        elif (x, y) in current_env.obstacles:
                            row_str += "  XXXX  "     # 障害物・壁
                        elif (x, y) in current_env.seats:
                            row_str += " [   0] "     # 席（まだ一度も配膳されていない）
                        else:
                            row_str += "   ..   "     # 通路
                    heatmap_msg += row_str + "\n"
                heatmap_msg += "======================================================\n"
                self._log(heatmap_msg, stage_idx, to_console=False)

        # ======================================================================
        # ▼▼▼ 追加機能: 学習終了時に最良指標（移動平均）を出力 ▼▼▼
        # ======================================================================
        
        # 1. チーム全体の合計報酬と合計配膳数を計算
        n_eps = len(next(iter(self.episode_rewards.values())))
        team_rewards = np.zeros(n_eps)
        team_served = np.zeros(n_eps)
        
        for agent_name in self.episode_rewards:
            team_rewards += np.array(self.episode_rewards[agent_name])
            team_served += np.array(self.served_stats[agent_name])

        # 2. 移動平均の計算ヘルパー (Window=50)
        def get_best_ma(data, mode='max', window=50):
            if len(data) < window:
                return data[-1] if len(data) > 0 else 0.0
            
            # 移動平均を計算 ('valid'モードでウィンドウ分短くなる)
            ma = np.convolve(data, np.ones(window)/window, mode='valid')
            
            if mode == 'max':
                return np.max(ma)
            else: # min
                return np.min(ma)

        # 3. 各指標の最良値を算出
        best_reward_ma = get_best_ma(team_rewards, mode='max')
        best_served_ma = get_best_ma(team_served, mode='max')
        best_col_rate_ma = get_best_ma(self.collision_rates, mode='min')
        best_wait_time_ma = get_best_ma(self.avg_wait_times, mode='min')

        # 4. ログ出力 (ターミナルにも表示)
        final_msg = (f"\n{'='*70}\n"
                     f"📊 FINAL PERFORMANCE SUMMARY (Best Moving Avg over 50 eps)\n"
                     f"{'='*70}\n"
                     f"  ★ Best Team Reward:       {best_reward_ma:8.2f}\n"
                     f"  ★ Best Total Served:      {best_served_ma:8.2f} dishes\n"
                     f"  ★ Lowest Collision Rate:  {best_col_rate_ma:8.4f}\n"
                     f"  ★ Lowest Avg Wait Time:   {best_wait_time_ma:8.2f} steps\n"
                     f"{'='*70}\n")
        self._log(final_msg, stage_idx, to_console=True)

        # ======================================================================

        # 戻り値に新しい指標を追加
        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, self.collision_rates, self.avg_wait_times, current_env

    def _run_episode_collect_only(self, env):
        """
        ParallelEnv用のデータ収集ループ
        全エージェントが同時に行動決定・実行を行う
        """
        observations, infos = env.reset()
        
        states_seq = []
        actions_seq = []
        rewards_seq = []
        dones_seq = []
        global_states_seq = []

        # エージェントIDの固定順序（保存データの整合性のため）
        agent_ids = env.possible_agents # ['agent_0', 'agent_1']
        
        def get_global_state(obs_dict):
            # 辞書からリスト順に観測を取り出して結合
            return np.concatenate([obs_dict[aid] for aid in agent_ids])

        current_global_state = get_global_state(observations)
        episode_reward_sum = 0
        
        # ParallelEnv のループ
        while env.agents: # PettingZoo: エージェントが残っている限りループ
            step_actions_dict = {}
            step_states_list = []
            
            # ▼▼▼ Action Maskingの取得 ▼▼▼
            avail_actions = env.get_avail_actions()
            
            # 1. 全エージェントの行動決定 (同時)
            # 観測データの準備
            for agent_name in agent_ids:
                # 終了したエージェントは observations に含まれない場合がある
                if agent_name in observations:
                    step_states_list.append(observations[agent_name])
                else:
                    pass

            if self.use_vdn:
                # VDN: まとめて行動選択
                step_actions_dict = self.agents['vdn'].select_actions(observations, avail_actions=avail_actions)
            elif getattr(self, 'use_qmix', False):
                step_actions_dict = self.agents['qmix'].select_actions(observations, avail_actions=avail_actions)
            else:
                # Independent DQN
                for agent_name in env.agents:
                    step_actions_dict[agent_name] = self.agents[agent_name].select_action(
                        observations[agent_name],
                        avail_actions=avail_actions[agent_name]
                    )
            
            # 2. 環境を1ステップ進める (同時更新)
            next_observations, rewards, terminations, truncations, infos = env.step(step_actions_dict)
            
            # 3. データ保存 (リスト順序を揃える)
            # 現在のステップのデータ
            s_row = []
            a_row = []
            r_row = []
            d_row = []
            
            for agent_name in agent_ids:
                # 状態
                if agent_name in observations:
                    s_row.append(observations[agent_name])
                else:
                    s_dim = env.observation_space(agent_name).shape[0]
                    s_row.append(np.zeros(s_dim))

                # 行動
                if agent_name in step_actions_dict:
                    a_row.append(step_actions_dict[agent_name])
                else:
                    a_row.append(0) # No-op or Dummy
            
                # 報酬
                r = rewards.get(agent_name, 0.0)
                r_row.append(r)
                episode_reward_sum += r
             
                # 終了判定
                term = terminations.get(agent_name, False)
                trunc = truncations.get(agent_name, False)
                d_row.append(term or trunc)

            global_states_seq.append(current_global_state)
            states_seq.append(np.array(s_row))
            actions_seq.append(np.array(a_row))
            rewards_seq.append(np.array(r_row))
            dones_seq.append(np.array(d_row))
            
            # 次のステップへ
            observations = next_observations
            
            if env.agents:
                current_global_state = get_global_state(observations)
            else:
                pass

        # ループ終了後の処理
        # T+1 個目の s_{T} (Terminal State) を追加しておく
        
        final_s_row = []
        for agent_name in agent_ids:
            s_dim = env.observation_space(agent_name).shape[0]
            final_s_row.append(np.zeros(s_dim))
        states_seq.append(np.array(final_s_row))
        
        # global state も同様に最後を追加
        s_dim = env.observation_space(agent_ids[0]).shape[0]
        final_global = np.zeros(s_dim * len(agent_ids))
        global_states_seq.append(final_global)

        return {
            'states': np.array(states_seq), # (T+1, N, Dim)
            'actions': np.array(actions_seq),       # (T, N)
            'rewards': np.array(rewards_seq),       # (T, N)
            'dones': np.array(dones_seq),           # (T, N)
            'global_states': np.array(global_states_seq), # (T+1, GlobalDim)
            'total_reward': episode_reward_sum
        }

    def _store_and_train_agents(self, trajectory, shaped_rewards):
        """バッファ保存と学習"""
        T = len(trajectory['actions'])
        agent_ids = ['agent_0', 'agent_1']
        
        for t in range(T):
            s_t = trajectory['states'][t]
            a_t = trajectory['actions'][t]
            ns_t = trajectory['states'][t+1]
            d_t = trajectory['dones'][t]

            g_s_t = trajectory['global_states'][t]
            g_ns_t = trajectory['global_states'][t+1]
            
            if shaped_rewards is not None:
                r_t = shaped_rewards[t]
            else:
                r_t = trajectory['rewards'][t]

            s_dict = {name: s_t[i] for i, name in enumerate(agent_ids)}
            a_dict = {name: a_t[i] for i, name in enumerate(agent_ids)}
            r_dict = {name: r_t[i] for i, name in enumerate(agent_ids)}
            ns_dict = {name: ns_t[i] for i, name in enumerate(agent_ids)}
            d_dict = {name: bool(d_t[i]) for i, name in enumerate(agent_ids)}

            if getattr(self, 'use_qmix', False):
                self.agents['qmix'].store_transition(s_dict, a_dict, r_dict, ns_dict, d_dict, g_s_t, g_ns_t)
                self.agents['qmix'].train()
            elif self.use_vdn:
                self.agents['vdn'].store_transition(s_dict, a_dict, r_dict, ns_dict, d_dict)
                self.agents['vdn'].train()
            else:
                for i, name in enumerate(agent_ids):
                    self.agents[name].store_transition(
                        s_dict[name], a_dict[name], r_dict[name], ns_dict[name], d_dict[name]
                    )
                    self.agents[name].train()
    
    def save_agents(self, directory="models", suffix=""):
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.use_qmix:
            path = f"{directory}/qmix_agent{suffix}.pth"
            self.agents['qmix'].save_model(path)
            print(f"Saved QMIX model to {path}")
        elif self.use_vdn:
            path = f"{directory}/vdn_agent{suffix}.pth"
            self.agents['vdn'].save_model(path)
            print(f"Saved VDN model to {path}")
        else:
            # Independent DQN
            for agent_name, agent in self.agents.items():
                path = f"{directory}/{agent_name}{suffix}.pth"
                agent.save_model(path)
                print(f"Saved {agent_name} model to {path}")

    # ▼▼▼ 統計テストモード用メソッド ▼▼▼
    def evaluate(self, env, agents, num_episodes=100):
        """
        学習済みモデルを用いて指定エピソード数だけ実行し、統計指標を出力する
        """
        # 1. 探索率(epsilon)を0にして「活用」のみにする
        for key, agent in agents.items():
            if hasattr(agent, 'epsilon'):
                agent.epsilon = 0.0
        
        # 統計用リスト
        total_rewards = []
        served_counts = []
        collision_rates = []
        wait_times = []

        print(f"\nRunning {num_episodes} episodes for benchmark...")

        for ep in range(num_episodes):
            # エピソード実行
            # self.agents を一時的に切り替えて _run_episode_collect_only を再利用する
            original_agents = self.agents
            self.agents = agents 

            trajectory = self._run_episode_collect_only(env)
            
            # self.agents の復元
            self.agents = original_agents

            # --- 指標の収集 ---
            # 1. 報酬
            total_rewards.append(trajectory['total_reward'])

            # 2. 配膳数
            ep_served = sum(env.served_count.values())
            served_counts.append(ep_served)

            # 3. 衝突率
            total_collisions = sum(env.collision_count.values())
            total_steps = env.num_moves * len(env.possible_agents)
            c_rate = total_collisions / total_steps if total_steps > 0 else 0.0
            collision_rates.append(c_rate)

            # 4. 平均待ち時間
            if hasattr(env, 'completed_wait_times') and len(env.completed_wait_times) > 0:
                avg_wait = np.mean(env.completed_wait_times)
            else:
                avg_wait = 0.0
            wait_times.append(avg_wait)

            # 進捗表示
            if (ep + 1) % 10 == 0:
                print(f"  Processed {ep + 1}/{num_episodes} episodes...", end='\r')

        # --- 結果の集計と出力 ---
        print(f"\n{'='*60}")
        print(f"📊 TEST RESULTS (Average over {num_episodes} episodes)")
        print(f"{'='*60}")
        
        mean_r, std_r = np.mean(total_rewards), np.std(total_rewards)
        mean_s, std_s = np.mean(served_counts), np.std(served_counts)
        mean_c, std_c = np.mean(collision_rates), np.std(collision_rates)
        mean_w, std_w = np.mean(wait_times), np.std(wait_times)

        print(f"  ★ Average Reward:       {mean_r:8.2f} ± {std_r:.2f}")
        print(f"  ★ Average Served:       {mean_s:8.2f} ± {std_s:.2f} dishes")
        print(f"  ★ Collision Rate:       {mean_c:8.4f} ± {std_c:.4f}")
        print(f"  ★ Avg Wait Time:        {mean_w:8.2f} ± {std_w:.2f} steps")
        print(f"{'='*60}")
        
        print(f"  [Best Record in Test]")
        print(f"    Max Reward: {np.max(total_rewards):.2f}")
        print(f"    Max Served: {np.max(served_counts)}")
        print(f"{'='*60}\n")