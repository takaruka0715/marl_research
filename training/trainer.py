import numpy as np
import torch
import random
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

        # TAR2用バッファ
        self.tar2 = None
        self.tar2_buffer = []
    
    def train(self):
        """学習ループ実行"""
        action_dim = 5 # ParallelEnvに変更したためアクション次元を確認 (Wait含むなら5、移動のみなら4)
                       # ※Env側で spaces.Discrete(5) に設定したため 5 に修正
        
        # 初期ステージ取得
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
        
        print(f"\n{'='*70}")
        print(f"=== STARTING STAGE: {current_stage['description']} ===")
        print(f"{'='*70}")
        
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
                print(f"\n{'='*70}")
                print(f"🔄 CURRICULUM PROGRESSION")
                print(f"   From: {current_stage['description']}")
                print(f"   To:   {new_stage['description']}")
                print(f"   Why:  {reason}")
                print(f"   Perf: {current_served_performance:.1f}")
                print(f"{'='*70}")
                
                current_stage = new_stage
                current_env = RestaurantEnv(
                    layout_type=current_stage['layout'],
                    enable_customers=current_stage['customers'],
                    customer_spawn_interval=current_stage['spawn_interval'],
                    local_obs_size=5,
                    coop_factor=self.config.coop_factor,
                    config=self.config
                )
                stage_episode_count = 0
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
                # ParallelEnvでは個別に報酬が出るが、チーム評価用に合計を2分割して記録（従来互換）
                # もしくは trajectory['rewards'] から個別の合計を出すのが正確
                agent_idx = current_env.possible_agents.index(agent_name)
                agent_total_r = np.sum(trajectory_data['rewards'][:, agent_idx])
                
                self.episode_rewards[agent_name].append(agent_total_r)
                self.avg_rewards[agent_name].append(np.mean(self.episode_rewards[agent_name][-50:]))
                self.served_stats[agent_name].append(current_env.served_count[agent_name])

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
            
            # ログ表示
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
                
                print(f"Ep {episode:4d} | StgEp: {stage_episode_count:4d} | "
                      f"AvgReward: {team_avg_reward:6.1f} | "
                      f"Total Served: {total_served:4.1f} (A0:{served_a0:.1f}, A1:{served_a1:.1f}) | "
                      f"ε={eps:.3f}{tar2_msg}")
        
        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, current_env

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
        # ParallelEnvでは dict で返るため、順序を保証する必要がある
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
            
            # 1. 全エージェントの行動決定 (同時)
            # ParallelEnvなので、この時点での observations は全員「移動前」の状態
            
            # 観測データの準備
            for agent_name in agent_ids:
                # 終了したエージェントは observations に含まれない場合がある
                if agent_name in observations:
                    step_states_list.append(observations[agent_name])
                else:
                    # 既にDoneのエージェント: ゼロ埋めなどで対処するか、
                    # ここでは簡単のため直前の値を保持するか、もしくはスキップ。
                    # 通常、env.agents に入っていなければ行動不要。
                    pass

            if self.use_vdn:
                # VDN: まとめて行動選択
                # observations は {agent_id: obs} なのでそのまま渡せる
                step_actions_dict = self.agents['vdn'].select_actions(observations)
            elif getattr(self, 'use_qmix', False):
                step_actions_dict = self.agents['qmix'].select_actions(observations)
            else:
                # Independent DQN
                for agent_name in env.agents:
                    step_actions_dict[agent_name] = self.agents[agent_name].select_action(observations[agent_name])
            
            # 2. 環境を1ステップ進める (同時更新)
            next_observations, rewards, terminations, truncations, infos = env.step(step_actions_dict)
            
            # 3. データ保存 (リスト順序を揃える)
            # 全エージェント分のデータを1行の配列として保存する
            
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
                    # エージェントがいなくなった場合（途中離脱）、ダミーデータを入れるか、
                    # 最後の観測を使う。ここでは前のループで確保したshapeに合わせてゼロ埋め推奨だが
                    # 簡易的に zeros を入れる
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
            
            # global state 更新 (エージェントが減っても shape を維持するための工夫が必要だが、
            # 今回の環境は全員同時に終わる設定なので、observations から再構築でOK)
            # ただし、全員終了して empty になった場合は next_obs が空になる
            if env.agents:
                current_global_state = get_global_state(observations)
            else:
                # 終了時は次状態のグローバルステートは不要（あるいは全部0）
                # store_transition で next_state を使うため、便宜上最後の state をコピーするかゼロ埋め
                pass

        # ループ終了後の処理
        # store_and_train で next_state を使うため、最後の next_state も保存する必要があるが、
        # 配列長を合わせるため、呼び出し元で t と t+1 をスライスして使う。
        # ここでは states_seq には T ステップ分の s_t が入っている。
        # T+1 個目の s_{T} (Terminal State) を追加しておく
        
        # 終了時の観測（ゼロ埋め or 最後のobs）を作成
        final_s_row = []
        for agent_name in agent_ids:
            # 完全に終了しているので env.agents は空。
            # ゼロベクトルを追加
            s_dim = env.observation_space(agent_name).shape[0]
            final_s_row.append(np.zeros(s_dim))
        states_seq.append(np.array(final_s_row))
        
        # global state も同様に最後を追加
        s_dim = env.observation_space(agent_ids[0]).shape[0]
        final_global = np.zeros(s_dim * len(agent_ids))
        global_states_seq.append(final_global)

        return {
            'states': np.array(states_seq),         # (T+1, N, Dim)
            'actions': np.array(actions_seq),       # (T, N)
            'rewards': np.array(rewards_seq),       # (T, N)
            'dones': np.array(dones_seq),           # (T, N)
            'global_states': np.array(global_states_seq), # (T+1, GlobalDim)
            'total_reward': episode_reward_sum
        }

    def _store_and_train_agents(self, trajectory, shaped_rewards):
        """バッファ保存と学習 (変更なし・ただし辞書作成ロジックは確認)"""
        # states_seq の長さは T+1, actions_seq は T
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