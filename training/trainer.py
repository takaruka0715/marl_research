import numpy as np
import torch
import csv
import os
from envs import RestaurantEnv
from agents import DQNAgent, VDNAgent, SharedReplayBuffer
from .curriculum import Curriculum

class Trainer:
    def __init__(self, num_episodes=15000, use_shared_replay=True, use_vdn=False, use_tar2=False, config=None):
        self.num_episodes = num_episodes
        self.use_shared_replay = use_shared_replay
        self.use_vdn = use_vdn
        self.config = config
        self.curriculum = Curriculum()
        self.agents = {}
        self.episode_rewards = {'agent_0': [], 'agent_1': []}
        self.avg_rewards = {'agent_0': [], 'agent_1': []}
        self.served_stats = {'agent_0': [], 'agent_1': []}
        self.total_served_history = [] 
        self.stats_log = []
        self.stage_transitions = [] 

    def _run_episode_collect_only(self, env):
        env.reset()
        states_seq, actions_seq, rewards_seq, dones_seq = [], [], [], []
        states = {a: env.observe(a) for a in env.possible_agents}
        reward_sum, cancelled_total = 0, 0
        prev_positions = {a: None for a in env.possible_agents}

        for step in range(self.config.max_steps):
            actions = {}
            if self.use_vdn:
                actions = self.agents['vdn'].select_actions(states)
            else:
                for a in env.possible_agents:
                    actions[a] = self.agents[a].select_action(states[a])
            
            # --- 修正: 報酬の初期化をTrainer側で管理し、シェイピング報酬を加算 ---
            for a in env.possible_agents:
                env.rewards[a] = 0 # ステップ開始時にリセット
                
                # 所持ペナルティ
                if states[a][2] > 0: 
                    env.rewards[a] += self.config.holding_item_step_cost
                
                # 静止ペナルティ
                curr_pos = env.get_agent_pos(a)
                if curr_pos == prev_positions[a]: 
                    env.rewards[a] += self.config.idle_penalty
                prev_positions[a] = curr_pos

            # 環境の更新
            for a in env.possible_agents:
                if env.agent_selection == a: 
                    env.step(actions[a])
            
            # 顧客キャンセルペナルティの適用
            timeout_num = env.check_and_handle_timeouts()
            if timeout_num > 0:
                cancelled_total += timeout_num
                for a in env.possible_agents: 
                    env.rewards[a] += self.config.penalty_customer_left

            # データの記録
            step_s, step_a, step_r, step_d = [], [], [], []
            for a in env.possible_agents:
                step_s.append(env.observe(a))
                step_a.append(actions[a])
                r = env.rewards.get(a, 0)
                step_r.append(r)
                step_d.append(env.truncations.get(a, False))
                reward_sum += r
            
            states_seq.append(np.array(step_s))
            actions_seq.append(np.array(step_a))
            rewards_seq.append(np.array(step_r))
            dones_seq.append(np.array(step_d))

            if all(env.truncations.values()): break
            states = {a: env.observe(a) for a in env.possible_agents}
        
        return {
            'states': np.array(states_seq), 
            'actions': np.array(actions_seq), 
            'rewards': np.array(rewards_seq), 
            'dones': np.array(dones_seq), 
            'total_reward': reward_sum, 
            'cancelled_count': cancelled_total
        }

    def train(self):
        current_stage = self.curriculum.get_current_stage()
        self.stage_transitions.append((0, current_stage['description']))
        current_env = RestaurantEnv(
            layout_type=current_stage['layout'], 
            enable_customers=current_stage['customers'],
            customer_spawn_interval=current_stage['spawn_interval'], 
            config=self.config
        )
        
        state_dim = 15
        action_dim = 4
        shared_buffer = SharedReplayBuffer(capacity=50000) if self.use_shared_replay else None
        
        if self.use_vdn:
            self.agents = {'vdn': VDNAgent(state_dim, action_dim, num_agents=2, shared_buffer=shared_buffer)}
        else:
            self.agents = {a: DQNAgent(state_dim, action_dim, shared_buffer=shared_buffer) for a in current_env.possible_agents}
        
        stage_ep_count = 0
    
        for episode in range(self.num_episodes):
            avg_served = np.mean(self.total_served_history[-50:]) if self.total_served_history else 0
            should_proceed, _ = self.curriculum.check_progression(avg_served, stage_ep_count)
            
            if should_proceed:
                current_stage = self.curriculum.get_current_stage()
                self.stage_transitions.append((episode, current_stage['description']))
                current_env = RestaurantEnv(layout_type=current_stage['layout'], config=self.config)
                stage_ep_count = 0
                reset_eps = 0.5 
                if self.use_vdn: self.agents['vdn'].epsilon = max(self.agents['vdn'].epsilon, reset_eps)
                else: [setattr(a, 'epsilon', max(a.epsilon, reset_eps)) for a in self.agents.values()]
                print(f"\n>>> STAGE MOVED: {current_stage['description']}")

            traj = self._run_episode_collect_only(current_env)
            stage_ep_count += 1
            served_total = sum(current_env.served_count.values())
            self.total_served_history.append(served_total)
            
            for a in current_env.possible_agents:
                self.episode_rewards[a].append(traj['total_reward'] / 2)
                self.avg_rewards[a].append(np.mean(self.episode_rewards[a][-50:]))
                self.served_stats[a].append(current_env.served_count[a])
            
            self._store_and_train_agents(traj)
            
            if self.use_vdn: self.agents['vdn'].decay_epsilon()
            else: [a.decay_epsilon() for a in self.agents.values()]
            
            if episode % 10 == 0: [a.update_target_network() for a in self.agents.values()]
            
            if episode % 100 == 0:
                avg_r = self.avg_rewards['agent_0'][-1]
                eps = self.agents['vdn'].epsilon if self.use_vdn else self.agents['agent_0'].epsilon
                cancel = traj.get('cancelled_count', 0)
                print(f"Ep {episode:4d} | StgEp: {stage_ep_count:4d} | AvgReward: {avg_r:6.1f} | "
                      f"Served: {avg_served:4.1f} | Cancel: {int(cancel):d} | ε={eps:.3f}")

        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, self.stats_log, self.stage_transitions, current_env

    def _store_and_train_agents(self, traj):
        T = len(traj['states'])
        for t in range(T - 1):
            s_d = {f'agent_{i}': traj['states'][t][i] for i in range(2)}
            a_d = {f'agent_{i}': traj['actions'][t][i] for i in range(2)}
            r_d = {f'agent_{i}': traj['rewards'][t][i] for i in range(2)}
            ns_d = {f'agent_{i}': traj['states'][t+1][i] for i in range(2)}
            d_d = {f'agent_{i}': bool(traj['dones'][t][i]) for i in range(2)}
            if self.use_vdn:
                self.agents['vdn'].store_transition(s_d, a_d, r_d, ns_d, d_d)
                self.agents['vdn'].train()
            else:
                for i in range(2):
                    n = f'agent_{i}'
                    self.agents[n].store_transition(s_d[n], a_d[n], r_d[n], ns_d[n], d_d[n])
                    self.agents[n].train()