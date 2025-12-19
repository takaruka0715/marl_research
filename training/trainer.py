import numpy as np
import torch
import csv
import os
from env import RestaurantEnv
from agents import DQNAgent, VDNAgent, SharedReplayBuffer
from .curriculum import Curriculum

class Trainer:
    def __init__(self, num_episodes=15000, use_shared_replay=True, use_vdn=False, use_tar2=False, config=None):
        self.num_episodes = num_episodes
        self.use_shared_replay = use_shared_replay
        self.use_vdn = use_vdn
        self.use_tar2 = False 
        self.config = config
        self.curriculum = Curriculum()
        
        self.agents = {}
        self.episode_rewards = {'agent_0': [], 'agent_1': []}
        self.avg_rewards = {'agent_0': [], 'agent_1': []}
        self.served_stats = {'agent_0': [], 'agent_1': []}
        self.total_served_history = [] 
        self.stats_log = []
        self.stage_transitions = [] 

    def train(self):
        current_stage = self.curriculum.get_current_stage()
        self.stage_transitions.append((0, current_stage['description']))
        
        current_env = RestaurantEnv(
            layout_type=current_stage['layout'],
            enable_customers=current_stage['customers'],
            customer_spawn_interval=current_stage['spawn_interval'],
            config=self.config
        )

        state_dim = current_env.observation_space('agent_0').shape[0]
        action_dim = 4
        shared_buffer = SharedReplayBuffer(capacity=50000) if self.use_shared_replay else None
        
        if self.use_vdn:
            self.agents = {'vdn': VDNAgent(state_dim, action_dim, num_agents=2, shared_buffer=shared_buffer)}
        else:
            self.agents = {a: DQNAgent(state_dim, action_dim, shared_buffer=shared_buffer) for a in current_env.possible_agents}
        
        stage_ep_count = 0

        for episode in range(self.num_episodes):
            avg_served = np.mean(self.total_served_history[-50:]) if self.total_served_history else 0
            should_proceed, reason = self.curriculum.check_progression(avg_served, stage_ep_count)

            if should_proceed:
                current_stage = self.curriculum.get_current_stage()
                self.stage_transitions.append((episode, current_stage['description']))
                current_env = RestaurantEnv(layout_type=current_stage['layout'], config=self.config)
                stage_ep_count = 0
                
                # εリセット
                reset_eps = 0.5 
                if self.use_vdn: self.agents['vdn'].epsilon = max(self.agents['vdn'].epsilon, reset_eps)
                else:
                    for a in self.agents.values(): a.epsilon = max(a.epsilon, reset_eps)
                print(f"\n>>> STAGE MOVED: {current_stage['description']} | ε reset to {reset_eps}")

            traj = self._run_episode_collect_only(current_env)
            stage_ep_count += 1
            
            served_total = sum(current_env.served_count.values()) if isinstance(current_env.served_count, dict) else current_env.served_count
            self.total_served_history.append(served_total)
            
            coll = sum(current_env.collision_count.values()) if isinstance(getattr(current_env, 'collision_count', 0), dict) else getattr(current_env, 'collision_count', 0)
            wait = np.mean(list(current_env.get_average_wait_time().values())) if hasattr(current_env, 'get_average_wait_time') else 0

            for a in current_env.possible_agents:
                self.episode_rewards[a].append(traj['total_reward'] / 2)
                self.avg_rewards[a].append(np.mean(self.episode_rewards[a][-50:]))
                self.served_stats[a].append(current_env.served_count[a] if isinstance(current_env.served_count, dict) else 0)

            self.stats_log.append({
                'episode': int(episode), 'served_count': float(served_total),
                'cancelled_count': float(traj.get('cancelled_count', 0)),
                'success_rate': float(served_total / (served_total + traj.get('cancelled_count', 0) + 1e-6)),
                'avg_wait_time': float(wait), 'collisions': float(coll)
            })

            self._store_and_train_agents(traj)
            if self.use_vdn: self.agents['vdn'].decay_epsilon()
            else: [self.agents[a].decay_epsilon() for a in current_env.possible_agents]
            
            if episode % 10 == 0:
                for a in self.agents.values(): a.update_target_network()
            
            # --- 表示形式を以前のスタイルに戻す ---
            if episode % 100 == 0:
                avg_r = self.avg_rewards['agent_0'][-1]
                eps = self.agents['vdn'].epsilon if self.use_vdn else self.agents['agent_0'].epsilon
                cancel = traj.get('cancelled_count', 0)
                # 1行に全ての重要指標を並べる
                print(f"Ep {episode:4d} | StgEp: {stage_ep_count:4d} | AvgReward: {avg_r:6.1f} | "
                      f"Served: {avg_served:4.1f} | Cancel: {cancel:d} | ε={eps:.3f}")

        self._save_stats_to_csv()
        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, self.stats_log, self.stage_transitions, current_env

    def _run_episode_collect_only(self, env):
        env.reset()
        states_seq, actions_seq, rewards_seq, dones_seq = [], [], [], []
        states = {a: env.observe(a) for a in env.possible_agents}
        reward_sum, cancelled = 0, 0
        prev_positions = {a: None for a in env.possible_agents}

        for step in range(self.config.max_steps):
            actions = {}
            if self.use_vdn: actions = self.agents['vdn'].select_actions(states)
            else: [actions.update({a: self.agents[a].select_action(states[a])}) for a in env.possible_agents]
            
            for a in env.possible_agents:
                # サボり防止
                if states[a][2] > 0: env.rewards[a] += self.config.holding_item_step_cost
                curr_pos = env.get_agent_pos(a) if hasattr(env, 'get_agent_pos') else None
                if curr_pos == prev_positions[a]: env.rewards[a] += self.config.idle_penalty
                prev_positions[a] = curr_pos

                if env.agent_selection == a: env.step(actions[a])
            
            if hasattr(env, 'check_and_handle_timeouts'):
                timeout_num = env.check_and_handle_timeouts()
                if timeout_num > 0:
                    cancelled += timeout_num
                    for a in env.possible_agents: env.rewards[a] += self.config.penalty_customer_left

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
        
        return {'states': np.array(states_seq), 'actions': np.array(actions_seq), 'rewards': np.array(rewards_seq), 
                'dones': np.array(dones_seq), 'total_reward': reward_sum, 'cancelled_count': cancelled}

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

    def _save_stats_to_csv(self):
        mode = "vdn" if self.use_vdn else "dqn"
        path = f"results/stats_{mode}.csv"
        os.makedirs("results", exist_ok=True)
        if self.stats_log:
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.stats_log[0].keys())
                w.writeheader()
                w.writerows(self.stats_log)