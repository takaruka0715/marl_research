import numpy as np
import random
from env import RestaurantEnv
from agents import DQNAgent, SharedReplayBuffer
from .curriculum import Curriculum

class Trainer:
    """マルチエージェント学習トレーナー"""
    
    def __init__(self, num_episodes=30000, use_shared_replay=True, config=None):
        self.num_episodes = num_episodes
        self.use_shared_replay = use_shared_replay
        self.config = config
        self.curriculum = Curriculum()
        
        self.agents = {}
        self.episode_rewards = {}
        self.avg_rewards = {}
        self.served_stats = {}
    
    def train(self):
        """学習ループ実行"""
        action_dim = 4
        
        # 環境初期化
        temp_env = RestaurantEnv(layout_type='empty', local_obs_size=5, config=self.config)
        state_dim = temp_env.observation_space('agent_0').shape[0]
        print(f"State Dimension: {state_dim}")
        
        # バッファ・エージェント初期化
        shared_buffer = SharedReplayBuffer(capacity=50000) if self.use_shared_replay else None
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
                    for agent_name in self.agents:
                        self.agents[agent_name].epsilon = 0.6
                
                if stage['layout'] == 'empty' and episode >= 1500:
                    current_env.seats = [(np.random.randint(2, 13), np.random.randint(2, 13)) 
                                         for _ in range(5)]
            
            episode_reward = self._run_episode(current_env, stage)
            
            for agent_name in current_env.possible_agents:
                self.episode_rewards[agent_name].append(episode_reward[agent_name])
                self.avg_rewards[agent_name].append(np.mean(self.episode_rewards[agent_name][-50:]))
                self.served_stats[agent_name].append(current_env.served_count[agent_name])
                self.agents[agent_name].decay_epsilon()
            
            if episode % 10 == 0:
                for agent_name in current_env.possible_agents:
                    self.agents[agent_name].update_target_network()
            
            if episode % 100 == 0:
                avg_0 = self.avg_rewards['agent_0'][-1]
                avg_1 = self.avg_rewards['agent_1'][-1]
                eps = self.agents['agent_0'].epsilon
                served_0 = np.mean(self.served_stats['agent_0'][-50:])
                served_1 = np.mean(self.served_stats['agent_1'][-50:])
                print(f"Ep {episode:4d} | Avg: A0={avg_0:6.1f}, A1={avg_1:6.1f} | "
                      f"Served: A0={served_0:.1f}, A1={served_1:.1f} | ε={eps:.3f}")
        
        return self.agents, self.episode_rewards, self.avg_rewards, self.served_stats, current_env
    
    def _run_episode(self, env, stage):
        """1エピソード実行"""
        env.reset()
        episode_reward = {agent: 0 for agent in env.possible_agents}
        states = {agent: env.observe(agent) for agent in env.possible_agents}
        
        # ランダムオーダー生成（empty ステージ）
        if stage['layout'] == 'empty' and len(env.seats) > 0:
            if np.random.random() < 0.3:
                order_pos = random.choice(env.seats)
                if order_pos not in env.active_orders:
                    env.active_orders.append(order_pos)
                    env.ready_dishes += 1
        
        for step in range(600):
            agent_name = env.agent_selection
            if env.truncations.get(agent_name, False):
                env.step(None)
                continue
            
            state = states[agent_name]
            action = self.agents[agent_name].select_action(state)
            env.step(action)
            
            next_state = env.observe(agent_name)
            reward = env.rewards.get(agent_name, 0)
            done = env.truncations.get(agent_name, False)
            
            self.agents[agent_name].store_transition(state, action, reward, next_state, done)
            self.agents[agent_name].train()
            
            states[agent_name] = next_state
            episode_reward[agent_name] += reward
            
            if all(env.truncations.get(a, False) for a in env.possible_agents):
                break
        
        return episode_reward
