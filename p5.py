import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import functools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ==============================================================================
# 1. 環境定義 (Environment)
# ==============================================================================

class Customer:
    """客クラス"""
    def __init__(self, customer_id, position, seat_position):
        self.id = customer_id
        self.position = position
        self.seat_position = seat_position
        self.state = 'entering'
        self.wait_time = 0
        self.order_wait_time = np.random.randint(5, 15)
        self.has_ordered = False
        self.served = False

class RestaurantEnv(AECEnv):
    metadata = {"name": "restaurant_v2_cooking", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True, 
                 customer_spawn_interval=20, local_obs_size=5, coop_factor=0.5):
        super().__init__()
        
        # ==============================================================================
        # 【変更点】報酬パラメータを一か所にまとめました
        #  ここを書き換えるだけで、すべての報酬・ペナルティ設定が変更されます
        # ==============================================================================
        self.reward_params = {
            'delivery': 100.0,      # 配膳成功時の報酬 (ゴール)
            'pickup': 50.0,         # キッチンで料理を受け取った時の報酬
            'collision': -10.0,     # 壁・客・他エージェントへの衝突ペナルティ
            'step_cost': -0.1,      # 1ステップごとの移動コスト (動かない場合は0にする)
            'wait_penalty': -0.5,   # 客を待たせすぎている時のペナルティ
            'coop_bonus_threshold': 20.0 # 協調報酬を発生させる最低報酬ライン
        }

        self.grid_size = grid_size
        self.layout_type = layout_type
        self.enable_customers = enable_customers
        self.customer_spawn_interval = customer_spawn_interval
        self.max_steps = 500
        
        self.local_obs_size = local_obs_size
        self.coop_factor = coop_factor
        
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        
        self._action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        
        obs_extra_dim = 12
        obs_dim = self.local_obs_size + obs_extra_dim
        self._observation_spaces = {
            agent: spaces.Box(low=-5, high=grid_size, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.reset()
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def _create_restaurant_layout(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.obstacles = []
        self.tables = []
        self.seats = []
        self.counter_pos = None
        self.entrance_pos = None
        
        if self.layout_type == 'empty':
            self.counter_pos = (self.grid_size//2, self.grid_size//2)
            self.obstacles.append(self.counter_pos)
        
        elif self.layout_type == 'basic':
            self._add_walls()
            self._add_counter(7, 1, length=3, horizontal=False)
            self._add_table(3, 3)
            self._add_table(3, 8)
            self._add_table(8, 3)
            self._add_table(8, 8)
            self._add_table(6, 11)
            self.entrance_pos = (1, 7)
        
        elif self.layout_type == 'complex':
            self._add_walls()
            self._add_counter(7, 1, length=5, horizontal=False)
            for i in range(3):
                self._add_obstacle(12, 5+i)
            for tx in [2, 6, 10]:
                for ty in [2, 6, 10]:
                    self._add_table(tx, ty)
            self._add_table(4, 12)
            self.entrance_pos = (1, 7)
        
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox, oy] = -1
    
    def _add_walls(self):
        for i in range(self.grid_size):
            self.obstacles.append((0, i))
            self.obstacles.append((self.grid_size-1, i))
            self.obstacles.append((i, 0))
            self.obstacles.append((i, self.grid_size-1))
    
    def _add_table(self, x, y):
        if x < self.grid_size-1 and y < self.grid_size-1:
            self.tables.append((x, y))
            for dx in [0, 1]:
                for dy in [0, 1]:
                    self.obstacles.append((x+dx, y+dy))
            seats = [(x-1, y), (x+2, y), (x, y-1), (x, y+2)]
            for sx, sy in seats:
                if (0 < sx < self.grid_size-1 and 0 < sy < self.grid_size-1 and
                    (sx, sy) not in self.obstacles):
                    self.seats.append((sx, sy))
    
    def _add_counter(self, x, y, length=3, horizontal=True):
        self.counter_pos = (x, y) 
        for i in range(length):
            if horizontal:
                self.obstacles.append((x, y+i))
            else:
                self.obstacles.append((x+i, y))
    
    def _add_obstacle(self, x, y):
        self.obstacles.append((x, y))
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        
        self._create_restaurant_layout()
        
        self.agent_positions = {}
        self.agent_directions = {}
        
        self.agent_inventory = {agent: 0 for agent in self.possible_agents}
        self.kitchen_queue = []
        self.ready_dishes = 0
        
        start_positions = [(6, 2), (8, 2)]
        for i, agent in enumerate(self.agents):
            self.agent_positions[agent] = start_positions[i]
            self.agent_directions[agent] = np.random.randint(0, 4)
        
        self.customers = []
        self.customer_counter = 0
        self.steps_since_last_customer = 0
        self.active_orders = []
        
        self.served_count = {agent: 0 for agent in self.possible_agents}
        self.collision_count = {agent: 0 for agent in self.possible_agents}
        
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.history = [{
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': self.ready_dishes
        }]
    
    def _spawn_customer(self):
        if not self.enable_customers or len(self.seats) == 0:
            return
        occupied_seats = [c.seat_position for c in self.customers if c.state in ['seated', 'ordered', 'waiting_for_food']]
        available_seats = [s for s in self.seats if s not in occupied_seats]
        
        if len(available_seats) > 0 and self.entrance_pos:
            seat = random.choice(available_seats)
            customer = Customer(self.customer_counter, self.entrance_pos, seat)
            self.customers.append(customer)
            self.customer_counter += 1
    
    def _update_customers(self):
        if self.enable_customers:
            for customer in self.customers[:]:
                if customer.state == 'entering':
                    customer.position = customer.seat_position
                    customer.state = 'seated'
                elif customer.state == 'seated':
                    customer.wait_time += 1
                    if customer.wait_time >= customer.order_wait_time and not customer.has_ordered:
                        customer.state = 'ordered'
                        customer.has_ordered = True
                        if customer.seat_position not in self.active_orders:
                            self.active_orders.append(customer.seat_position)
                            self.kitchen_queue.append({'time_left': 5})
                elif customer.state == 'served':
                    customer.wait_time += 1
                    if customer.wait_time >= 15:
                        customer.state = 'leaving'
                elif customer.state == 'leaving':
                    self.customers.remove(customer)

        for item in self.kitchen_queue[:]:
            item['time_left'] -= 1
            if item['time_left'] <= 0:
                self.kitchen_queue.remove(item)
                self.ready_dishes += 1
    
    def get_global_grid(self, agent):
        observation_grid = self.grid.copy().astype(np.float32)
        for other_agent in self.possible_agents:
            if other_agent != agent:
                ox, oy = self.agent_positions[other_agent]
                observation_grid[ox, oy] = 2
        for customer in self.customers:
            if customer.state in ['seated', 'ordered', 'served']:
                cx, cy = customer.position
                observation_grid[cx, cy] = 3
        for order_x, order_y in self.active_orders:
            observation_grid[order_x, order_y] = 4
        return observation_grid

    def observe(self, agent):
        my_pos = self.agent_positions[agent]
        my_direction = self.agent_directions[agent]
        x, y = my_pos
        
        full_grid = self.get_global_grid(agent)
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = dir_vectors[my_direction]
        
        local_obs = []
        for i in range(self.local_obs_size):
            tx = x + (dx * i)
            ty = y + (dy * i)
            if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
                local_obs.append(full_grid[tx, ty])
            else:
                local_obs.append(-1.0)
                
        local_obs = np.array(local_obs, dtype=np.float32)
        
        obs = np.zeros(len(local_obs) + 12, dtype=np.float32)
        obs[:len(local_obs)] = local_obs
        
        idx = len(local_obs)
        obs[idx:idx+2] = my_pos
        obs[idx+2] = my_direction
        
        current_inv = self.agent_inventory[agent]
        target_vec = [0, 0]
        
        if current_inv == 0:
            if self.counter_pos:
                cx, cy = self.counter_pos
                target_vec = [cx - my_pos[0], cy - my_pos[1]]
        else:
            if len(self.active_orders) > 0:
                nearest_order = min(self.active_orders, 
                                key=lambda o: abs(o[0]-my_pos[0]) + abs(o[1]-my_pos[1]))
                target_vec = [nearest_order[0] - my_pos[0], nearest_order[1] - my_pos[1]]
        
        obs[idx+3:idx+5] = target_vec
        obs[idx+5] = len(self.active_orders)
        obs[idx+6] = self.served_count[agent]
        obs[idx+7] = self.collision_count[agent]
        obs[idx+8] = current_inv / 4.0
        obs[idx+9] = 1.0 if current_inv < 4 else 0.0
        obs[idx+10] = min(self.ready_dishes, 5) / 5.0
        obs[idx+11] = 1.0 if self.ready_dishes > 0 else 0.0
        
        return obs
    
    def step(self, action):
        agent = self.agent_selection
        
        if self.truncations[agent]:
            if action is not None:
                raise ValueError("Cannot step with a truncated agent")
            self.agent_selection = self._agent_selector.next()
            return
        
        self.rewards = {a: 0 for a in self.possible_agents}
        
        x, y = self.agent_positions[agent]
        direction = self.agent_directions[agent]
        new_x, new_y = x, y
        new_direction = direction
        
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        if action == 0: 
            dx, dy = dir_vectors[direction]
            new_x = max(0, min(self.grid_size - 1, x + dx))
            new_y = max(0, min(self.grid_size - 1, y + dy))
        elif action == 1:
            new_direction = (direction + 1) % 4
        elif action == 2:
            new_direction = (direction - 1) % 4
        
        self.agent_directions[agent] = new_direction
        collision = False
        
        # --- 衝突判定 (変数参照に変更) ---
        if (new_x, new_y) in self.obstacles:
            self.rewards[agent] = self.reward_params['collision']
            self.collision_count[agent] += 1
            collision = True
            new_x, new_y = x, y
        
        customer_positions = [c.position for c in self.customers if c.state in ['seated', 'ordered', 'served']]
        if (new_x, new_y) in customer_positions:
            self.rewards[agent] = self.reward_params['collision']
            self.collision_count[agent] += 1
            collision = True
            new_x, new_y = x, y
            
        other_positions = [pos for a, pos in self.agent_positions.items() if a != agent]
        if (new_x, new_y) in other_positions:
            self.rewards[agent] = self.reward_params['collision']
            self.collision_count[agent] += 1
            collision = True
            new_x, new_y = x, y
        
        if not collision:
            self.agent_positions[agent] = (new_x, new_y)
            
        # ==============================================================================
        # インタラクション処理 (変数参照に変更 & 衝突時でも判定可能に)
        # ==============================================================================
        
        # 1. キッチンでの料理ピックアップ判定
        is_near_counter = False
        if self.counter_pos:
            cx, cy = self.counter_pos
            if abs(new_x - cx) + abs(new_y - cy) <= 1:
                is_near_counter = True
        
        if is_near_counter:
            if self.ready_dishes > 0 and self.agent_inventory[agent] < 4:
                self.ready_dishes -= 1
                self.agent_inventory[agent] += 1
                self.rewards[agent] += self.reward_params['pickup'] # 変数参照
        
        # 2. 客への配膳判定
        if self.agent_inventory[agent] > 0:
            for order_pos in self.active_orders[:]:
                ox, oy = order_pos
                adjacent = [(ox-1, oy), (ox+1, oy), (ox, oy-1), (ox, oy+1)]
                
                if (new_x, new_y) in adjacent:
                    self.agent_inventory[agent] -= 1
                    self.rewards[agent] += self.reward_params['delivery'] # 変数参照
                    self.served_count[agent] += 1
                    self.active_orders.remove(order_pos)
                    
                    for customer in self.customers:
                        if customer.seat_position == order_pos and customer.state == 'ordered':
                            customer.state = 'served'
                            customer.wait_time = 0
                    break 
        
        # 移動コスト (変数参照)
        if action == 0:
            self.rewards[agent] += self.reward_params['step_cost']
        
        # 待機ペナルティ (変数参照)
        if len(self.active_orders) > 3:
            self.rewards[agent] += self.reward_params['wait_penalty']
            
        # 協調報酬 (変数参照)
        current_reward = self.rewards[agent]
        if current_reward >= self.reward_params['coop_bonus_threshold']: 
            for other_agent in self.possible_agents:
                if other_agent != agent:
                    self.rewards[other_agent] += current_reward * self.coop_factor

        self.history.append({
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': self.ready_dishes
        })
        
        self._cumulative_rewards[agent] += self.rewards[agent]
        self.num_moves += 1
        
        if agent == self.possible_agents[-1]:
            self.steps_since_last_customer += 1
            if self.steps_since_last_customer >= self.customer_spawn_interval:
                self._spawn_customer()
                self.steps_since_last_customer = 0
            self._update_customers()
        
        if self.num_moves >= self.max_steps:
            self.truncations = {a: True for a in self.possible_agents}
        
        self.agent_selection = self._agent_selector.next()


# ==============================================================================
# 2. 学習用クラス (Agents & Buffer)
# ==============================================================================

class SharedReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
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
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.use_shared_buffer:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def train(self):
        buffer_size = len(self.memory)
        if buffer_size < self.batch_size:
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
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ==============================================================================
# 3. 学習ループ (Training Loop)
# ==============================================================================

def train_restaurant_agents(num_episodes=10000, use_shared_replay=True):
    action_dim = 4
    
    curriculum = [
        {'episodes': (0, 1500), 'layout': 'empty', 'customers': False, 'spawn_interval': 999, 'description': 'Basic Movement'},
        {'episodes': (1500, 3000), 'layout': 'empty', 'customers': False, 'spawn_interval': 10, 'description': 'Random Delivery'},
        {'episodes': (3000, 5500), 'layout': 'basic', 'customers': False, 'spawn_interval': 15, 'description': 'Navigation Obstacles'},
        {'episodes': (5500, 10000), 'layout': 'basic', 'customers': True, 'spawn_interval': 30, 'description': 'Simple Service'},
        {'episodes': (10000, 15000), 'layout': 'complex', 'customers': False, 'spawn_interval': 15, 'description': 'Complex Navigation'},
        {'episodes': (15000, 30000), 'layout': 'complex', 'customers': True, 'spawn_interval': 20, 'description': 'Full Service'},
    ]
    
    temp_env = RestaurantEnv(layout_type='empty', local_obs_size=5)
    state_dim = temp_env.observation_space('agent_0').shape[0]
    print(f"State Dimension (Included Inventory info): {state_dim}")
    
    shared_buffer = SharedReplayBuffer(capacity=50000) if use_shared_replay else None
    
    agents = {
        agent_name: DQNAgent(state_dim, action_dim, shared_buffer=shared_buffer) 
        for agent_name in temp_env.possible_agents
    }
    
    episode_rewards = {agent: [] for agent in temp_env.possible_agents}
    avg_rewards = {agent: [] for agent in temp_env.possible_agents}
    served_stats = {agent: [] for agent in temp_env.possible_agents}
    
    current_env = None
    current_stage = None
    
    for episode in range(num_episodes):
        stage = next((s for s in curriculum if s['episodes'][0] <= episode < s['episodes'][1]), None)
        
        if stage != current_stage:
            prev_stage_desc = current_stage['description'] if current_stage else "None"
            current_stage = stage
            
            print(f"\n{'='*70}")
            print(f"=== Curriculum Change: {prev_stage_desc} -> {stage['description']} ===")
            print(f"=== Episode {episode} / {num_episodes} ===")
            print(f"{'='*70}")
            
            current_env = RestaurantEnv(
                layout_type=stage['layout'],
                enable_customers=stage['customers'],
                customer_spawn_interval=stage['spawn_interval'],
                local_obs_size=5,
                coop_factor=0.5
            )
            
            if episode > 0:
                print("!!! Resetting Epsilon to 0.6 for new stage adaptation !!!")
                for agent_name in agents:
                    agents[agent_name].epsilon = 0.6
            
            if stage['layout'] == 'empty' and episode >= 1500:
                current_env.seats = [(np.random.randint(2, 13), np.random.randint(2, 13)) for _ in range(5)]
        
        current_env.reset()
        episode_reward = {agent: 0 for agent in current_env.possible_agents}
        states = {agent: current_env.observe(agent) for agent in current_env.possible_agents}
        
        # 学習用ランダムオーダー生成
        if stage['layout'] == 'empty' and episode >= 1500 and len(current_env.seats) > 0:
            if np.random.random() < 0.3:
                order_pos = random.choice(current_env.seats)
                if order_pos not in current_env.active_orders:
                    current_env.active_orders.append(order_pos)
                    current_env.ready_dishes += 1
        
        for step in range(600):
            agent_name = current_env.agent_selection
            if current_env.truncations.get(agent_name, False):
                current_env.step(None)
                continue
            
            state = states[agent_name]
            action = agents[agent_name].select_action(state)
            
            current_env.step(action)
            
            next_state = current_env.observe(agent_name)
            reward = current_env.rewards.get(agent_name, 0)
            done = current_env.truncations.get(agent_name, False)
            
            agents[agent_name].store_transition(state, action, reward, next_state, done)
            agents[agent_name].train()
            
            states[agent_name] = next_state
            episode_reward[agent_name] += reward
            
            if all(current_env.truncations.get(a, False) for a in current_env.possible_agents):
                break
        
        for agent_name in current_env.possible_agents:
            episode_rewards[agent_name].append(episode_reward[agent_name])
            avg_rewards[agent_name].append(np.mean(episode_rewards[agent_name][-50:]))
            served_stats[agent_name].append(current_env.served_count[agent_name])
            agents[agent_name].decay_epsilon()
        
        if episode % 10 == 0:
            for agent_name in current_env.possible_agents:
                agents[agent_name].update_target_network()
        
        if episode % 100 == 0:
            avg_0 = avg_rewards['agent_0'][-1]
            avg_1 = avg_rewards['agent_1'][-1]
            eps = agents['agent_0'].epsilon
            served_0 = np.mean(served_stats['agent_0'][-50:])
            served_1 = np.mean(served_stats['agent_1'][-50:])
            print(f"Ep {episode:4d} | Avg R: A0={avg_0:6.1f}, A1={avg_1:6.1f} | Served: A0={served_0:.1f}, A1={served_1:.1f} | ε={eps:.3f}")
    
    return agents, episode_rewards, avg_rewards, served_stats, current_env


# ==============================================================================
# 4. 可視化関数 (Visualization)
# ==============================================================================

def plot_learning_curves(episode_rewards, avg_rewards, served_stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    stage_boundaries = [1500, 3000, 5500, 7500, 8750]
    
    for i, agent in enumerate(['agent_0', 'agent_1']):
        axes[0, i].plot(episode_rewards[agent], alpha=0.2, color='gray')
        axes[0, i].plot(avg_rewards[agent], linewidth=2, color='blue', label='Avg Reward')
        for b in stage_boundaries:
            axes[0, i].axvline(x=b, color='red', linestyle='--', alpha=0.3)
        axes[0, i].set_title(f'{agent} Rewards')
        axes[0, i].legend()
        
        axes[1, i].plot(served_stats[agent], alpha=0.3, color='green')
        window = 50
        if len(served_stats[agent]) >= window:
            served_ma = [np.mean(served_stats[agent][max(0, j-window):j+1]) for j in range(len(served_stats[agent]))]
            axes[1, i].plot(served_ma, linewidth=2, color='darkgreen', label='Avg Served')
        for b in stage_boundaries:
            axes[1, i].axvline(x=b, color='red', linestyle='--', alpha=0.3)
        axes[1, i].set_title(f'{agent} Served Count')
        axes[1, i].legend()
        
    plt.tight_layout()
    plt.savefig('restaurant_learning_curves_cooking.png')
    print("Saved learning curves.")
    plt.close()

def create_restaurant_gif(env, agents, filename='restaurant_service_cooking.gif'):
    env.reset(seed=42)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def draw_frame(frame_data):
        ax.clear()
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#f5f5dc')
        
        # 障害物とカウンター
        for x, y in env.obstacles:
            color = '#8B4513' if (x,y) in env.tables else 'gray' if env.counter_pos and abs(x-env.counter_pos[0])<=1 else 'black'
            ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
            
            # キッチンカウンターに料理があるか表示
            if env.counter_pos and x == env.counter_pos[0] and y == env.counter_pos[1]:
                ready_count = frame_data['ready_dishes']
                if ready_count > 0:
                    ax.text(y, x, f"Food:{ready_count}", ha='center', va='center', color='yellow', fontweight='bold', fontsize=8)
            
        for sx, sy in env.seats:
            ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
            
        for c in frame_data['customers']:
            if c['state'] in ['seated', 'ordered', 'served']:
                color = 'orange' if c['state'] == 'ordered' else 'lightgreen'
                ax.add_patch(Circle((c['position'][1], c['position'][0]), 0.3, facecolor=color, ec='black'))
        
        for ox, oy in frame_data['active_orders']:
            ax.add_patch(Circle((oy, ox), 0.4, facecolor='yellow', alpha=0.5, ec='red'))
            
        agent_colors = ['red', 'blue']
        
        # Y軸反転に合わせて、0(上)を270度、2(下)を90度に変更
        dir_angles = [270, 0, 90, 180]
        
        for idx, agent in enumerate(env.possible_agents):
            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            inv = frame_data['agent_inventory'][agent]
            
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            if inv > 0:
                ax.text(pos[1], pos[0], str(inv), color='white', ha='center', va='center', fontweight='bold')
            
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, alpha=0.4, color='black'))
            
            dx_map = [-1, 0, 1, 0]
            dy_map = [0, 1, 0, -1]
            dx, dy = dx_map[d], dy_map[d]
            end_row = pos[0] + dx * (env.local_obs_size - 0.5)
            end_col = pos[1] + dy * (env.local_obs_size - 0.5)
            ax.plot([pos[1], end_col], [pos[0], end_row], 
                    color=agent_colors[idx], linestyle='--', alpha=0.5, linewidth=2)

        ax.invert_yaxis()
        ax.set_title(f'Step: {len(env.history)} | ReadyDishes: {frame_data["ready_dishes"]}')

    for step in range(400):
        agent_name = env.agent_selection
        if env.truncations.get(agent_name, False):
            env.step(None)
            continue
        
        state = env.observe(agent_name)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agents[agent_name].device)
            action = agents[agent_name].q_network(state_tensor).argmax().item()
        env.step(action)
        if all(env.truncations.get(a, False) for a in env.possible_agents):
            break
            
    ani = animation.FuncAnimation(fig, draw_frame, frames=env.history[::2], interval=150)
    ani.save(filename, writer='pillow', fps=6)
    print(f"Saved GIF to {filename}")
    plt.close()


# ==============================================================================
# 5. メイン実行 (Main)
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Restaurant Robot: Cooking & Pickup Simulation")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    trained_agents, episode_rewards, avg_rewards, served_stats, final_env = train_restaurant_agents(
        num_episodes=30000,
        use_shared_replay=True
    )
    
    print("\nPlotting learning curves...")
    plot_learning_curves(episode_rewards, avg_rewards, served_stats)
    
    print("\nGenerating animation...")
    create_restaurant_gif(final_env, trained_agents)
    
    print("\nAll Done.")