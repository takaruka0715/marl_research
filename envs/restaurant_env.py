import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import functools
import random
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces

from .customers import CustomerManager
from .layout import LayoutBuilder
from .utils_env import check_collision, get_adjacent_positions

class RestaurantEnv(AECEnv):
    metadata = {"name": "restaurant_v2_cooking", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True,
                 customer_spawn_interval=20, local_obs_size=5, coop_factor=0.0, config=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.layout_type = layout_type
        self.local_obs_size = local_obs_size

        self.possible_agents = ["agent_0", "agent_1"]
        self.n_agents = len(self.possible_agents)
        
        self.max_seats_obs = 20 
        self.seat_obs_dim = self.max_seats_obs * 4
        
        # 360度カメラ仕様：周囲8方向 × 各方向の視野ステップ
        self.num_view_directions = 8 
        self.view_dim = self.num_view_directions * self.local_obs_size 
        
        if config is not None:
            self.reward_params = {
                'delivery': config.delivery_reward,
                'pickup': config.pickup_reward,
                'collision': config.collision_penalty,
                'step_cost': config.step_cost,
                'wait_penalty': config.wait_penalty,
                'coop_bonus_threshold': config.coop_bonus_threshold
            }
            self.max_steps = config.max_steps
            self.coop_factor = config.coop_factor
        else:
            self.reward_params = {
                'delivery': 100.0, 'pickup': 50.0, 'collision': -10.0,
                'step_cost': -0.1, 'wait_penalty': -0.5, 'coop_bonus_threshold': 20.0
            }
            self.max_steps = 500
            self.coop_factor = coop_factor
        
        self.agents = self.possible_agents[:]
        
        # 観測次元の合計を計算（視野情報 40 + 基本状態 10 + 座席情報 80 + ID 2 = 132次元）
        obs_extra_dim = 10 + self.seat_obs_dim + self.n_agents 
        obs_dim = self.view_dim + obs_extra_dim
        
        self._action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Box(low=-5, high=grid_size, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.customer_manager = CustomerManager(enable_customers, customer_spawn_interval)
        self.reset()
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        
        obstacles, tables, seats, counter_pos, entrance_pos = LayoutBuilder.create_layout(
            self.layout_type, self.grid_size)
        self.obstacles = obstacles
        self.tables = tables
        self.seats = seats
        self.counter_pos = counter_pos
        self.entrance_pos = entrance_pos
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox, oy] = -1
        
        # 【修正】初期位置のランダム化：進入禁止エリア（障害物、座席、カウンター）を除外
        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        forbidden = set(self.obstacles) | set(self.seats)
        if self.counter_pos:
            forbidden.add(self.counter_pos)
        
        available_spaces = [p for p in all_positions if p not in forbidden]
        start_positions = random.sample(available_spaces, self.n_agents)
        
        self.agent_positions = {agent: start_positions[i] for i, agent in enumerate(self.agents)}
        self.agent_directions = {agent: np.random.randint(0, 4) for agent in self.agents}
        self.agent_inventory = {agent: 0 for agent in self.agents}
        
        self.kitchen_queue = []
        self.ready_dishes = 0
        self.active_orders = []
        
        self.served_count = {agent: 0 for agent in self.agents}
        self.collision_count = {agent: 0 for agent in self.agents}
        
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.customer_manager.customers = []
        self.customer_manager.customer_counter = 0
        self.customer_manager.steps_since_last_spawn = 0
        
        self.history = [{
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': self.ready_dishes
        }]

    def observe(self, agent):
        my_pos = self.agent_positions[agent]
        my_direction = self.agent_directions[agent]
        x, y = my_pos
        
        obs_grid = self.grid.copy().astype(np.float32)
        for other_agent in self.possible_agents:
            if other_agent != agent:
                ox, oy = self.agent_positions[other_agent]
                obs_grid[ox, oy] = 2
        for customer in self.customer_manager.customers:
            if customer.state in ['seated', 'ordered', 'served']:
                cx, cy = customer.position
                obs_grid[cx, cy] = 3
        for order_x, order_y in self.active_orders:
            obs_grid[order_x, order_y] = 4
        
        # 【修正】360度・視界5マス・遮蔽ありのスキャン
        # 8方向（上、右上、右、右下、下、左下、左、左上）
        scan_directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        local_obs = []
        for dx, dy in scan_directions:
            view_blocked = False
            for i in range(1, self.local_obs_size + 1):
                if view_blocked:
                    local_obs.append(-1.0) # 遮蔽後は壁と同じ値
                    continue
                
                tx, ty = x + (dx * i), y + (dy * i)
                if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
                    cell_val = obs_grid[tx, ty]
                    local_obs.append(cell_val)
                    if cell_val == -1.0: # 障害物に当たったら遮蔽フラグを立てる
                        view_blocked = True
                else:
                    local_obs.append(-1.0) # 範囲外
                    view_blocked = True
        
        # 共通観測情報の構築
        standard_obs = np.zeros(len(local_obs) + 10, dtype=np.float32)
        standard_obs[:len(local_obs)] = local_obs
        
        idx = len(local_obs)
        standard_obs[idx:idx+2] = [x / self.grid_size, y / self.grid_size]
        standard_obs[idx+2] = my_direction / 4.0
        
        standard_obs[idx+3] = len(self.active_orders) / self.max_seats_obs
        standard_obs[idx+4] = self.served_count[agent] / 20.0
        standard_obs[idx+5] = self.collision_count[agent] / 100.0
        standard_obs[idx+6] = self.agent_inventory[agent] / 4.0
        standard_obs[idx+7] = 1.0 if self.agent_inventory[agent] < 4 else 0.0
        standard_obs[idx+8] = min(self.ready_dishes, 5) / 5.0
        standard_obs[idx+9] = 1.0 if self.ready_dishes > 0 else 0.0

        # 座席情報の相対座標化
        seat_information = []
        active_customer_seats = [c.seat_position for c in self.customer_manager.customers 
                                 if c.state in ['seated', 'ordered', 'served']]

        for i in range(self.max_seats_obs):
            if i < len(self.seats):
                sx, sy = self.seats[i]
                seat_information.append((sx - x) / self.grid_size)
                seat_information.append((sy - y) / self.grid_size)
                seat_information.append(1.0 if (sx, sy) in active_customer_seats else 0.0)
                seat_information.append(1.0 if (sx, sy) in self.active_orders else 0.0)
            else:
                seat_information.extend([0.0, 0.0, 0.0, 0.0])

        full_obs = np.concatenate([standard_obs, np.array(seat_information, dtype=np.float32)])
        agent_idx = self.possible_agents.index(agent)
        agent_id_feature = np.zeros(self.n_agents, dtype=np.float32)
        agent_id_feature[agent_idx] = 1.0

        return np.concatenate([full_obs, agent_id_feature])
    
    def _move_agent(self, agent, action):
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
        
        customer_positions = [c.position for c in self.customer_manager.customers 
                            if c.state in ['seated', 'ordered', 'served']]
        other_positions = [pos for a, pos in self.agent_positions.items() if a != agent]
        
        if check_collision((new_x, new_y), self.obstacles, 
                          customer_positions, other_positions, self.seats):
            self.rewards[agent] = self.reward_params['collision']
            self.collision_count[agent] += 1
            collision = True
            new_x, new_y = x, y

        if not collision:
            self.agent_positions[agent] = (new_x, new_y)
        return collision
    
    def _process_interaction(self, agent, action):
        x, y = self.agent_positions[agent]
        
        is_near_counter = False
        if self.counter_pos:
            cx, cy = self.counter_pos
            if abs(x - cx) + abs(y - cy) <= 1:
                is_near_counter = True
        
        if is_near_counter and self.ready_dishes > 0 and self.agent_inventory[agent] < 4:
            self.ready_dishes -= 1
            self.agent_inventory[agent] += 1
            self.rewards[agent] += self.reward_params['pickup']
        
        if self.agent_inventory[agent] > 0:
            for order_pos in self.active_orders[:]:
                adjacent = get_adjacent_positions(order_pos)
                if (x, y) in adjacent:
                    self.agent_inventory[agent] -= 1
                    self.rewards[agent] += self.reward_params['delivery']
                    self.served_count[agent] += 1
                    self.active_orders.remove(order_pos)
                    
                    for customer in self.customer_manager.customers:
                        if customer.seat_position == order_pos and customer.state == 'ordered':
                            customer.state = 'served'
                            customer.wait_time = 0
                    break
        
        if action == 0:
            self.rewards[agent] += self.reward_params['step_cost']
        if len(self.active_orders) > 3:
            self.rewards[agent] += self.reward_params['wait_penalty']
        
        current_reward = self.rewards[agent]
        if current_reward >= self.reward_params['coop_bonus_threshold']:
            for other_agent in self.possible_agents:
                if other_agent != agent:
                    self.rewards[other_agent] += current_reward * self.coop_factor
    
    def step(self, action):
        agent = self.agent_selection
        if self.truncations[agent]:
            if action is not None:
                raise ValueError("Cannot step with a truncated agent")
            self.agent_selection = self._agent_selector.next()
            return
        
        self.rewards = {a: 0 for a in self.possible_agents}
        self._move_agent(agent, action)
        self._process_interaction(agent, action)
        
        self.history.append({
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': self.ready_dishes
        })
        
        self._cumulative_rewards[agent] += self.rewards[agent]
        self.num_moves += 1
        
        if agent == self.possible_agents[-1]:
            self.customer_manager.steps_since_last_spawn += 1
            if self.customer_manager.steps_since_last_spawn >= self.customer_manager.spawn_interval:
                self.customer_manager.spawn_customer(self.entrance_pos, self.seats)
                self.customer_manager.steps_since_last_spawn = 0
            
            new_orders, new_kitchen = self.customer_manager.update_customers()
            self.active_orders.extend([o for o in new_orders if o not in self.active_orders])
            self.kitchen_queue.extend(new_kitchen)
        
        for item in self.kitchen_queue[:]:
            item['time_left'] -= 1
            if item['time_left'] <= 0:
                self.kitchen_queue.remove(item)
                self.ready_dishes += 1
        
        if self.num_moves >= self.max_steps:
            self.truncations = {a: True for a in self.possible_agents}
        
        self.agent_selection = self._agent_selector.next()