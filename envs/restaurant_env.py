import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces

from .customers import CustomerManager
from .layout import LayoutBuilder
from .utils_env import check_collision, get_adjacent_positions

class RestaurantEnv(AECEnv):
    metadata = {"name": "restaurant_v2_cooking", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True,
                 customer_spawn_interval=20, local_obs_size=5, coop_factor=0.5, config=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.layout_type = layout_type
        self.local_obs_size = local_obs_size
        self.coop_factor = coop_factor
        
        # config があれば使う
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
        else:
            # デフォルト
            self.reward_params = {
                'delivery': 100.0, 'pickup': 50.0, 'collision': -10.0,
                'step_cost': -0.1, 'wait_penalty': -0.5, 'coop_bonus_threshold': 20.0
            }
            self.max_steps = 500
        
        self.customer_manager = CustomerManager(enable_customers, customer_spawn_interval)
        
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
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        
        # レイアウト生成
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
        
        self.agent_positions = {agent: [(6, 2), (8, 2)][i] for i, agent in enumerate(self.agents)}
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
        
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = dir_vectors[my_direction]
        
        local_obs = []
        for i in range(self.local_obs_size):
            tx = x + (dx * i)
            ty = y + (dy * i)
            if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
                local_obs.append(obs_grid[tx, ty])
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
        
        if current_inv == 0 and self.counter_pos:
            cx, cy = self.counter_pos
            target_vec = [cx - my_pos[0], cy - my_pos[1]]
        elif current_inv > 0 and len(self.active_orders) > 0:
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
    
    def _move_agent(self, agent, action):
        """エージェント移動と衝突処理"""
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
        """ピックアップ・配膳・報酬処理"""
        x, y = self.agent_positions[agent]
        
        # ピックアップ
        is_near_counter = False
        if self.counter_pos:
            cx, cy = self.counter_pos
            if abs(x - cx) + abs(y - cy) <= 1:
                is_near_counter = True
        
        if is_near_counter and self.ready_dishes > 0 and self.agent_inventory[agent] < 4:
            self.ready_dishes -= 1
            self.agent_inventory[agent] += 1
            self.rewards[agent] += self.reward_params['pickup']
        
        # 配膳
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
        
        # ペナルティ
        if action == 0:
            self.rewards[agent] += self.reward_params['step_cost']
        if len(self.active_orders) > 3:
            self.rewards[agent] += self.reward_params['wait_penalty']
        
        # 協調ボーナス
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
        
        # キッチン処理
        for item in self.kitchen_queue[:]:
            item['time_left'] -= 1
            if item['time_left'] <= 0:
                self.kitchen_queue.remove(item)
                self.ready_dishes += 1
        
        if self.num_moves >= self.max_steps:
            self.truncations = {a: True for a in self.possible_agents}
        
        self.agent_selection = self._agent_selector.next()
