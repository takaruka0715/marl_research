import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
from .customers import CustomerManager
from .layout import LayoutBuilder

class RestaurantEnv(AECEnv):
    metadata = {"name": "restaurant_v2_cooking", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True,
                 customer_spawn_interval=20, local_obs_size=5, config=None):
        super().__init__()
        self.grid_size = grid_size
        self.layout_type = layout_type
        self.config = config
        
        # 高速化フラグ：学習中は False に設定
        self.record_enabled = False 
        
        # 報酬設定の読み込み
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
            p_limit = config.customer_patience_limit
        else:
            self.reward_params = {
                'delivery': 100.0, 'pickup': 30.0, 'collision': -10.0,
                'step_cost': -0.05, 'wait_penalty': 0.0, 'coop_bonus_threshold': 20.0
            }
            self.max_steps = 600
            p_limit = 200
        
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        
        # PettingZoo/Trainer 互換の設定
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: spaces.Box(low=-5, high=grid_size, shape=(15,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.customer_manager = CustomerManager(enable_customers, customer_spawn_interval, patience_limit=p_limit)
        self.history = [] 
        self.reset()

    # PettingZoo標準のメソッド
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_agent_pos(self, agent):
        return self.agent_positions[agent]

    def check_and_handle_timeouts(self):
        return self.last_step_cancelled

    def get_average_wait_time(self):
        if not self.customer_manager.customers:
            return {'all': 0.0}
        waits = [c.wait_time for c in self.customer_manager.customers]
        return {'all': np.mean(waits)}

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        
        # レイアウトの取得
        obs, tables, seats, counter, entrance = LayoutBuilder.create_layout(self.layout_type, self.grid_size)
        self.obstacles, self.tables, self.seats, self.counter_pos, self.entrance_pos = obs, tables, seats, counter, entrance
        self.obstacles_set = set(self.obstacles)
        
        # 初期座標設定
        self.agent_positions = {agent: [(self.grid_size-2, 1), (self.grid_size-2, self.grid_size-2)][i] for i, agent in enumerate(self.agents)}
        self.agent_directions = {agent: 0 for agent in self.agents}
        self.agent_inventory = {agent: 0 for agent in self.agents}
        self.ready_dishes = 0
        self.active_orders = []
        self.served_count = {agent: 0 for agent in self.agents}
        self.last_step_cancelled = 0
        self.num_moves = 0
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.customer_manager.customers = []
        self.history = [] 
        if self.record_enabled:
            self._record_history()

    def observe(self, agent):
        pos = self.agent_positions[agent]
        inv = self.agent_inventory[agent]
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = pos[0] / self.grid_size
        obs[1] = pos[1] / self.grid_size
        obs[2] = float(inv)
        return obs

    def step(self, action):
        agent = self.agent_selection
        if self.truncations[agent]:
            self.agent_selection = self._agent_selector.next()
            return
        
        self.rewards[agent] = 0
        self._move_agent(agent, action)
        self._process_interaction(agent)
        
        if agent == self.possible_agents[-1]:
            self.customer_manager.steps_since_last_spawn += 1
            if self.customer_manager.steps_since_last_spawn >= self.customer_manager.spawn_interval:
                self.customer_manager.spawn_customer(self.entrance_pos, self.seats)
                self.customer_manager.steps_since_last_spawn = 0
            
            new_orders, new_kitchen, cancel_num = self.customer_manager.update_customers()
            self.last_step_cancelled = cancel_num 
            
            for o in new_orders:
                if o not in self.active_orders: self.active_orders.append(o)
            if new_kitchen: self.ready_dishes += len(new_kitchen)

        self.num_moves += 1
        if self.num_moves >= self.max_steps:
            self.truncations = {a: True for a in self.possible_agents}
        
        if self.record_enabled:
            self._record_history()
        self.agent_selection = self._agent_selector.next()

    def _move_agent(self, agent, action):
        x, y = self.agent_positions[agent]
        direction = self.agent_directions[agent]
        new_x, new_y = x, y
        
        if action == 0:
            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
            new_x, new_y = x + dr, y + dc
            
            other_agents = [self.agent_positions[a] for a in self.possible_agents if a != agent]
            if (new_x, new_y) in self.obstacles_set or (new_x, new_y) in other_agents:
                self.rewards[agent] += self.reward_params['collision']
            elif 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                self.agent_positions[agent] = (new_x, new_y)
                
        elif action == 1: self.agent_directions[agent] = (direction + 1) % 4
        elif action == 2: self.agent_directions[agent] = (direction - 1) % 4

    def _process_interaction(self, agent):
        pos = self.agent_positions[agent]
        if self.counter_pos and abs(pos[0]-self.counter_pos[0]) + abs(pos[1]-self.counter_pos[1]) <= 1:
            if self.ready_dishes > 0 and self.agent_inventory[agent] < 1:
                self.ready_dishes -= 1
                self.agent_inventory[agent] = 1
                self.rewards[agent] += self.reward_params['pickup']
        
        if self.agent_inventory[agent] > 0:
            for op in self.active_orders[:]:
                if abs(pos[0]-op[0]) + abs(pos[1]-op[1]) <= 1:
                    self.agent_inventory[agent] = 0
                    self.served_count[agent] += 1
                    self.rewards[agent] += self.reward_params['delivery']
                    self.active_orders.remove(op)
                    for c in self.customer_manager.customers:
                        if c.seat_position == op: c.state = 'served'
                    break
        self.rewards[agent] += self.reward_params['step_cost']

    def _record_history(self):
        self.history.append({
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': self.ready_dishes
        })