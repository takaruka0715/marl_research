import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import functools
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
from .customers import CustomerManager
from .layout import LayoutBuilder

class RestaurantEnv(AECEnv):
    metadata = {"name": "restaurant_v2_cooking", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True,
                 customer_spawn_interval=20, local_obs_size=5, coop_factor=0.5, config=None):
        super().__init__()
        self.grid_size = grid_size
        self.layout_type = layout_type
        self.local_obs_size = local_obs_size
        self.coop_factor = coop_factor
        self.config = config
        
        # 報酬設定の読み込み [cite: 137, 138]
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
            self.reward_params = {
                'delivery': 100.0, 'pickup': 30.0, 'collision': -10.0,
                'step_cost': -0.05, 'wait_penalty': 0.0, 'coop_bonus_threshold': 20.0
            }
            self.max_steps = 600
        
        self.customer_manager = CustomerManager(enable_customers, customer_spawn_interval)
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        
        # Trainerの期待する action_dim=4 に合わせる [cite: 79]
        self._action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: spaces.Box(low=-5, high=grid_size, shape=(15,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # 属性の初期化
        self.history = [] 
        self.reset()

    def get_agent_pos(self, agent):
        """Trainer.py が座標判定に使用 [cite: 91]"""
        return self.agent_positions[agent]

    def get_average_wait_time(self):
        """Trainer.py が統計取得に使用 [cite: 83]"""
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
        self.infos = {agent: {} for agent in self.possible_agents}
        
        # レイアウトの取得 
        obs, tables, seats, counter, entrance = LayoutBuilder.create_layout(self.layout_type, self.grid_size)
        self.obstacles, self.tables, self.seats, self.counter_pos, self.entrance_pos = obs, tables, seats, counter, entrance
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for ox, oy in self.obstacles: 
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox, oy] = -1
        
        # 初期座標設定 [cite: 144]
        self.agent_positions = {agent: [(self.grid_size-2, 1), (self.grid_size-2, self.grid_size-2)][i] for i, agent in enumerate(self.agents)}
        self.agent_directions = {agent: 0 for agent in self.agents}
        self.agent_inventory = {agent: 0 for agent in self.agents}
        self.ready_dishes = 0
        self.active_orders = []
        self.served_count = {agent: 0 for agent in self.agents}
        self.collision_count = {agent: 0 for agent in self.agents}
        self.num_moves = 0
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.customer_manager.customers = []
        self.history = [] # ここでリストを空にする
        self._record_history() # 履歴の初回記録

    def observe(self, agent):
        """Trainer.py のサボり判定 に合わせ index 2 に inventory を配置"""
        pos = self.agent_positions[agent]
        inv = self.agent_inventory[agent]
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = pos[0] / self.grid_size
        obs[1] = pos[1] / self.grid_size
        obs[2] = float(inv)
        obs[3] = self.agent_directions[agent]
        obs[4] = self.ready_dishes / 5.0
        return obs

    def step(self, action):
        agent = self.agent_selection
        if self.truncations[agent]:
            self.agent_selection = self._agent_selector.next()
            return
        
        self.rewards[agent] = 0
        self._move_agent(agent, action)
        self._process_interaction(agent, action)
        
        # 顧客の生成と注文管理（全エージェントが動いた後に更新） [cite: 169, 170]
        if agent == self.possible_agents[-1]:
            self.customer_manager.steps_since_last_spawn += 1
            if self.customer_manager.steps_since_last_spawn >= self.customer_manager.spawn_interval:
                self.customer_manager.spawn_customer(self.entrance_pos, self.seats)
                self.customer_manager.steps_since_last_spawn = 0
            
            new_orders, new_kitchen = self.customer_manager.update_customers()
            for o in new_orders:
                if o not in self.active_orders: self.active_orders.append(o)
            if new_kitchen: self.ready_dishes += len(new_kitchen)

        self._cumulative_rewards[agent] += self.rewards[agent]
        self.num_moves += 1
        if self.num_moves >= self.max_steps:
            self.truncations = {a: True for a in self.possible_agents}
        
        self._record_history()
        self.agent_selection = self._agent_selector.next()

    def _move_agent(self, agent, action):
        x, y = self.agent_positions[agent]
        direction = self.agent_directions[agent]
        new_x, new_y = x, y
        
        if action == 0: # 前進
            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
            new_x, new_y = x + dr, y + dc
        elif action == 1: # 右回転
            self.agent_directions[agent] = (direction + 1) % 4
        elif action == 2: # 左回転
            self.agent_directions[agent] = (direction - 1) % 4
        
        new_x = max(0, min(self.grid_size - 1, new_x))
        new_y = max(0, min(self.grid_size - 1, new_y))
        
        other_agents = [self.agent_positions[a] for a in self.possible_agents if a != agent]
        if (new_x, new_y) in self.obstacles or (new_x, new_y) in other_agents:
            self.rewards[agent] += self.reward_params['collision']
            self.collision_count[agent] += 1
        else:
            self.agent_positions[agent] = (new_x, new_y)

    def _process_interaction(self, agent, action):
        pos = self.agent_positions[agent]
        
        # 【ピックアップ】カウンターに隣接（距離1）
        if self.counter_pos and abs(pos[0]-self.counter_pos[0]) + abs(pos[1]-self.counter_pos[1]) <= 1:
            if self.ready_dishes > 0 and self.agent_inventory[agent] < 1:
                self.ready_dishes -= 1
                self.agent_inventory[agent] = 1
                self.rewards[agent] += self.reward_params['pickup']
        
        # 【配膳】注文がある座席（椅子）に隣接（距離1）
        if self.agent_inventory[agent] > 0:
            for order_pos in self.active_orders[:]:
                # ロボットは椅子の上に乗れないため、隣接した状態で配膳を行う
                if abs(pos[0]-order_pos[0]) + abs(pos[1]-order_pos[1]) <= 1:
                    self.agent_inventory[agent] = 0
                    self.served_count[agent] += 1
                    self.rewards[agent] += self.reward_params['delivery']
                    self.active_orders.remove(order_pos)
                    # 顧客の状態を「配膳済み」に更新
                    for c in self.customer_manager.customers:
                        if c.seat_position == order_pos:
                            c.state = 'served'
                    break
        
        # ステップコストの付与
        self.rewards[agent] += self.reward_params['step_cost']

    def _record_history(self):
        """gif_maker.py が参照するデータ構造を保存 """
        self.history.append({
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': self.ready_dishes
        })

    def observation_space(self, agent):
        from collections import namedtuple
        Space = namedtuple('Space', ['shape'])
        return Space(shape=(15,))