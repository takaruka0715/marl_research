import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import functools
import random
from collections import deque  # 到達可能領域探索用
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

from .customers import CustomerManager
from .layout import LayoutBuilder
from .utils_env import check_collision, get_adjacent_positions

class RestaurantEnv(ParallelEnv):
    metadata = {"name": "restaurant_v2_parallel", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True,
                 customer_spawn_interval=20, local_obs_size=5, coop_factor=0.0, config=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.layout_type = layout_type
        self.local_obs_size = local_obs_size

        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        self.n_agents = len(self.possible_agents)
        
        self.max_seats_obs = 20 
        
        # 【修正】座席ごとの観測次元を 6 -> 5 に最適化
        # 1.相対X
        # 2.相対Y
        # 3.注文あり (客がいるか情報はこれに含まれるとみなす)
        # 4.自分が所持している料理の宛先か
        # 5.カウンターにこの座席宛の料理があるか
        self.seat_obs_dim = self.max_seats_obs * 5
        
        self.num_view_directions = 8 
        self.view_dim = self.num_view_directions * self.local_obs_size 
        
        if config is not None:
            self.reward_params = {
                'delivery': config.delivery_reward,
                'pickup': config.pickup_reward,
                'collision': config.collision_penalty,
                'step_cost': config.step_cost,
                'wait_penalty': config.wait_penalty,
                'coop_bonus_threshold': config.coop_bonus_threshold,
                
                # [cite_start]★追加: 設定ファイルから動的ペナルティ用パラメータを読み込む [cite: 274, 275]
                'max_wait_limit': config.max_wait_limit,
                'wait_penalty_scale': config.wait_penalty_scale
            }
            self.max_steps = config.max_steps
            self.coop_factor = config.coop_factor
        else:
            self.reward_params = {
                'delivery': 100.0, 'pickup': 50.0, 'collision': -10.0,
                'step_cost': -0.1, 'wait_penalty': -0.5, 'coop_bonus_threshold': 20.0,
                # ★追加: デフォルト値
                'max_wait_limit': 50.0,
                'wait_penalty_scale': 0.1
            }
            self.max_steps = 500
            self.coop_factor = coop_factor
        
        # 観測空間の定義
        obs_extra_dim = 10 + self.seat_obs_dim + self.n_agents 
        obs_dim = self.view_dim + obs_extra_dim
        
        self.observation_spaces = {
            agent: spaces.Box(low=-5, high=grid_size, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(5) # 0-3: Move, 4: Wait (No-op)
            for agent in self.possible_agents
        }
        
        self.customer_manager = CustomerManager(enable_customers, customer_spawn_interval)
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _get_connected_free_spaces(self, forbidden_positions):
        """
        エントランスまたはカウンター周辺から到達可能な、
        「孤立していない」自由空間のリストを返す (BFS探索)
        """
        # 1. 探索の始点（シード）を決める
        start_node = self.entrance_pos if self.entrance_pos else (1, 1)
        
        # もし始点自体が障害物なら、周辺の空いているマスを探す
        if start_node in forbidden_positions:
            found = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = start_node[0] + dx, start_node[1] + dy
                    if (0 <= nx < self.grid_size and 
                        0 <= ny < self.grid_size and 
                        (nx, ny) not in forbidden_positions):
                        start_node = (nx, ny)
                        found = True
                        break
                if found: break
        
        # 2. 幅優先探索 (BFS) で到達可能エリアを特定
        reachable = []
        queue = deque([start_node])
        visited = {start_node}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右

        while queue:
            cx, cy = queue.popleft()
            # 始点自体がforbiddenでない場合のみ追加
            if (cx, cy) not in forbidden_positions:
                reachable.append((cx, cy))

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    if (nx, ny) not in visited and (nx, ny) not in forbidden_positions:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return reachable

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        
        # レイアウト生成
        obstacles, tables, seats, counter_pos, entrance_pos = LayoutBuilder.create_layout(
            self.layout_type, self.grid_size)
        self.obstacles = obstacles
        self.tables = tables
        self.seats = seats
        self.counter_pos = counter_pos
        self.entrance_pos = entrance_pos
        
        # グリッド初期化
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox, oy] = -1
        
        # エージェント初期配置
        forbidden = set(self.obstacles) | set(self.seats)
        if self.counter_pos:
            forbidden.add(self.counter_pos)
        
        available_spaces = self._get_connected_free_spaces(forbidden)

        if len(available_spaces) < self.n_agents:
            all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
            available_spaces = [p for p in all_positions if p not in forbidden]
        
        start_positions = random.sample(available_spaces, self.n_agents)
        
        self.agent_positions = {agent: start_positions[i] for i, agent in enumerate(self.agents)}
        self.agent_directions = {agent: np.random.randint(0, 4) for agent in self.agents}
        
        self.agent_inventory = {agent: [] for agent in self.agents}
        
        self.kitchen_queue = []
        self.ready_dishes = []
        
        self.active_orders = []
        self.completed_wait_times = []
        
        self.served_count = {agent: 0 for agent in self.agents}
        self.collision_count = {agent: 0 for agent in self.agents}
        
        self.customer_manager.customers = []
        self.customer_manager.customer_counter = 0
        self.customer_manager.steps_since_last_spawn = 0
        
        self.history = [{
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': self.agent_inventory.copy(),
            'ready_dishes': list(self.ready_dishes)
        }]
        
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # 1. 移動ロジック
        self._move_agents_simultaneously(actions, rewards)

        # 2. インタラクション処理
        for agent in self.agents:
            if agent in actions:
                self._process_interaction(agent, actions[agent], rewards)

        # 3. 顧客生成と更新
        self.customer_manager.steps_since_last_spawn += 1
        if self.customer_manager.steps_since_last_spawn >= self.customer_manager.spawn_interval:
            self.customer_manager.spawn_customer(self.entrance_pos, self.seats)
            self.customer_manager.steps_since_last_spawn = 0
        
        new_orders, new_kitchen = self.customer_manager.update_customers()
        self.active_orders.extend([o for o in new_orders if o not in self.active_orders])
        self.kitchen_queue.extend(new_kitchen)
        
        # 調理進行
        for item in self.kitchen_queue[:]:
            item['time_left'] -= 1
            if item['time_left'] <= 0:
                self.kitchen_queue.remove(item)
                self.ready_dishes.append(item)

        # 4. 共通ペナルティ (★修正: 動的待機ペナルティへの変更)
        # 設定ファイルから値を参照
        max_wait = self.reward_params.get('max_wait_limit', 50.0)
        scale = self.reward_params.get('wait_penalty_scale', 0.1)

        total_wait_penalty = 0.0
        
        for c in self.customer_manager.customers:
            if c.state == 'ordered':
                # 動的ペナルティ計算: 待ち時間が長いほど重くなる
                urgency = (c.wait_time / max_wait) ** 2
                total_wait_penalty -= urgency * scale

        # 全エージェントにペナルティ適用
        for agent in self.agents:
            rewards[agent] += total_wait_penalty

        # 5. ステップ数カウントと終了判定
        self.num_moves += 1
        if self.num_moves >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
            self.agents = []

        # 6. 観測生成
        observations = {}
        if self.agents:
            observations = {agent: self.observe(agent) for agent in self.agents}
        else:
             observations = {agent: self.observe(agent) for agent in self.possible_agents}

        # 履歴保存
        if self.agents:
             self.history.append({
                'agent_positions': self.agent_positions.copy(),
                'agent_directions': self.agent_directions.copy(),
                'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
                'active_orders': self.active_orders.copy(),
                'agent_inventory': self.agent_inventory.copy(),
                'ready_dishes': list(self.ready_dishes)
            })

        return observations, rewards, terminations, truncations, infos

    def _move_agents_simultaneously(self, actions, rewards):
        intended_positions = {}
        intended_directions = {}
        dir_vectors = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for agent in self.agents:
            if agent not in actions:
                intended_positions[agent] = self.agent_positions[agent]
                intended_directions[agent] = self.agent_directions[agent]
                continue

            action = actions[agent]
            curr_pos = self.agent_positions[agent]
            curr_dir = self.agent_directions[agent]
            
            next_pos = curr_pos
            next_dir = curr_dir
            
            if action == 0: # Move Forward
                dx, dy = dir_vectors[curr_dir]
                temp_x = max(0, min(self.grid_size - 1, curr_pos[0] + dx))
                temp_y = max(0, min(self.grid_size - 1, curr_pos[1] + dy))
                next_pos = (temp_x, temp_y)
                rewards[agent] += self.reward_params['step_cost']
                
            elif action == 1: # Turn Right
                next_dir = (curr_dir + 1) % 4
            elif action == 2: # Turn Left
                next_dir = (curr_dir - 1) % 4

            intended_positions[agent] = next_pos
            intended_directions[agent] = next_dir

        move_success = {agent: True for agent in self.agents}

        customer_positions = [c.position for c in self.customer_manager.customers 
                              if c.state in ['seated', 'ordered', 'served']]
        
        for agent in self.agents:
            if intended_positions[agent] == self.agent_positions[agent]:
                continue
            
            pos = intended_positions[agent]
            if (pos in self.obstacles) or (pos in self.seats) or (pos in customer_positions):
                move_success[agent] = False
                rewards[agent] += self.reward_params['collision']
                self.collision_count[agent] += 1

        dest_counts = {}
        for agent in self.agents:
            if not move_success[agent]: continue
            dest = intended_positions[agent]
            if dest not in dest_counts: dest_counts[dest] = []
            dest_counts[dest].append(agent)
        
        for dest, agents_at_dest in dest_counts.items():
            if len(agents_at_dest) > 1:
                for agent in agents_at_dest:
                    move_success[agent] = False
                    rewards[agent] += self.reward_params['collision']
                    self.collision_count[agent] += 1
        
        for i, agent_a in enumerate(self.agents):
            for agent_b in self.agents[i+1:]:
                if move_success[agent_a] and move_success[agent_b]:
                    if (intended_positions[agent_a] == self.agent_positions[agent_b] and 
                        intended_positions[agent_b] == self.agent_positions[agent_a]):
                        
                        move_success[agent_a] = False
                        move_success[agent_b] = False
                        rewards[agent_a] += self.reward_params['collision']
                        rewards[agent_b] += self.reward_params['collision']
                        self.collision_count[agent_a] += 1
                        self.collision_count[agent_b] += 1

        for agent in self.agents:
            if move_success[agent]:
                self.agent_positions[agent] = intended_positions[agent]
                self.agent_directions[agent] = intended_directions[agent]

    def _process_interaction(self, agent, action, rewards):
        x, y = self.agent_positions[agent]
        
        is_near_counter = False
        if self.counter_pos:
            cx, cy = self.counter_pos
            if abs(x - cx) + abs(y - cy) <= 1:
                is_near_counter = True
        
        # --- ピックアップ処理 ---
        if is_near_counter and len(self.ready_dishes) > 0 and len(self.agent_inventory[agent]) < 4:
            dish = self.ready_dishes.pop(0)
            self.agent_inventory[agent].append(dish)
            rewards[agent] += self.reward_params['pickup']
        
        # --- 配膳処理 ---
        if len(self.agent_inventory[agent]) > 0:
            for order_pos in self.active_orders[:]:
                adjacent = get_adjacent_positions(order_pos)
                if (x, y) in adjacent:
                    target_dish = None
                    for dish in self.agent_inventory[agent]:
                        if dish.get('target_seat') == order_pos:
                            target_dish = dish
                            break
                    
                    if target_dish:
                        self.agent_inventory[agent].remove(target_dish)
                        rewards[agent] += self.reward_params['delivery']
                        self.served_count[agent] += 1
                        if order_pos in self.active_orders:
                            self.active_orders.remove(order_pos)
                        
                        for customer in self.customer_manager.customers:
                            if customer.seat_position == order_pos and customer.state == 'ordered':
                                self.completed_wait_times.append(customer.wait_time)
                                customer.state = 'served'
                                customer.wait_time = 0
                        break
        
        current_reward = rewards[agent]
        if current_reward >= self.reward_params['coop_bonus_threshold']:
            for other_agent in self.agents:
                if other_agent != agent:
                    rewards[other_agent] += current_reward * self.coop_factor

    def observe(self, agent):
        my_pos = self.agent_positions[agent]
        my_direction = self.agent_directions[agent]
        x, y = my_pos
        
        obs_grid = self.grid.copy().astype(np.float32)
        for other_agent in self.possible_agents:
            if other_agent != agent:
                if other_agent in self.agent_positions:
                    ox, oy = self.agent_positions[other_agent]
                    obs_grid[ox, oy] = 2
        
        for customer in self.customer_manager.customers:
            if customer.state in ['seated', 'ordered', 'served']:
                cx, cy = customer.position
                obs_grid[cx, cy] = 3
        for order_x, order_y in self.active_orders:
            obs_grid[order_x, order_y] = 4
        
        scan_directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        local_obs = []
        for dx, dy in scan_directions:
            view_blocked = False
            for i in range(1, self.local_obs_size + 1):
                if view_blocked:
                    local_obs.append(-1.0)
                    continue
                
                tx, ty = x + (dx * i), y + (dy * i)
                if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
                    cell_val = obs_grid[tx, ty]
                    local_obs.append(cell_val)
                    if cell_val == -1.0: 
                        view_blocked = True
                else:
                    local_obs.append(-1.0)
                    view_blocked = True
        
        standard_obs = np.zeros(len(local_obs) + 10, dtype=np.float32)
        standard_obs[:len(local_obs)] = local_obs
        
        idx = len(local_obs)
        standard_obs[idx:idx+2] = [x / self.grid_size, y / self.grid_size]
        standard_obs[idx+2] = my_direction / 4.0
        standard_obs[idx+3] = len(self.active_orders) / self.max_seats_obs
        standard_obs[idx+4] = self.served_count[agent] / 20.0
        standard_obs[idx+5] = self.collision_count[agent] / 100.0
        standard_obs[idx+6] = len(self.agent_inventory[agent]) / 4.0
        standard_obs[idx+7] = 1.0 if len(self.agent_inventory[agent]) < 4 else 0.0
        standard_obs[idx+8] = min(len(self.ready_dishes), 5) / 5.0
        standard_obs[idx+9] = 1.0 if len(self.ready_dishes) > 0 else 0.0

        seat_information = []
        active_customer_seats = [c.seat_position for c in self.customer_manager.customers 
                                 if c.state in ['seated', 'ordered', 'served']]
        
        my_target_seats = [d['target_seat'] for d in self.agent_inventory[agent]]
        counter_target_seats = [d['target_seat'] for d in self.ready_dishes]

        for i in range(self.max_seats_obs):
            if i < len(self.seats):
                sx, sy = self.seats[i]
                seat_information.append((sx - x) / self.grid_size)
                seat_information.append((sy - y) / self.grid_size)
                # "客がいるか" は削除し、"注文があるか"に集約
                seat_information.append(1.0 if (sx, sy) in self.active_orders else 0.0)
                seat_information.append(1.0 if (sx, sy) in my_target_seats else 0.0)
                seat_information.append(1.0 if (sx, sy) in counter_target_seats else 0.0)
            else:
                # パディングも5次元に変更
                seat_information.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        full_obs = np.concatenate([standard_obs, np.array(seat_information, dtype=np.float32)])
        agent_idx = self.possible_agents.index(agent)
        agent_id_feature = np.zeros(self.n_agents, dtype=np.float32)
        agent_id_feature[agent_idx] = 1.0

        return np.concatenate([full_obs, agent_id_feature])