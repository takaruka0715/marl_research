# ================================================
# FILE: envs/restaurant_env.py
# ================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import functools
import random
from collections import deque
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

from .customers import CustomerManager
from .layout import LayoutBuilder
from .utils_env import check_collision, get_adjacent_positions

class RestaurantEnv(ParallelEnv):
    metadata = {"name": "restaurant_v2_parallel", "render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=15, layout_type='basic', enable_customers=True,
                 customer_spawn_interval=20, local_obs_size=5, coop_factor=0.0, 
                 min_customer_dist=0, max_customer_dist=float('inf'), num_food_types=3, config=None,
                 spawn_mode='random', difficulty_categories=None, difficulty_thresholds=None): # ★修正
        super().__init__()
        
        self.grid_size = grid_size
        self.layout_type = layout_type
        self.local_obs_size = local_obs_size
        
        self.min_customer_dist = min_customer_dist
        self.max_customer_dist = max_customer_dist # ★追加

        # ★追加: Stage4で到達難易度カテゴリごとに均等に顧客を出すための設定
        self.spawn_mode = spawn_mode
        self.difficulty_categories = difficulty_categories or ['easy', 'medium', 'hard']
        self.difficulty_thresholds = difficulty_thresholds
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        self.n_agents = len(self.possible_agents)
        
        self.max_seats_obs = 20 
        self.num_food_types = num_food_types # ★修正
        
        self.seat_obs_dim = self.max_seats_obs * 6 
        
        self.num_view_directions = 8 
        self.view_dim = self.num_view_directions * self.local_obs_size 
        
        if config is not None:
            self.reward_params = {
                'delivery': config.delivery_reward,
                'pickup': config.pickup_reward,
                'collision': config.collision_penalty,
                'step_cost': config.step_cost,
                'coop_bonus_threshold': config.coop_bonus_threshold,
                'max_wait_limit': getattr(config, 'max_wait_limit', 50.0),
                'wait_penalty_scale': getattr(config, 'wait_penalty_scale', 0.1),
                'urgency_bonus_scale': getattr(config, 'urgency_bonus_scale', 10.0),
                'holding_penalty': getattr(config, 'holding_penalty', -0.01)
            }
            self.max_steps = config.max_steps
            self.coop_factor = config.coop_factor
        else:
            self.reward_params = {
                'delivery': 100.0, 'pickup': 50.0, 'collision': -10.0,
                'step_cost': -0.1, 
                'coop_bonus_threshold': 20.0,
                'max_wait_limit': 50.0,
                'wait_penalty_scale': 0.1,
                'urgency_bonus_scale': 10.0,
                'holding_penalty': -0.01 
            }
            self.max_steps = 500
            self.coop_factor = coop_factor
        
        obs_extra_dim = 12 + self.seat_obs_dim + self.n_agents 
        obs_dim = self.view_dim + obs_extra_dim
        
        self.observation_spaces = {
            agent: spaces.Box(low=-5, high=grid_size, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # ★変更点: アクション空間を拡張 (移動4 + 特定料理最大3)
        self.action_spaces = {
            agent: spaces.Discrete(4 + 3) # ★常に最大(7)に固定
            for agent in self.possible_agents
        }
        
        self.customer_manager = CustomerManager(enable_customers, customer_spawn_interval, num_food_types=self.num_food_types) # ★修正
        
        # ★デバッグ用: 全エピソードを通じた配膳ヒートマップ
        self.total_delivery_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # ★追加: 出現数・注文数・未配膳数も記録し、席別成功率を評価できるようにする
        self.total_customer_spawn_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.total_order_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.total_unserved_order_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _get_connected_free_spaces(self, forbidden_positions):
        start_node = self.entrance_pos if self.entrance_pos else (1, 1)
        
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
                if found:
                    break
        
        reachable = []
        queue = deque([start_node])
        visited = {start_node}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) not in forbidden_positions:
                reachable.append((cx, cy))

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    if (nx, ny) not in visited and (nx, ny) not in forbidden_positions:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return reachable


    def _compute_shortest_path_from_counter(self):
        """
        カウンター隣接マスを始点として、障害物・座席・カウンターを避けた最短距離をBFSで計算する。
        座席そのものは進入不可なので、配膳可能な「座席の隣接マス」までの距離を後で参照する。
        """
        distances = {}
        if self.counter_pos is None:
            return distances

        blocked = set(self.obstacles) | set(self.seats)
        blocked.add(self.counter_pos)

        start_positions = [
            pos for pos in get_adjacent_positions(self.counter_pos)
            if (0 <= pos[0] < self.grid_size and
                0 <= pos[1] < self.grid_size and
                pos not in blocked)
        ]

        queue = deque()
        for pos in start_positions:
            distances[pos] = 0
            queue.append(pos)

        while queue:
            x, y = queue.popleft()
            for nx, ny in get_adjacent_positions((x, y)):
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                if (nx, ny) in blocked:
                    continue
                if (nx, ny) in distances:
                    continue

                distances[(nx, ny)] = distances[(x, y)] + 1
                queue.append((nx, ny))

        return distances

    def _compute_seat_difficulty_info(self):
        """
        各席について、カウンターから配膳可能隣接マスまでの最短経路距離を計算し、
        easy / medium / hard に分類する。

        difficulty_thresholds が指定されている場合:
            easy:   distance <= easy_threshold
            medium: distance <= medium_threshold
            hard:   otherwise

        指定されていない場合:
            到達可能席の距離分布を3分割し、自動で easy / medium / hard に分類する。
            これによりマップ拡大後もカテゴリが空になりにくい。
        """
        cell_distances = self._compute_shortest_path_from_counter()
        seat_path_distances = {}

        for seat in self.seats:
            delivery_cells = [
                pos for pos in get_adjacent_positions(seat)
                if (0 <= pos[0] < self.grid_size and
                    0 <= pos[1] < self.grid_size and
                    pos in cell_distances)
            ]

            if delivery_cells:
                seat_path_distances[seat] = min(cell_distances[pos] for pos in delivery_cells)
            else:
                seat_path_distances[seat] = float('inf')

        finite_distances = sorted(
            dist for dist in seat_path_distances.values()
            if np.isfinite(dist)
        )

        if not finite_distances:
            seat_difficulties = {seat: 'hard' for seat in self.seats}
            seats_by_difficulty = {'easy': [], 'medium': [], 'hard': list(self.seats)}
            return seat_path_distances, seat_difficulties, seats_by_difficulty

        if self.difficulty_thresholds is not None:
            easy_threshold = self.difficulty_thresholds.get('easy', finite_distances[0])
            medium_threshold = self.difficulty_thresholds.get('medium', finite_distances[-1])
        else:
            # 距離分布を三分割する。席数が少ない場合でも安全にインデックスを丸める。
            n = len(finite_distances)
            easy_idx = min(max(int(np.ceil(n / 3.0)) - 1, 0), n - 1)
            medium_idx = min(max(int(np.ceil(2 * n / 3.0)) - 1, 0), n - 1)
            easy_threshold = finite_distances[easy_idx]
            medium_threshold = finite_distances[medium_idx]

        seat_difficulties = {}
        seats_by_difficulty = {'easy': [], 'medium': [], 'hard': []}

        for seat, dist in seat_path_distances.items():
            if not np.isfinite(dist):
                category = 'hard'
            elif dist <= easy_threshold:
                category = 'easy'
            elif dist <= medium_threshold:
                category = 'medium'
            else:
                category = 'hard'

            seat_difficulties[seat] = category
            seats_by_difficulty[category].append(seat)

        return seat_path_distances, seat_difficulties, seats_by_difficulty

    def _initialize_episode_seat_metrics(self):
        """1エピソード内の席別評価指標を初期化する。"""
        self.episode_customer_spawn_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.episode_order_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.episode_delivery_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.episode_unserved_order_heatmap = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._unserved_recorded_this_episode = False

    def _record_customer_spawn(self, seat_pos):
        if seat_pos is None:
            return
        x, y = seat_pos
        self.episode_customer_spawn_heatmap[x, y] += 1
        self.total_customer_spawn_heatmap[x, y] += 1

    def _record_order_created(self, seat_pos):
        if seat_pos is None:
            return
        x, y = seat_pos
        self.episode_order_heatmap[x, y] += 1
        self.total_order_heatmap[x, y] += 1

    def _record_delivery_completed(self, seat_pos):
        if seat_pos is None:
            return
        x, y = seat_pos
        self.episode_delivery_heatmap[x, y] += 1
        self.total_delivery_heatmap[x, y] += 1

    def _record_unserved_orders_at_episode_end(self):
        """エピソード終了時に、注文済みだが未配膳の席を記録する。"""
        if getattr(self, '_unserved_recorded_this_episode', False):
            return

        for customer in self.customer_manager.customers:
            if customer.state == 'ordered':
                x, y = customer.seat_position
                self.episode_unserved_order_heatmap[x, y] += 1
                self.total_unserved_order_heatmap[x, y] += 1

        self._unserved_recorded_this_episode = True

    def get_seat_performance_report(self, use_total=True):
        """
        席別・難易度別の評価指標を返す。

        Returns:
            dict: {
                'seat_rows': [...],
                'difficulty_summary': {...}
            }
        """
        if use_total:
            spawn_hm = self.total_customer_spawn_heatmap
            order_hm = self.total_order_heatmap
            delivery_hm = self.total_delivery_heatmap
            unserved_hm = self.total_unserved_order_heatmap
        else:
            spawn_hm = self.episode_customer_spawn_heatmap
            order_hm = self.episode_order_heatmap
            delivery_hm = self.episode_delivery_heatmap
            unserved_hm = self.episode_unserved_order_heatmap

        seat_rows = []
        difficulty_summary = {
            'easy': {'spawned': 0, 'ordered': 0, 'delivered': 0, 'unserved': 0},
            'medium': {'spawned': 0, 'ordered': 0, 'delivered': 0, 'unserved': 0},
            'hard': {'spawned': 0, 'ordered': 0, 'delivered': 0, 'unserved': 0},
        }

        for seat in self.seats:
            x, y = seat
            spawned = int(spawn_hm[x, y])
            ordered = int(order_hm[x, y])
            delivered = int(delivery_hm[x, y])
            unserved = int(unserved_hm[x, y])
            success_rate = delivered / ordered if ordered > 0 else 0.0
            category = self.seat_difficulties.get(seat, 'unknown')
            path_distance = self.seat_path_distances.get(seat, float('inf'))

            seat_rows.append({
                'seat': seat,
                'difficulty': category,
                'path_distance': path_distance,
                'spawned': spawned,
                'ordered': ordered,
                'delivered': delivered,
                'unserved': unserved,
                'success_rate': success_rate,
            })

            if category in difficulty_summary:
                difficulty_summary[category]['spawned'] += spawned
                difficulty_summary[category]['ordered'] += ordered
                difficulty_summary[category]['delivered'] += delivered
                difficulty_summary[category]['unserved'] += unserved

        for category, stats in difficulty_summary.items():
            ordered = stats['ordered']
            delivered = stats['delivered']
            stats['success_rate'] = delivered / ordered if ordered > 0 else 0.0

        return {
            'seat_rows': seat_rows,
            'difficulty_summary': difficulty_summary,
        }

    def format_seat_performance_report(self, use_total=True):
        """席別・難易度別の評価指標をログ出力用の文字列に整形する。"""
        report = self.get_seat_performance_report(use_total=use_total)
        lines = []

        lines.append("\n=== Seat Performance by Difficulty ===")
        lines.append("Difficulty | Spawned | Ordered | Delivered | Unserved | SuccessRate")
        lines.append("-----------|---------|---------|-----------|----------|------------")
        for category in ['easy', 'medium', 'hard']:
            stats = report['difficulty_summary'][category]
            lines.append(
                f"{category:10s} | "
                f"{stats['spawned']:7d} | "
                f"{stats['ordered']:7d} | "
                f"{stats['delivered']:9d} | "
                f"{stats['unserved']:8d} | "
                f"{stats['success_rate'] * 100:10.2f}%"
            )

        lines.append("\n=== Seat-wise Delivery Success Rate ===")
        lines.append("Seat       | Diff     | PathDist | Spawned | Ordered | Delivered | Unserved | SuccessRate")
        lines.append("-----------|----------|----------|---------|---------|-----------|----------|------------")

        sorted_rows = sorted(
            report['seat_rows'],
            key=lambda row: (self._difficulty_sort_key(row['difficulty']), row['path_distance'], row['seat'])
        )

        for row in sorted_rows:
            path_distance = row['path_distance']
            if np.isfinite(path_distance):
                dist_str = f"{int(path_distance):8d}"
            else:
                dist_str = "     inf"

            lines.append(
                f"{str(row['seat']):10s} | "
                f"{row['difficulty']:8s} | "
                f"{dist_str} | "
                f"{row['spawned']:7d} | "
                f"{row['ordered']:7d} | "
                f"{row['delivered']:9d} | "
                f"{row['unserved']:8d} | "
                f"{row['success_rate'] * 100:10.2f}%"
            )

        lines.append("======================================================")
        return "\n".join(lines)

    @staticmethod
    def _difficulty_sort_key(category):
        order = {'easy': 0, 'medium': 1, 'hard': 2}
        return order.get(category, 99)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        
        obstacles, tables, seats, counter_pos, entrance_pos = LayoutBuilder.create_layout(
            self.layout_type, self.grid_size)
        self.obstacles = obstacles
        self.tables = tables
        self.seats = seats
        self.counter_pos = counter_pos
        self.entrance_pos = entrance_pos

        # ★追加: 各席の最短経路距離と難易度カテゴリを計算
        (self.seat_path_distances,
         self.seat_difficulties,
         self.seats_by_difficulty) = self._compute_seat_difficulty_info()
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.grid[ox, oy] = -1
        
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

        # ★追加: このエピソード内の席別評価指標を初期化
        self._initialize_episode_seat_metrics()
        
        self.customer_manager.customers = []
        self.customer_manager.customer_counter = 0
        self.customer_manager.steps_since_last_spawn = 0
        
        self.history = [{
            'agent_positions': self.agent_positions.copy(),
            'agent_directions': self.agent_directions.copy(),
            'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
            'active_orders': self.active_orders.copy(),
            'agent_inventory': {k: v[:] for k, v in self.agent_inventory.items()},
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

        self._move_agents_simultaneously(actions, rewards)

        for agent in self.agents:
            if agent in actions:
                self._process_interaction(agent, actions[agent], rewards)

        self.customer_manager.steps_since_last_spawn += 1
        if self.customer_manager.steps_since_last_spawn >= self.customer_manager.spawn_interval:
            
            valid_seats = self.seats
            if self.counter_pos:
                cx, cy = self.counter_pos
                valid_seats = [
                    seat for seat in self.seats 
                    if self.min_customer_dist <= (abs(seat[0] - cx) + abs(seat[1] - cy)) <= self.max_customer_dist # ★適用
                ]
            
            if not valid_seats:
                valid_seats = self.seats

            new_customer = self.customer_manager.spawn_customer(
                self.entrance_pos, 
                valid_seats,
                counter_pos=self.counter_pos,
                min_dist_from_counter=0,
                seat_difficulties=self.seat_difficulties,
                spawn_mode=self.spawn_mode,
                difficulty_categories=self.difficulty_categories
            )
            if new_customer is not None:
                self._record_customer_spawn(new_customer.seat_position)

            self.customer_manager.steps_since_last_spawn = 0
        
        new_orders, new_kitchen = self.customer_manager.update_customers()
        self.active_orders.extend([o for o in new_orders if o not in self.active_orders])
        for order_pos in new_orders:
            self._record_order_created(order_pos)
        self.kitchen_queue.extend(new_kitchen)
        
        for item in self.kitchen_queue[:]:
            item['time_left'] -= 1
            if item['time_left'] <= 0:
                self.kitchen_queue.remove(item)
                self.ready_dishes.append({'food_type': item.get('food_type')})

        max_wait = self.reward_params.get('max_wait_limit', 50.0)
        scale = self.reward_params.get('wait_penalty_scale', 0.1)

        total_wait_penalty = 0.0
        for c in self.customer_manager.customers:
            if c.state == 'ordered':
                urgency = (c.wait_time / max_wait) ** 2
                urgency = min(urgency, 2.0)
                total_wait_penalty -= urgency * scale

        for agent in self.agents:
            rewards[agent] += total_wait_penalty
            rewards[agent] += self.reward_params['step_cost'] # ★追加: 生存ペナルティとして一律適用

        self.num_moves += 1
        if self.num_moves >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
            for c in self.customer_manager.customers:
                if c.state == 'ordered':
                    self.completed_wait_times.append(c.wait_time)
            self._record_unserved_orders_at_episode_end()
            self.agents = []

        observations = {}
        if self.agents:
            observations = {agent: self.observe(agent) for agent in self.agents}
        else:
             observations = {agent: self.observe(agent) for agent in self.possible_agents}

        if self.agents:
             self.history.append({
                'agent_positions': self.agent_positions.copy(),
                'agent_directions': self.agent_directions.copy(),
                'customers': [c.__dict__.copy() for c in self.customer_manager.customers],
                'active_orders': self.active_orders.copy(),
                'agent_inventory': {k: v[:] for k, v in self.agent_inventory.items()},
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
            
            if action == 0: 
                dx, dy = dir_vectors[curr_dir]
                temp_x = max(0, min(self.grid_size - 1, curr_pos[0] + dx))
                temp_y = max(0, min(self.grid_size - 1, curr_pos[1] + dy))
                next_pos = (temp_x, temp_y)
                
            elif action == 1: 
                next_dir = (curr_dir + 1) % 4
            elif action == 2: 
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
        
        if 4 <= action < 4 + 3 and is_near_counter and len(self.agent_inventory[agent]) < 4:
            requested_food_type = action - 4
            
            target_dish_idx = -1
            for idx, dish in enumerate(self.ready_dishes):
                if dish.get('food_type') == requested_food_type:
                    target_dish_idx = idx
                    break
            
            if target_dish_idx != -1:
                dish = self.ready_dishes.pop(target_dish_idx)
                self.agent_inventory[agent].append({'food_type': dish.get('food_type')})
                rewards[agent] += self.reward_params['pickup']
        
        if len(self.agent_inventory[agent]) > 0:
            for order_pos in self.active_orders[:]:
                adjacent = get_adjacent_positions(order_pos)
                if (x, y) in adjacent:
                    target_dish = None
                    
                    target_customer = next((c for c in self.customer_manager.customers if c.seat_position == order_pos), None)
                    
                    for dish in self.agent_inventory[agent]:
                        if target_customer and dish.get('food_type') == target_customer.order_type:
                            target_dish = dish
                            break
                    
                    if target_dish:
                        self.agent_inventory[agent].remove(target_dish)
                        
                        max_wait_limit = self.reward_params.get('max_wait_limit', 50.0)
                        
                        urgency_score = 0.0
                        distance_bonus = 0.0
                        
                        if target_customer:
                            # ① ボーナスの上限(min)を撤廃。待たせた分だけ青天井でボーナスが増える
                            urgency_score = target_customer.wait_time / max_wait_limit
                            
                            # ② 物理的な「遠距離手当」を導入
                            # マンハッタン距離ではなく、障害物を考慮した最短経路距離を使う。
                            cx, cy = self.counter_pos if self.counter_pos else (7, 1)
                            tx, ty = target_customer.seat_position
                            manhattan_dist = abs(tx - cx) + abs(ty - cy)
                            path_dist = self.seat_path_distances.get(target_customer.seat_position, manhattan_dist)
                            if not np.isfinite(path_dist):
                                path_dist = manhattan_dist
                            
                            # 1歩遠いごとに基本給に +5.0 点（割引率の減衰を相殺する）
                            distance_bonus = path_dist * 5.0

                        base_reward = self.reward_params['delivery']
                        bonus_scale = self.reward_params.get('urgency_bonus_scale', 10.0) # confs.pyで大きく設定していればそのまま適用
                        
                        # 遠いほど、そして待たせているほど爆発的な報酬が入るようにする
                        total_reward = base_reward + distance_bonus + (urgency_score * bonus_scale)
                        rewards[agent] += total_reward

                        self.served_count[agent] += 1
                        
                        self._record_delivery_completed(order_pos)
                        
                        if order_pos in self.active_orders:
                            self.active_orders.remove(order_pos)
                        
                        for customer in self.customer_manager.customers:
                            if customer.seat_position == order_pos and customer.state == 'ordered':
                                self.completed_wait_times.append(customer.wait_time)
                                customer.state = 'served'
                                customer.wait_time = 0
                                break
        
        inventory_count = len(self.agent_inventory[agent])
        if inventory_count > 0:
            rewards[agent] += (self.reward_params.get('holding_penalty', -0.01) * inventory_count)
            
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
        
        standard_obs = np.zeros(len(local_obs) + 12, dtype=np.float32)
        standard_obs[:len(local_obs)] = local_obs
        
        idx = len(local_obs)
        standard_obs[idx:idx+2] = [x / self.grid_size, y / self.grid_size]
        standard_obs[idx+2] = my_direction / 4.0
        standard_obs[idx+3] = len(self.active_orders) / self.max_seats_obs
        standard_obs[idx+4] = min(self.served_count[agent] / 50.0, 1.0)
        standard_obs[idx+5] = min(self.collision_count[agent] / 100.0, 1.0)
        standard_obs[idx+6] = len(self.agent_inventory[agent]) / 4.0
        standard_obs[idx+7] = 1.0 if len(self.agent_inventory[agent]) < 4 else 0.0
        standard_obs[idx+8] = min(len(self.ready_dishes), 5) / 5.0
        standard_obs[idx+9] = 1.0 if len(self.ready_dishes) > 0 else 0.0
        
        cx, cy = self.counter_pos if self.counter_pos else (7, 1)
        standard_obs[idx+10] = (cx - x) / self.grid_size
        standard_obs[idx+11] = (cy - y) / self.grid_size

        order_slots = []
        max_wait_limit = self.reward_params.get('max_wait_limit', 50.0)

        for i in range(self.max_seats_obs):
            if i < len(self.seats):
                seat_pos = self.seats[i]
                tx, ty = seat_pos
                
                c_obj = next((c for c in self.customer_manager.customers if c.seat_position == seat_pos and c.state == 'ordered'), None)
                
                urgency = min((c_obj.wait_time / max_wait_limit), 2.0) if c_obj else 0.0
                food_type = (c_obj.order_type / self.num_food_types) if c_obj else 0.0
                
                if c_obj:
                    target_food = c_obj.order_type
                    is_held = 1.0 if any(d.get('food_type') == target_food for d in self.agent_inventory[agent]) else 0.0
                    is_ready = 1.0 if any(d.get('food_type') == target_food for d in self.ready_dishes) else 0.0
                else:
                    is_held = 0.0
                    is_ready = 0.0
                
                order_slots.extend([
                    (tx - x) / self.grid_size, 
                    (ty - y) / self.grid_size, 
                    urgency, 
                    is_held, 
                    is_ready,
                    food_type 
                ])
            else:
                order_slots.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        full_obs = np.concatenate([standard_obs, np.array(order_slots, dtype=np.float32)])
        agent_idx = self.possible_agents.index(agent)
        agent_id_feature = np.zeros(self.n_agents, dtype=np.float32)
        agent_id_feature[agent_idx] = 1.0

        return np.concatenate([full_obs, agent_id_feature])

    def get_avail_actions(self):
        """
        各エージェントの有効な行動マスクを取得する
        戻り値: dict {agent_name: [1, 1, 1, 1, 0, 0, 0] のようなリスト}
        """
        avail_actions = {}
        for agent in self.possible_agents:
            if agent not in self.agent_positions:
                avail_actions[agent] = [1] * self.action_space(agent).n
                continue
                
            avail = [1] * self.action_space(agent).n
            x, y = self.agent_positions[agent]
            is_near_counter = False
            
            if self.counter_pos:
                cx, cy = self.counter_pos
                if abs(x - cx) + abs(y - cy) <= 1:
                    is_near_counter = True
            
            inventory_has_space = len(self.agent_inventory[agent]) < 4
            ready_food_types = {dish.get('food_type') for dish in self.ready_dishes}

            for food_type in range(3):
                action = 4 + food_type
                if (not is_near_counter or
                    not inventory_has_space or
                    food_type not in ready_food_types):
                    avail[action] = 0
            
            avail_actions[agent] = avail
            
        return avail_actions

